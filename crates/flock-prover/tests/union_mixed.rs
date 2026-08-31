//! M3: the first end-to-end MIXED proofs — BLAKE3 and SHA-256 in one
//! statement, proved through the union entries under the real multi-table
//! binding (`flock-mixed-v1`: registry digest + counts vector + commitment
//! cap, design doc §"Statement, transcript, wire format").
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
//! byte-identity differentials against the direct jagged path lived in
//! `tests/union_roundtrip.rs` on the harness binding.

use bincode::deserialize;
use bincode::serialize;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::lincheck::LincheckError;
use flock_core::pcs::commit;
use flock_core::pcs::commit_lane_major;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::ligerito::embedded_initial_k_or_default;
use flock_core::pcs::{PcsOpenError, PcsParams};
use flock_core::proof::R1csProofMergedLigerito;
use flock_core::r1cs::BlockR1cs;
use flock_core::scratch::give_f128;
use flock_core::scratch::give_u8;
use flock_core::scratch::prewarm_prover;
use flock_core::union::SlotWitness;
use flock_core::verifier::FlockVerifyError;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression as Blake3Compression, K_LOG as BLAKE3_K_LOG,
    build_block_r1cs as build_blake3_block_r1cs,
    generate_witness_batch_major as generate_blake3_witness_batch_major,
    generate_witness_batch_major_partial as generate_blake3_witness_batch_major_partial,
    generate_witness_batch_major_partial_into as generate_blake3_witness_batch_major_partial_into,
};
use flock_prover::r1cs_hashes::sha2::{
    Compression as Sha2Compression, K_LOG as SHA2_K_LOG, Sha256HybridSetup,
    build_block_r1cs as build_sha2_block_r1cs,
    generate_witness_batch_major as generate_sha2_witness_batch_major,
    generate_witness_batch_major_partial as generate_sha2_witness_batch_major_partial,
    generate_witness_batch_major_partial_into as generate_sha2_witness_batch_major_partial_into,
};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier;
use prover::prove_fast_ligerito_union;
use prover::prove_fast_ligerito_union_mixed_class;
use std::array::from_fn;
use std::env::var;
use std::sync::Mutex;
use std::sync::MutexGuard;
use verifier::verify_ligerito_union;
use verifier::verify_ligerito_union_mixed_class;

use flock_core::test_rng::Rng;
use std::time::Instant;

const DOMAIN: &[u8] = b"flock-mixed-e2e-v0";

/// Serialize the TIMING tests against each other: `--ignored` runs the heavy
/// provers concurrently on the shared rayon pool, which inflates every
/// wall-clock reading (a single-shot arm measured 3x its quiet value) and
/// occasionally trips the loose timing gates. Correctness tests stay
/// parallel; only tests that assert or print wall times take this lock.
fn timing_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<Blake3Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

fn random_sha2_inputs(rng: &mut Rng, n: usize) -> Vec<Sha2Compression> {
    (0..n)
        .map(|_| (from_fn(|_| rng.next_u32()), from_fn(|_| rng.next_u32())))
        .collect()
}

/// The M3 mixed registry: the real SHA-256 (κ = 15) and BLAKE3 (κ = 14)
/// base blocks (via `TableType::from_block_r1cs` on the modules' block
/// R1CS) at uniform capacity 2^ν, fed in width-ASCENDING order to exercise
/// the registry's canonical sort. Slot order — capacity area descending,
/// under uniform capacity simply κ descending — is SHA-256 then BLAKE3,
/// and M = ν + 16.
fn mixed_registry(nu: usize) -> (Registry, BlockR1cs, BlockR1cs) {
    let sha2_r1cs = build_sha2_block_r1cs(nu);
    let blake3_r1cs = build_blake3_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&blake3_r1cs),
            TableType::from_block_r1cs(&sha2_r1cs),
        ],
        nu,
    );
    assert_eq!(
        registry.types()[0].k_log,
        SHA2_K_LOG,
        "slot order: SHA-256 (wider) first"
    );
    assert_eq!(registry.types()[1].k_log, BLAKE3_K_LOG);
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
        log_batch_size: embedded_initial_k_or_default(union.dense_m(), LigeritoProfile::Fast),
        profile: LigeritoProfile::Fast,
        // Integer-lane commit: skip the encode + hash of the whole zero lanes
        // the power-of-two rounding of the dense stack leaves behind.
        num_lanes: union.commit_lanes(embedded_initial_k_or_default(
            union.dense_m(),
            LigeritoProfile::Fast,
        )),
        merkle_hash: Default::default(),
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
    // capacity, M4's grid exactly): 339 of 512 chunk-columns used (SHA-256
    // 199/256, BLAKE3 92/128; the top gap grew with the lin-id drops and the
    // Option-F fused/const adders on both tables), a genuinely non-identity
    // compaction (BLAKE3's columns stack at 199, not 256), rounding back to
    // the padded word count at this ratio.
    assert!(!union.compaction_is_identity());
    assert_eq!(union.dense_words(), (199 + 92) << nu);
    assert_eq!(union.committed_words(), 1 << (union.m_total() - 7));

    let mut rng = Rng::new(0x03_31_2B_B3);
    let blake3_inputs = random_blake3_inputs(&mut rng, n_per_type);
    let sha2_inputs = random_sha2_inputs(&mut rng, n_per_type);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();

    // ---- Prove: per-slot inputs in slot order (SHA-256 first).
    let slots = vec![
        UnionSlotProverInput::new(
            generate_sha2_witness_batch_major(&sha2_inputs, nu),
            sha2_circuit,
        ),
        UnionSlotProverInput::new(
            generate_blake3_witness_batch_major(&blake3_inputs, nu),
            blake3_circuit,
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch_p);

    // ---- The commitment is a commitment to the COMPACTED union buffer
    // (the M4 dense stack): regenerate the witnesses, assemble them
    // independently, compact, commit directly, and compare roots. Also pin
    // that the compaction genuinely moved data: q differs from the padded
    // buffer's prefix (BLAKE3's slot stacks 10 columns lower).
    let (z_s, a_s, b_s, _) = generate_sha2_witness_batch_major(&sha2_inputs, nu);
    let (z_b, a_b, b_b, _) = generate_blake3_witness_batch_major(&blake3_inputs, nu);
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
    // Mirrors the prover's dispatch: the integer-lane commit encodes only the
    // dense stack's nonzero high-bit lanes (`UnionInstance::commit_lanes`).
    let (comm_direct, _prover_data) = if pcs_params.num_lanes.is_some() {
        commit_lane_major(&q, &pcs_params)
    } else {
        commit(&q, &pcs_params)
    };
    assert_eq!(
        commitment.cap, comm_direct.cap,
        "commitment cap must equal a direct commit of the compacted union stack"
    );

    // ---- Verify (circuits in slot order).
    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let verify = |union: &UnionInstance<'_>, proof: &R1csProofMergedLigerito| {
        let mut ch_v = FsChallenger::new(DOMAIN);
        verify_ligerito_union(union, &circuits, &commitment, proof, &pcs_params, &mut ch_v)
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

    // ---- Tamper: the commitment's `num_lanes`. The integer-lane commit puts
    // an attacker-controlled field on the verifier's path — `num_ntts()` sets
    // the L0 leaf width AND selects the lane-grid rotation — while the
    // transcript binds only the commitment ROOT. The honest value is
    // count-derived (`UnionInstance::commit_lanes`), so the verifier requires
    // the commitment to carry exactly it. Both directions must reject.
    {
        assert!(
            pcs_params.num_lanes.is_some(),
            "this shape must exercise the integer-lane commit"
        );
        for bad_lanes in [
            Some(pcs_params.num_ntts() - 1),
            Some(pcs_params.num_ntts() + 1),
            None, // "no integer-lane commit at all"
        ] {
            let mut bad_comm = commitment.clone();
            bad_comm.params.num_lanes = bad_lanes;
            let mut ch_v = FsChallenger::new(DOMAIN);
            assert!(
                verify_ligerito_union(
                    &union,
                    &circuits,
                    &bad_comm,
                    &proof,
                    &pcs_params,
                    &mut ch_v,
                )
                .is_err(),
                "tampered num_lanes ({bad_lanes:?}) must reject"
            );
        }
    }

    // ---- Tamper: PIOP (a lincheck round message) — rejects through the
    // existing union-lincheck error path.
    {
        let mut bad = proof.clone();
        bad.lincheck.rounds[0].0.lo ^= 1;
        match verify(&union, &bad) {
            Err(FlockVerifyError::Lincheck(LincheckError::ConsistencyFailed { .. })) => {}
            other => panic!(
                "tampered lincheck round: expected Lincheck(ConsistencyFailed), got {other:?}"
            ),
        }
    }

    // ---- Tamper: opening (the merged sumcheck's claimed `q_eval`) —
    // rejects through the existing opening error path (the merged final
    // check maps to the same `VirtualOpen` error class).
    {
        let mut bad = proof.clone();
        bad.pcs_open.q_eval.lo ^= 1;
        match verify(&union, &bad) {
            Err(FlockVerifyError::PcsOpen(PcsOpenError::VirtualOpen)) => {}
            other => panic!("tampered q_eval: expected PcsOpen(VirtualOpen), got {other:?}"),
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
                generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
                sha2_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
                blake3_circuit,
            ),
        ];
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch_p);

        let verify = |union: &UnionInstance<'_>| {
            let mut ch_v = FsChallenger::new(DOMAIN);
            verify_ligerito_union(
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

/// THE DENSE FLOOR (envelope wall 2's capability): the same content commits
/// at a PINNED size above its natural one — `max(next_pow2(content),
/// 2^(m*−7))` — extending the zero tail the pow2 rounding already pays. At
/// ν = 7, counts (32, 32) naturally commit 2^15 words (the m22 config
/// floor); floored to m* = 23 they commit 2^16 with a ~6× content gap, so
/// the high-bit-lane commit carries only the few active lanes — the seam's
/// recorded care point, exercised at a gap the rounding tax alone never
/// produces. The floor is STATEMENT data like the counts: a verifier
/// without it sizes the m22 config and must reject the m23 proof.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn dense_floor_roundtrip_and_statement_binding() {
    let nu = 7usize;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let mut rng = Rng::new(0x_F1_00_12_28);
    let counts = vec![32usize, 32];
    let sha2_inputs = random_sha2_inputs(&mut rng, 32);
    let blake3_inputs = random_blake3_inputs(&mut rng, 32);
    let slots = || {
        vec![
            UnionSlotProverInput::new(
                generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
                sha2_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
                blake3_circuit,
            ),
        ]
    };

    let natural = UnionInstance::new(&registry, counts.clone());
    assert_eq!(
        natural.dense_m(),
        22,
        "the natural size is the config floor"
    );

    let mut floored = UnionInstance::new(&registry, counts.clone());
    floored.set_dense_floor(23);
    assert_eq!(floored.dense_m(), 23, "the floor binds");
    assert_eq!(floored.committed_words(), 1 << 16);
    assert_eq!(
        floored.dense_words(),
        natural.dense_words(),
        "content is floor-independent"
    );
    let params_f = union_pcs_params(&floored);
    assert_eq!(params_f.m, 23);
    let lanes = params_f
        .num_lanes
        .expect("a 6x content gap leaves whole zero lanes");
    assert_eq!(
        lanes,
        floored
            .dense_words()
            .div_ceil(1 << (23 - 7 - params_f.log_batch_size)),
        "active lanes stay CONTENT-derived under the floor"
    );

    // The floored proof roundtrips at the pinned size.
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_f, commitment_f, claim_f) =
        prove_fast_ligerito_union(&floored, &params_f, slots(), &mut ch_p);
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verify_ligerito_union(
        &floored,
        &circuits,
        &commitment_f,
        &proof_f,
        &params_f,
        &mut ch_v,
    )
    .expect("the floored proof verifies at the floored size");
    assert_eq!(claim_v, claim_f);

    // The SAME content still roundtrips naturally — the floor is opt-in.
    let params_n = union_pcs_params(&natural);
    assert_eq!(params_n.m, 22);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof_n, commitment_n, _) =
        prove_fast_ligerito_union(&natural, &params_n, slots(), &mut ch_p);
    let mut ch_v = FsChallenger::new(DOMAIN);
    verify_ligerito_union(
        &natural,
        &circuits,
        &commitment_n,
        &proof_n,
        &params_n,
        &mut ch_v,
    )
    .expect("the natural proof still verifies");

    // Statement binding: a verifier WITHOUT the floor sizes the m22 config
    // against the m23 proof — reject, never accept-with-different-statement.
    let mut ch_v = FsChallenger::new(DOMAIN);
    assert!(
        verify_ligerito_union(
            &natural,
            &circuits,
            &commitment_f,
            &proof_f,
            &params_n,
            &mut ch_v,
        )
        .is_err(),
        "the floor is statement data: verifying without it must reject"
    );
}

/// M5 — THE area gate, end to end: at ν = 7 (M = 23, padded 2^16 words) a
/// partial-utilization mix at counts (32, 32) commits the height-`n_t`
/// dense stack of 32·(199 + 92) = 9 312 words → 2^15 committed words
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
    // The halving: dense 9 312 words → committed 2^15 (config floor;
    // next_pow2 alone would give 2^14) vs M4's capacity-height 2^16.
    assert_eq!(union.dense_words(), 9_312);
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
            generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
            sha2_circuit,
        ),
        UnionSlotProverInput::new(
            generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
            blake3_circuit,
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch_p);
    assert_eq!(
        commitment.params.m, 22,
        "the produced commitment must be to the 2^15-word dense stack"
    );

    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let verify = |union: &UnionInstance<'_>| {
        let mut ch_v = FsChallenger::new(DOMAIN);
        verify_ligerito_union(
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
            generate_blake3_witness_batch_major(&blake3_inputs, nu),
            blake3_r1cs.csc_lincheck_circuit(),
        ),
        UnionSlotProverInput::new(
            generate_sha2_witness_batch_major(&sha2_inputs, nu),
            sha2_r1cs.csc_lincheck_circuit(),
        ),
    ];
    let mut ch_p = FsChallenger::new(DOMAIN);
    let _ = prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch_p);
}

/// A single-type instance through the NEW binding roundtrips. The proof is
/// (correctly) NOT byte-identical to
/// the single-table direct path on the same statement +
/// witness: the `flock-mixed-v1` binding absorbs the registry digest + the
/// counts vector where the direct path absorbs the `BlockR1cs` statement
/// digest — domain-separated on purpose — so no byte-identity is (or ever
/// will be) asserted here. The byte-identity regression anchor is
/// `tests/union_roundtrip.rs`, which pins the harness binding.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn blake3_single_type_roundtrip_under_mixed_binding() {
    let n_blocks = 256usize;
    let setup = Blake3Setup::new(n_blocks);
    let mut rng = Rng::new(0x03_31_00_B3);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    let registry = Registry::new(
        vec![TableType::from_block_r1cs(&setup.r1cs)],
        setup.r1cs.n_log(),
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    let slot = UnionSlotProverInput::new(
        generate_blake3_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) =
        prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch_p);

    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verify_ligerito_union(
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

/// Identity compaction through the MERGED arm of `prove_union_with_binding`:
/// a single-slot registry at full utilization leaves `dense_q` as `None`
/// (q IS the padded buffer), which the merged arm must handle by cloning —
/// it used to `expect` and panic. The shipped standalone body always handled
/// this (`prove_fast_ligerito_union`); this pins the enum path
/// the mixed-class entries take.
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn identity_compaction_roundtrips_over_the_merged_transport() {
    let n_blocks = 256usize;
    let setup = Blake3Setup::new(n_blocks);
    let mut rng = Rng::new(0x03_31_1D_B3);
    let inputs = random_blake3_inputs(&mut rng, n_blocks);
    let lc_circuit = setup.r1cs.csc_lincheck_circuit();

    let registry = Registry::new(
        vec![TableType::from_block_r1cs(&setup.r1cs)],
        setup.r1cs.n_log(),
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    assert!(
        union.compaction_is_identity(),
        "single slot at full utilization must be the identity compaction"
    );
    let slot = UnionSlotProverInput::new(
        generate_blake3_witness_batch_major(&inputs, setup.n_blocks_log()),
        lc_circuit,
    );
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claims) = prove_fast_ligerito_union_mixed_class(
        &union,
        &setup.pcs_params,
        vec![slot],
        Vec::new(),
        &mut ch_p,
    );
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claims_v = verify_ligerito_union_mixed_class(
        &union,
        &[lc_circuit],
        &commitment,
        &proof,
        &setup.pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("merged rejected an honest identity-compaction proof: {e:?}"));
    assert_eq!(claims_v, claims);
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
    let _quiet = timing_lock();

    let nu = 10usize;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let mut rng = Rng::new(0x05_31_77_77);
    prewarm_prover(registry.m_total());

    let mut results = Vec::new();
    for counts in [[8usize, 8usize], [1024, 1024]] {
        let [n_sha2, n_blake3] = counts;
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        let sha2_inputs = random_sha2_inputs(&mut rng, n_sha2);
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);
        // One untimed warm-up (hot scratch pool), then timed runs — MIN of
        // three: the ignored tests in this binary run concurrently (the
        // capacity sweep is a heavy neighbor), so a single-shot wall time
        // occasionally inflates past the loose gate below.
        let mut prove_ms = f64::INFINITY;
        for timed in [false, true, true, true] {
            let slots = vec![
                UnionSlotProverInput::new(
                    generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
                    s2_circuit,
                ),
                UnionSlotProverInput::new(
                    generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
                    b3_circuit,
                ),
            ];
            let mut ch = FsChallenger::new(DOMAIN);
            let t = Instant::now();
            let (proof, commitment, claim) =
                prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
            if timed {
                prove_ms = prove_ms.min(t.elapsed().as_secs_f64() * 1e3);
                // Roundtrip while we're here.
                let circuits: [&dyn LincheckCircuit; 2] = [s2_circuit, b3_circuit];
                let mut ch_v = FsChallenger::new(DOMAIN);
                let claim_v = verify_ligerito_union(
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

/// The MERGED jagged/ring-switch transport (design doc §"Capacity-free
/// ring-switching") proves and verifies end to end on real mixed
/// instances — full utilization and non-power-of-two partial counts — and
/// rejects tampering with the transport's new pieces (the merged sumcheck
/// rounds, the claimed q̂(ρ), the Frobenius assist's V) as well as the
/// PIOP. Prototype path: pow2-lane commitments, side by side with the
/// jagged transport (which stays the shipped default).
#[test]
#[ignore] // Heavier — run with `cargo test -p flock-prover --test union_mixed -- --ignored`
fn merged_transport_roundtrip_and_tamper() {
    let nu = 6usize;
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let circuits: [&dyn LincheckCircuit; 2] = [s2_circuit, b3_circuit];
    let mut rng = Rng::new(0x_4E_26_ED_11);

    for (counts, integer_lanes) in [([64usize, 64], true), ([64, 64], false), ([50, 37], true)] {
        let union = UnionInstance::new(&registry, counts.to_vec());
        let mut pcs_params = union_pcs_params(&union);
        if !integer_lanes {
            pcs_params.num_lanes = None; // pow2-lane coverage
        }
        let sha2_inputs = random_sha2_inputs(&mut rng, counts[0]);
        let blake3_inputs = random_blake3_inputs(&mut rng, counts[1]);
        let slots = vec![
            UnionSlotProverInput::new(
                generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
                s2_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
                b3_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
        let mut ch_v = FsChallenger::new(DOMAIN);
        let claim_v = verify_ligerito_union(
            &union,
            &circuits,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("merged transport rejected honest proof {counts:?}: {e:?}"));
        assert_eq!(claim_v, claim);

        // Tamper matrix over the transport's new pieces + one PIOP field.
        let reject = |p: &R1csProofMergedLigerito, what: &str| {
            let mut ch_v = FsChallenger::new(DOMAIN);
            assert!(
                verify_ligerito_union(&union, &circuits, &commitment, p, &pcs_params, &mut ch_v,)
                    .is_err(),
                "tampered proof ({what}) must be rejected at counts {counts:?}"
            );
        };
        // Field-complete walk of MergedOpenProof: every field of the merged
        // open is hit at least once (this is the surviving transport's
        // tamper oracle once the jagged transport is removed).
        let mut bad = proof.clone();
        bad.pcs_open.ring_switches[0].s_hat_v[0] += F128::ONE;
        reject(&bad, "ring-switch s_hat_v");
        let mut bad = proof.clone();
        bad.pcs_open.merged_rounds[3].0 += F128::ONE;
        reject(&bad, "merged sumcheck round");
        let mut bad = proof.clone();
        bad.pcs_open.merged_rounds.pop();
        reject(&bad, "merged sumcheck truncation");
        let mut bad = proof.clone();
        bad.pcs_open.q_eval += F128::ONE;
        reject(&bad, "q_eval");
        let mut bad = proof.clone();
        bad.pcs_open.frobenius.values[0][0] += F128::ONE;
        reject(&bad, "frobenius V");
        let mut bad = proof.clone();
        bad.pcs_open.frobenius.rounds[5].1 += F128::ONE;
        reject(&bad, "frobenius round");
        let mut bad = proof.clone();
        bad.pcs_open.inner.ligerito.initial_cap = vec![[0u8; 32]];
        reject(&bad, "inner open initial cap (wrong size + content)");
        let mut bad = proof.clone();
        bad.pcs_open.inner.ligerito.initial_cap[0][0] ^= 1;
        reject(&bad, "inner open initial cap node");
        let mut bad = proof.clone();
        bad.pcs_open.inner.ligerito.recursive_caps[0][0][0] ^= 1;
        reject(&bad, "inner open recursive cap node");
        let mut bad = proof.clone();
        bad.pcs_open.inner.ligerito.recursive_caps[0].push([0u8; 32]);
        reject(&bad, "inner open recursive cap length");
        let mut bad = proof.clone();
        bad.zerocheck.round1_ab[0] += F128::ONE;
        reject(&bad, "zerocheck round 1");

        // Statement tampers, ported from the jagged roundtrip's matrix: the
        // binding absorbs registry digest + counts before any challenge.
        {
            let union_bad = UnionInstance::new(&registry, vec![counts[0], counts[1] - 1]);
            let mut ch_v = FsChallenger::new(DOMAIN);
            assert!(
                verify_ligerito_union(
                    &union_bad,
                    &circuits,
                    &commitment,
                    &proof,
                    &pcs_params,
                    &mut ch_v,
                )
                .is_err(),
                "wrong counts vector must be rejected on the merged transport"
            );
        }
        {
            // `useful_bits + 1` rounds to the same chunk-column count, so the
            // ONLY verifier-side divergence is the registry digest inside the
            // binding — isolating it as load-bearing (same probe as the
            // jagged matrix).
            let mut blake3_ty = TableType::from_block_r1cs(&blake3_r1cs);
            blake3_ty.useful_bits += 1;
            let registry_bad =
                Registry::new(vec![TableType::from_block_r1cs(&sha2_r1cs), blake3_ty], nu);
            assert_ne!(registry.digest(), registry_bad.digest());
            let union_bad = UnionInstance::new(&registry_bad, counts.to_vec());
            assert_eq!(union.jagged_heights(), union_bad.jagged_heights());
            let mut ch_v = FsChallenger::new(DOMAIN);
            assert!(
                verify_ligerito_union(
                    &union_bad,
                    &circuits,
                    &commitment,
                    &proof,
                    &pcs_params,
                    &mut ch_v,
                )
                .is_err(),
                "tampered registry digest must be rejected on the merged transport"
            );
        }

        // Params-vs-root binding: the transcript binds only the commitment
        // ROOT, but the opening reads the leaf width / lane count from
        // `commitment.params`, which rides the proof attacker-controlled —
        // the verifier must reject a commitment whose params differ from
        // the count-derived ones (same soundness surface the jagged path
        // tamper-tests).
        let mut bad_commitment = commitment.clone();
        bad_commitment.params.num_lanes = match commitment.params.num_lanes {
            Some(_) => None,
            None => Some((1usize << commitment.params.log_batch_size) - 1),
        };
        let mut ch_v = FsChallenger::new(DOMAIN);
        assert!(
            verify_ligerito_union(
                &union,
                &circuits,
                &bad_commitment,
                &proof,
                &pcs_params,
                &mut ch_v,
            )
            .is_err(),
            "tampered commitment params (num_lanes) must be rejected at counts {counts:?}"
        );
        let mut bad_commitment = commitment.clone();
        bad_commitment.params.log_inv_rate += 1;
        let mut ch_v = FsChallenger::new(DOMAIN);
        assert!(
            verify_ligerito_union(
                &union,
                &circuits,
                &bad_commitment,
                &proof,
                &pcs_params,
                &mut ch_v,
            )
            .is_err(),
            "tampered commitment params (log_inv_rate) must be rejected at counts {counts:?}"
        );
    }
}

/// The merged transport at the real m30 load, across capacity tiers — the
/// merged reduction's design claim is CAPACITY-FREE transport (the Φ-pass
/// runs on the dense domain), so its prove time must be near-flat from
/// ν = 14 to ν = 16. Informational (printed, one loose ABSOLUTE assertion —
/// the jagged comparator was removed with the jagged transport; its final
/// A/B numbers are quoted at the assertion). Runs the shipped commit
/// config (integer lanes); this IS the wire-v6 protocol.
#[test]
#[ignore] // Heavy + informational — run explicitly with --ignored --nocapture
fn merged_transport_m30_probe() {
    let _quiet = timing_lock();

    const COUNTS: [usize; 2] = [16384, 16384];
    const NUS: [usize; 3] = [14, 15, 16];

    let cfgs: Vec<_> = NUS.iter().map(|&nu| mixed_registry(nu)).collect();
    prewarm_prover(cfgs.last().unwrap().0.m_total());
    let mut rng = Rng::new(0x_4E_26_ED_30);
    let sha2_inputs = random_sha2_inputs(&mut rng, COUNTS[0]);
    let blake3_inputs = random_blake3_inputs(&mut rng, COUNTS[1]);

    let mut mins = [f64::INFINITY; NUS.len()];
    for pass in 0..3 {
        for (i, &nu) in NUS.iter().enumerate() {
            let (registry, sha2_r1cs, blake3_r1cs) = &cfgs[i];
            let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
            let b3_circuit = blake3_r1cs.csc_lincheck_circuit();
            let circuits: [&dyn LincheckCircuit; 2] = [s2_circuit, b3_circuit];
            let union = UnionInstance::new(registry, COUNTS.to_vec());
            let slots = vec![
                UnionSlotProverInput::in_place(
                    |dst| generate_sha2_witness_batch_major_partial_into(&sha2_inputs, nu, dst),
                    s2_circuit,
                ),
                UnionSlotProverInput::in_place(
                    |dst| generate_blake3_witness_batch_major_partial_into(&blake3_inputs, nu, dst),
                    b3_circuit,
                ),
            ];
            let mut ch = FsChallenger::new(DOMAIN);
            let t = Instant::now();
            let pcs_params = union_pcs_params(&union);
            let (proof, commitment, claim) =
                prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if pass == 0 {
                let mut ch_v = FsChallenger::new(DOMAIN);
                let claim_v = verify_ligerito_union(
                    &union,
                    &circuits,
                    &commitment,
                    &proof,
                    &pcs_params,
                    &mut ch_v,
                )
                .expect("merged m30 proof verifies");
                assert_eq!(claim_v, claim);
            }
            if pass > 0 {
                mins[i] = mins[i].min(ms);
            }
        }
    }
    println!("merged m30 probe, counts {COUNTS:?} (min of 2, ms, prove incl. witgen):");
    for (i, &nu) in NUS.iter().enumerate() {
        println!(
            "  nu = {nu} (M = {}): merged {:.1}",
            cfgs[i].0.m_total(),
            mins[i]
        );
    }
    // The capacity-free design claim, now as an ABSOLUTE flatness bound (the
    // jagged comparator is gone — its final A/B record, 2026-08-02 on this
    // box: nu=14/15/16 jagged 110.8/148.5/156.7 ms vs merged
    // 113.2/113.4/114.9 ms, i.e. merged grew 1.7 ms where jagged grew
    // 45.9 ms). The bound is generous against box noise but far below the
    // jagged path's measured growth.
    let merged_growth = mins[NUS.len() - 1] - mins[0];
    assert!(
        merged_growth < 25.0,
        "merged transport must be near-capacity-free: prove grew \
         {merged_growth:.1} ms from nu=14 to nu=16 (measured 1.7 ms at the \
         final A/B, jagged comparator grew 45.9 ms)"
    );
}

/// The merged pipeline's dropped words are UNREAD — the evidence for the
/// `PooledDirty` witness mode: proofs must be byte-identical whether the
/// pooled buffers arrive clean or POISONED (every consumer is
/// support-gated: zerocheck run-lists, the count-proportional union
/// lincheck, declared-only compaction, the precomputed-`s_hat_v` ring
/// switch). Covers partial counts and a zero-count slot.
#[test]
fn merged_padding_unread_poison_pool() {
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(7);
    let s2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_circuit = blake3_r1cs.csc_lincheck_circuit();
    for counts in [[50usize, 37], [8, 0], [128, 128], [128, 37]] {
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        let mut rng = Rng::new(0x_D1_47_00 ^ counts[0] as u64);
        let sha2_inputs = random_sha2_inputs(&mut rng, counts[0]);
        let blake3_inputs = random_blake3_inputs(&mut rng, counts[1]);
        let prove = || {
            let slots = vec![
                UnionSlotProverInput::in_place(
                    |dst| generate_sha2_witness_batch_major_partial_into(&sha2_inputs, 7, dst),
                    s2_circuit,
                ),
                UnionSlotProverInput::in_place(
                    |dst| generate_blake3_witness_batch_major_partial_into(&blake3_inputs, 7, dst),
                    b3_circuit,
                ),
            ];
            let mut ch = FsChallenger::new(DOMAIN);
            prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch)
        };
        let (p1, c1, cl1) = prove();
        // Poison the pools with exact-size buffers so `take` hands them out.
        let len = union.packed_len();
        let poison = F128::new(0xDEAD_BEEF_DEAD_BEEF, 0xDEAD_BEEF_DEAD_BEEF);
        for _ in 0..6 {
            give_f128(vec![poison; len]);
        }
        for _ in 0..4 {
            give_u8(vec![0xAD; 1 << 20]);
        }
        let (p2, c2, cl2) = prove();
        assert_eq!(c1.cap, c2.cap, "commitment must ignore dropped words");
        assert_eq!(p1, p2, "proof must be byte-identical under a poisoned pool");
        assert_eq!(cl1, cl2);
        let circuits: [&dyn LincheckCircuit; 2] = [s2_circuit, b3_circuit];
        let mut chv = FsChallenger::new(DOMAIN);
        verify_ligerito_union(&union, &circuits, &c2, &p2, &pcs_params, &mut chv)
            .expect("poison-pool proof verifies");
    }
}

/// In-place witness generation is BYTE-IDENTICAL to prebuilt + scatter.
///
/// [`UnionSlotProverInput::in_place`] hands each driver the slot's aligned
/// block of the padded union buffers instead of letting it allocate its own
/// — sound because a slot's BatchMajor word index `(c << nu) + row` plus
/// `o_t >> 7` IS its union word index, so a slot's local layout is literally
/// a contiguous union sub-block. Nothing about the witness VALUES changes,
/// so the whole proof (commitment root + every sub-proof) must match bit for
/// bit. The oracle for the copy-free assembly path; covers full utilization,
/// non-power-of-two partial counts, and a zero count (whose slot block is
/// then all dummy rows).
#[test]
fn in_place_generation_matches_prebuilt_byte_identical() {
    let nu = 6usize; // M = 22, the smallest embedded Ligerito config
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    let sha2_circuit = sha2_r1cs.csc_lincheck_circuit();
    let blake3_circuit = blake3_r1cs.csc_lincheck_circuit();
    let circuits: [&dyn LincheckCircuit; 2] = [sha2_circuit, blake3_circuit];
    let mut rng = Rng::new(0x_1A_CE_2B_B3);

    for counts in [[64usize, 64], [50, 37], [0, 64]] {
        let union = UnionInstance::new(&registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        let sha2_inputs = random_sha2_inputs(&mut rng, counts[0]);
        let blake3_inputs = random_blake3_inputs(&mut rng, counts[1]);

        let prove = |slots: Vec<UnionSlotProverInput<'_>>| {
            let mut ch = FsChallenger::new(DOMAIN);
            let (proof, commitment, _claim) =
                prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
            (serialize(&proof).expect("proof serializes"), commitment)
        };

        let (prebuilt_bytes, prebuilt_comm) = prove(vec![
            UnionSlotProverInput::new(
                generate_sha2_witness_batch_major_partial(&sha2_inputs, nu),
                sha2_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major_partial(&blake3_inputs, nu),
                blake3_circuit,
            ),
        ]);
        let (in_place_bytes, in_place_comm) = prove(vec![
            UnionSlotProverInput::in_place(
                |dst| generate_sha2_witness_batch_major_partial_into(&sha2_inputs, nu, dst),
                sha2_circuit,
            ),
            UnionSlotProverInput::in_place(
                |dst| generate_blake3_witness_batch_major_partial_into(&blake3_inputs, nu, dst),
                blake3_circuit,
            ),
        ]);

        assert_eq!(
            prebuilt_comm.cap, in_place_comm.cap,
            "commitment root differs at counts {counts:?}"
        );
        assert_eq!(
            prebuilt_bytes, in_place_bytes,
            "proof bytes differ at counts {counts:?}"
        );

        // ...and the in-place proof really verifies (not just "same bytes").
        let mut ch = FsChallenger::new(DOMAIN);
        let proof: R1csProofMergedLigerito =
            deserialize(&in_place_bytes).expect("proof deserializes");
        verify_ligerito_union(
            &union,
            &circuits,
            &in_place_comm,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .unwrap_or_else(|e| panic!("in-place proof rejected at counts {counts:?}: {e:?}"));
    }
}

/// M6 — the m = 30-scale MIXED throughput measurement (Phase-1 gate at
/// production scale, the mixed-union analogue of `jagged_throughput`'s
/// BLAKE3 m = 30 sweep). ν = 14 puts the union at M = 30; at FULL
/// utilization (16384, 16384) the count-proportional dense stack —
/// (199 + 92)·2^14 = 4 767 744 words — rounds back up to the full 2^23-word
/// padded commit, i.e. `dense_m = 30` (the embedded m30 Ligerito config; the
/// count-proportional shrink only bites below full utilization). The two
/// single-type baselines run BLAKE3 (16384 blocks, m = 28) and SHA-256
/// (16384 compressions, m = 29) as single-slot MERGED unions at their
/// natural sizes. Warm-up + best-of-2 per path; the timed region is prove
/// INCLUDING witness generation, matching `jagged_throughput`'s accounting.
/// Big buffers are dropped between the three measurements — this is a
/// ~2 GB-scale run. Informational: prints the mixed headline, the singles
/// sum and mixed/sum ratio, per-type and combined invocations/sec for the
/// mixed proof, verify time, proof size, and committed-vs-padded words. No
/// timing assertions. Run with `cargo test --release -p flock-prover --test
/// union_mixed -- --ignored --nocapture mixed_m30_throughput`.
#[test]
#[ignore] // Heavy (M = 30, ~2 GB) + informational — run explicitly with --ignored --nocapture
fn mixed_m30_throughput() {
    let _quiet = timing_lock();

    // Timed runs after one warm-up; best reported. `MIXED_ITERS` overrides
    // (phase-median A/Bs want 5+).
    let iters: usize = var("MIXED_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    // M = nu + 16; full utilization = 2^nu invocations per type. Default
    // nu 14 (M = 30); `MIXED_NU=16` benches the M = 32 point.
    let nu: usize = var("MIXED_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(14);
    let n_per_type = 1usize << nu;
    let mut rng = Rng::new(0x30_31_2B_B3);
    let blake3_inputs = random_blake3_inputs(&mut rng, n_per_type);
    let sha2_inputs = random_sha2_inputs(&mut rng, n_per_type);

    // ---- Single-type BLAKE3 baseline as a single-slot merged union (m = 28).
    // One untimed warm-up (hot scratch pool), then best-of-ITERS timed. The
    // setup and its buffers drop at the end of the block.
    let (b3_ms, b3_m) = {
        let setup = Blake3Setup::new(n_per_type);
        assert_eq!(setup.n_blocks_log(), nu);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(vec![TableType::from_block_r1cs(&setup.r1cs)], nu);
        let union = UnionInstance::new(&registry, vec![n_per_type]);
        {
            let slot = UnionSlotProverInput::new(
                generate_blake3_witness_batch_major(&blake3_inputs, nu),
                circuit,
            );
            let mut ch = FsChallenger::new(DOMAIN);
            let _ = prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let mut ch = FsChallenger::new(DOMAIN);
            let t = Instant::now();
            let slot = UnionSlotProverInput::new(
                generate_blake3_witness_batch_major(&blake3_inputs, nu),
                circuit,
            );
            let _ = prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (best, setup.m())
    };

    // ---- Single-type SHA-256 baseline as a single-slot merged union
    // (m = 29).
    let (s2_ms, s2_m) = {
        let setup = Sha256HybridSetup::new(n_per_type);
        assert_eq!(setup.n_blocks_log(), nu);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let registry = Registry::new(vec![TableType::from_block_r1cs(&setup.r1cs)], nu);
        let union = UnionInstance::new(&registry, vec![n_per_type]);
        {
            let slot = UnionSlotProverInput::new(
                generate_sha2_witness_batch_major(&sha2_inputs, nu),
                circuit,
            );
            let mut ch = FsChallenger::new(DOMAIN);
            let _ = prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let mut ch = FsChallenger::new(DOMAIN);
            let t = Instant::now();
            let slot = UnionSlotProverInput::new(
                generate_sha2_witness_batch_major(&sha2_inputs, nu),
                circuit,
            );
            let _ = prove_fast_ligerito_union(&union, &setup.pcs_params, vec![slot], &mut ch);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (best, setup.m())
    };

    // ---- The mixed union at the same per-type sizes (M = 30). The proof,
    // commitment, and claim from the last timed run survive the block for
    // verification and size reporting; the witness buffers are consumed each
    // iteration and drop inside prove.
    let (registry, sha2_r1cs, blake3_r1cs) = mixed_registry(nu);
    assert_eq!(registry.m_total(), nu + 16);
    let union = UnionInstance::new(&registry, vec![n_per_type, n_per_type]);
    let pcs_params = union_pcs_params(&union);
    // Full utilization: the dense stack rounds back to the padded commit, so
    // dense_m lands on the embedded m30 Ligerito config.
    assert_eq!(union.dense_words(), (199 + 92) << nu);
    assert_eq!(union.dense_m(), nu + 16);
    assert_eq!(union.committed_words(), union.packed_len());
    assert_eq!(pcs_params.m, nu + 16);
    prewarm_prover(registry.m_total());
    let s2_mix_circuit = sha2_r1cs.csc_lincheck_circuit();
    let b3_mix_circuit = blake3_r1cs.csc_lincheck_circuit();
    {
        let slots = vec![
            UnionSlotProverInput::new(
                generate_sha2_witness_batch_major(&sha2_inputs, nu),
                s2_mix_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major(&blake3_inputs, nu),
                b3_mix_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let _ = prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
    }
    let mut mixed_ms = f64::INFINITY;
    let mut mixed_out = None;
    for _ in 0..iters {
        let slots = vec![
            UnionSlotProverInput::new(
                generate_sha2_witness_batch_major(&sha2_inputs, nu),
                s2_mix_circuit,
            ),
            UnionSlotProverInput::new(
                generate_blake3_witness_batch_major(&blake3_inputs, nu),
                b3_mix_circuit,
            ),
        ];
        let mut ch = FsChallenger::new(DOMAIN);
        let t = Instant::now();
        let out = prove_fast_ligerito_union(&union, &pcs_params, slots, &mut ch);
        mixed_ms = mixed_ms.min(t.elapsed().as_secs_f64() * 1e3);
        mixed_out = Some(out);
    }
    let (proof, commitment, claim) = mixed_out.unwrap();

    // ---- Verify (circuits in slot order: SHA-256 then BLAKE3).
    let circuits: [&dyn LincheckCircuit; 2] = [s2_mix_circuit, b3_mix_circuit];
    let t = Instant::now();
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verify_ligerito_union(
        &union,
        &circuits,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("mixed m=30 verifier rejected honest proof: {e:?}"));
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(claim_v, claim);
    let proof_bytes = serialize(&proof).unwrap().len();

    // ---- Report.
    let singles = b3_ms + s2_ms;
    let mixed_s = mixed_ms / 1e3;
    let per_type_hps = n_per_type as f64 / mixed_s;
    let combined_hps = (2 * n_per_type) as f64 / mixed_s;
    println!(
        "mixed m=30 throughput, {n_per_type} invocations per type (full util), \
         best-of-{iters} (prove incl. witgen):"
    );
    println!("  blake3-only jagged (m = {b3_m}): {b3_ms:.0} ms");
    println!("  sha2-only jagged   (m = {s2_m}): {s2_ms:.0} ms");
    println!(
        "  mixed union        (M = {}): {mixed_ms:.0} ms   <-- headline",
        registry.m_total()
    );
    println!(
        "  singles sum {singles:.0} ms; mixed / sum = {:.2}",
        mixed_ms / singles
    );
    println!(
        "  mixed invocations/sec: blake3 {per_type_hps:.0}, sha2 {per_type_hps:.0}, \
         combined {combined_hps:.0}"
    );
    println!("  verify: {verify_ms:.1} ms");
    println!("  proof size: {proof_bytes} B");
    println!(
        "  committed 2^{} words vs padded 2^{} words",
        union.committed_words().trailing_zeros(),
        union.packed_len().trailing_zeros()
    );
}
