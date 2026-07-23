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

/// PCS params over the union address space: the committed buffer is the
/// full 2^{M−7}-word union buffer (the Phase 1/2 dense commit), so
/// `m = M`; rate, batch size, and profile match the single-type setups.
fn union_pcs_params(registry: &Registry) -> PcsParams {
    PcsParams {
        m: registry.m_total(),
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
    let pcs_params = union_pcs_params(&registry);

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

    // ---- The commitment is a commitment to the assembled union buffer:
    // regenerate the witnesses, assemble them independently, commit
    // directly, and compare roots.
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
    let (comm_direct, _prover_data) = flock_core::pcs::commit(&z_union, &pcs_params);
    assert_eq!(
        commitment.root, comm_direct.root,
        "commitment root must equal a direct commit of the assembled union buffer"
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
    // a diverged transcript from the first squeeze (the jagged heights and
    // const-pin targets would also mismatch downstream) — reject.
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
    let pcs_params = union_pcs_params(&registry);
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
    let pcs_params = union_pcs_params(&registry);
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
