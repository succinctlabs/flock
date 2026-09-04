use super::*;

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::prover::prove_ligerito;
use flock_core::challenger::FsChallenger;
use flock_core::circuit::boolean::{PortDirection, PortEncoding};
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;

const SHA256_ABC: [u32; 8] = [
    0xba78_16bf,
    0x8f01_cfea,
    0x4141_40de,
    0x5dae_2223,
    0xb003_61a3,
    0x9617_7a9c,
    0xb410_ff61,
    0xf200_15ad,
];

fn median_time(iterations: usize, mut operation: impl FnMut()) -> Duration {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[iterations / 2]
}

fn assert_matrix_rows_equal(side: &str, actual: &[Vec<usize>], expected: &[Vec<usize>]) {
    assert_eq!(actual.len(), expected.len());
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{side} differs at row {row}");
    }
}

#[test]
#[ignore = "builds and compares large SHA artifacts; run explicitly"]
fn dsl_matches_legacy_relation_and_witness() {
    let dsl = sha256_circuit();
    let actual_r1cs = sha256_relation_projection(3);
    assert!(Arc::ptr_eq(&actual_r1cs, &sha256_relation_projection(3)));
    let expected_r1cs = sha2::build_block_r1cs(3);

    assert_eq!(actual_r1cs.m, expected_r1cs.m);
    assert_eq!(actual_r1cs.k_log, expected_r1cs.k_log);
    assert_eq!(actual_r1cs.k_skip, expected_r1cs.k_skip);
    assert_eq!(actual_r1cs.useful_bits, expected_r1cs.useful_bits);
    assert_eq!(actual_r1cs.layout, expected_r1cs.layout);
    assert_eq!(actual_r1cs.const_pin, expected_r1cs.const_pin);
    assert_matrix_rows_equal("A", &actual_r1cs.a_0.rows, &expected_r1cs.a_0.rows);
    assert_matrix_rows_equal("B", &actual_r1cs.b_0.rows, &expected_r1cs.b_0.rows);
    assert_matrix_rows_equal("C", &actual_r1cs.c_0.rows, &expected_r1cs.c_0.rows);
    assert_eq!(
        actual_r1cs.statement_digest(),
        expected_r1cs.statement_digest()
    );
    drop(expected_r1cs);

    for (name, direction, len, position) in [
        (
            H_PORT,
            PortDirection::Input,
            sha2::H_WORDS * sha2::WORD_BITS,
            sha2::H_BASE,
        ),
        (
            MESSAGE_PORT,
            PortDirection::Input,
            sha2::M_WORDS * sha2::WORD_BITS,
            sha2::M_BASE,
        ),
        (
            H_OUT_PORT,
            PortDirection::Output,
            sha2::N_OUT_WORDS * sha2::WORD_BITS,
            sha2::H_OUT_BASE,
        ),
    ] {
        let port = dsl.circuit().port(name).unwrap();
        assert_eq!(port.direction(), direction);
        assert_eq!(port.values().len(), len);
        assert_eq!(
            port.encoding(),
            PortEncoding::LittleEndianWord {
                alignment_bits: sha2::SLOT_BITS,
            }
        );
        for (offset, &value) in port.values().iter().enumerate() {
            assert_eq!(dsl.layout().value_position(value), Some(position + offset));
        }
    }
    assert_eq!(
        dsl.layout().value_position(dsl.circuit().one()),
        Some(sha2::Z_CONST_POS)
    );

    let plan = sha256_walk_projection();
    assert!(std::ptr::eq(plan, sha256_walk_projection()));
    assert!(plan.c_is_identity());
    assert_eq!(dsl.circuit().expression_count(), 116_982);
    assert_eq!(plan.stats().actions, 116_981);
    assert_eq!(plan.stats().structural_edges, 285_800);
    assert!(plan.stats().max_live_temporaries < 2_000);
    assert_eq!(plan.stats().xor_term_allocations, plan.stats().xor_nodes);
    assert!(plan.stats().action_bytes < 16 * 1024 * 1024);
    assert!(plan.stats().xor_term_bytes < 4 * 1024 * 1024);
    assert!(dsl.circuit().normalized_support_bytes() < 320 * 1024 * 1024);

    let mut rng = flock_core::test_rng::Rng::new(0x5A25_6D51);
    let cases = [
        ([0u32; 8], [0u32; 16]),
        (sha2::SHA256_IV, [0u32; 16]),
        (
            sha2::SHA256_IV,
            [
                0x6162_6380,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0x0000_0018,
            ],
        ),
        (
            std::array::from_fn(|i| 0x1020_3040u32.wrapping_mul(i as u32 + 1)),
            std::array::from_fn(|i| 0xA5A5_5A5Au32.rotate_left(i as u32)),
        ),
        (
            std::array::from_fn(|_| rng.next_u32()),
            std::array::from_fn(|_| rng.next_u32()),
        ),
    ];
    let mut active_witness = None;
    for (case, &(h_in, message)) in cases.iter().enumerate() {
        let actual_witness = dsl.evaluate_block(&h_in, &message);
        let expected_witness = sha2::build_block_witness(&h_in, &message);
        assert_eq!(actual_witness, expected_witness);
        assert_eq!(
            pcs::pack_witness(&actual_witness, sha2::K_LOG),
            pcs::pack_witness(&expected_witness, sha2::K_LOG)
        );
        let walked = plan
            .forward(&block_inputs(&h_in, &message), sha2::K_LOG)
            .expect("the structural walk must evaluate SHA-256");
        assert_eq!(walked.z, actual_witness);
        let batched_witness: Vec<bool> = actual_witness.repeat(1 << 3);
        assert!(actual_r1cs.satisfies(&batched_witness));
        let h_out = sha2::read_h_out(&actual_witness);
        assert_eq!(h_out, sha2::sha256_compress(&h_in, &message));
        if case == 2 {
            assert_eq!(h_out, SHA256_ABC);
            active_witness = Some(actual_witness);
        }
    }

    // Standalone block padding contains complete valid compressions because
    // the statement pins ONE in every block.
    let active_witness = active_witness.unwrap();
    let zero_padding = dsl.evaluate_block(&[0; 8], &[0; 16]);
    let mut partially_filled = active_witness.clone();
    for _ in 1..1 << 3 {
        partially_filled.extend_from_slice(&zero_padding);
    }
    assert!(actual_r1cs.satisfies(&partially_filled));

    let mut invalid_column_padding = active_witness.repeat(1 << 3);
    invalid_column_padding[sha2::USEFUL_BITS] = true;
    assert!(!actual_r1cs.satisfies(&invalid_column_padding));

    // The union's partial-count format has a different contract: active rows
    // equal the DSL walk, while inactive rows are entirely zero.
    let partial_inputs = &cases[..2];
    let (partial_z, partial_a, partial_b, _) =
        sha2::generate_witness_batch_major_partial(partial_inputs, 3);
    let chunks_per_block = sha2::K / 128;
    for (block, (h_in, message)) in partial_inputs.iter().enumerate() {
        let walked = plan
            .forward(&block_inputs(h_in, message), sha2::K_LOG)
            .unwrap();
        for (actual, expected) in [
            (&partial_z, pcs::pack_witness(&walked.z, sha2::K_LOG)),
            (&partial_a, pcs::pack_witness(&walked.a_z, sha2::K_LOG)),
            (&partial_b, pcs::pack_witness(&walked.b_z, sha2::K_LOG)),
        ] {
            for chunk in 0..chunks_per_block {
                assert_eq!(actual[(chunk << 3) + block], expected[chunk]);
            }
        }
    }
    for values in [&partial_z, &partial_a, &partial_b] {
        for block in partial_inputs.len()..1 << 3 {
            for chunk in 0..chunks_per_block {
                assert_eq!(values[(chunk << 3) + block], F128::ZERO);
            }
        }
    }

    let const_pin = actual_r1cs.const_pin;
    drop(actual_r1cs);
    let adapter = plan
        .lincheck_circuit(sha2::K_LOG)
        .expect("SHA walk fits its base dimension");
    assert_eq!(adapter.const_pin_col(), const_pin);
    let legacy = sha2::Sha2LincheckCircuit;
    let alpha = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
    let eq: Vec<F128> = (0..sha2::K)
        .map(|i| F128::new(i as u64, (i as u64).rotate_left(17)))
        .collect();
    assert_eq!(adapter.const_pin_col(), Some(sha2::Z_CONST_POS));
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &eq),
        legacy.fold_alpha_batched(alpha, &eq)
    );
}

#[test]
#[ignore = "full m=22 Ligerito proof over the large SHA DSL artifact"]
fn dsl_relation_proves_and_verifies_with_walk_adapter() {
    let dsl = sha256_circuit();
    let r1cs = dsl.to_block_r1cs(7);
    let one_block = dsl.evaluate_block(&sha2::SHA256_IV, &[0u32; 16]);
    let witness = one_block.repeat(1 << 7);
    assert!(r1cs.satisfies(&witness));

    let pcs_params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let mut prover_challenger = FsChallenger::new(b"sha256-dsl-proof-v0");
    let (proof, commitment, prover_claim) = prove_ligerito(
        &r1cs,
        pcs::pack_witness(&witness, r1cs.m),
        &pcs_params,
        &mut prover_challenger,
    );

    let plan = dsl.walk_plan();
    let adapter = plan
        .lincheck_circuit(sha2::K_LOG)
        .expect("SHA walk fits its base dimension");
    assert_eq!(adapter.const_pin_col(), r1cs.const_pin);
    let mut verifier_challenger = FsChallenger::new(b"sha256-dsl-proof-v0");
    let verifier_claim = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &adapter,
        &pcs_params,
        &mut verifier_challenger,
    )
    .expect("the SHA DSL proof must verify through the walk adapter");
    assert_eq!(prover_claim, verifier_claim);
}

/// Run with:
/// `/usr/bin/time -v cargo +stable test --release -p flock-prover --lib \
/// r1cs_hashes::sha2_dsl::tests::performance_profile -- --ignored --exact --nocapture`
#[test]
#[ignore = "diagnostic Phase 4 profile; run alone with --nocapture"]
fn performance_profile() {
    let start = Instant::now();
    let dsl = sha256_circuit();
    let dsl_build = start.elapsed();

    let start = Instant::now();
    let plan = dsl.walk_plan();
    let plan_build = start.elapsed();

    let start = Instant::now();
    let relation = dsl.to_block_r1cs(3);
    black_box(relation.statement_digest());
    let dsl_lower = start.elapsed();
    drop(relation);

    let start = Instant::now();
    let relation = sha2::build_block_r1cs(3);
    black_box(relation.statement_digest());
    let legacy_setup = start.elapsed();
    drop(relation);

    let h_in = sha2::SHA256_IV;
    let message = [0u32; 16];
    let inputs = block_inputs(&h_in, &message);
    let reference_witness = dsl.evaluate_block(&h_in, &message);
    let walked = plan.forward(&inputs, sha2::K_LOG).unwrap();
    let optimized_witness = sha2::build_block_witness(&h_in, &message);
    assert_eq!(walked.z, reference_witness);
    assert_eq!(optimized_witness, reference_witness);

    let reference_eval = median_time(3, || {
        black_box(dsl.evaluate_block(&h_in, &message));
    });
    let forward_walk = median_time(21, || {
        black_box(plan.forward(&inputs, sha2::K_LOG).unwrap());
    });
    let optimized_eval = median_time(21, || {
        black_box(sha2::build_block_witness(&h_in, &message));
    });

    let alpha = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
    let eq: Vec<F128> = (0..sha2::K)
        .map(|i| F128::new(i as u64, (i as u64).rotate_left(17)))
        .collect();
    let adapter = plan.lincheck_circuit(sha2::K_LOG).unwrap();
    let legacy = sha2::Sha2LincheckCircuit;
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &eq),
        legacy.fold_alpha_batched(alpha, &eq)
    );
    let reverse_walk = median_time(21, || {
        black_box(adapter.fold_alpha_batched(alpha, &eq));
    });
    let legacy_reverse = median_time(21, || {
        black_box(legacy.fold_alpha_batched(alpha, &eq));
    });

    println!("SHA-256 Phase 4 profile (median unless noted):");
    println!("  DSL build (once):       {dsl_build:?}");
    println!("  walk-plan build (once): {plan_build:?}");
    println!("  DSL sparse lowering:    {dsl_lower:?}");
    println!("  legacy sparse setup:    {legacy_setup:?}");
    println!("  DSL reference witness:  {reference_eval:?}");
    println!("  structural forward:     {forward_walk:?}");
    println!("  optimized witness:      {optimized_eval:?}");
    println!("  structural reverse:     {reverse_walk:?}");
    println!("  legacy reverse:         {legacy_reverse:?}");
    println!(
        "  normalized supports:    {} terms / {} bytes",
        dsl.circuit().normalized_support_terms(),
        dsl.circuit().normalized_support_bytes()
    );
    println!("  walk storage:           {:?}", plan.stats());
}
