use super::*;

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use super::super::dsl::tests::{assert_relation_equal, median_time};

use crate::prover::prove_ligerito;
use flock_core::challenger::FsChallenger;
use flock_core::circuit::boolean::{LoweringMode, PortDirection, PortEncoding, RowPlacement};
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;

const EMPTY_FLAGS: u32 = (1 << 0) | (1 << 1) | (1 << 3);
const BLAKE3_EMPTY: [u8; 32] = [
    0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
    0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62,
];

fn block_inputs(
    cv: &[u32; 8],
    message: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> Vec<bool> {
    let mut inputs = Vec::with_capacity(28 * blake3::WORD_BITS);
    for &word in cv.iter().chain(message) {
        inputs.extend((0..blake3::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs.extend((0..64).map(|bit| counter >> bit & 1 == 1));
    for word in [block_len, flags] {
        inputs.extend((0..blake3::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs
}

fn read_word(witness: &[bool], base: usize) -> u32 {
    (0..blake3::WORD_BITS).fold(0, |word, bit| {
        word | (u32::from(witness[base + bit]) << bit)
    })
}

fn read_output(witness: &[bool]) -> [u32; 16] {
    std::array::from_fn(|word| {
        let base = if word < 8 {
            blake3::OUT_LO_BASE + word * blake3::WORD_BITS
        } else {
            blake3::OUT_HI_BASE + (word - 8) * blake3::WORD_BITS
        };
        read_word(witness, base)
    })
}

fn output_bytes(output: &[u32; 16]) -> [u8; 32] {
    std::array::from_fn(|i| output[i / 4].to_le_bytes()[i % 4])
}

fn cases() -> [blake3::Compression; 4] {
    let mut rng = flock_core::test_rng::Rng::new(0xB1A3_5EED);
    [
        ([0; 8], [0; 16], 0, 0, 0),
        (blake3::BLAKE3_IV, [0; 16], 0, 0, EMPTY_FLAGS),
        (
            std::array::from_fn(|i| 0x1020_3040u32.wrapping_mul(i as u32 + 1)),
            std::array::from_fn(|i| 0xA5A5_5A5Au32.rotate_left(i as u32)),
            0x0123_4567_89ab_cdef,
            0xfedc_ba98,
            0xa5a5_5a5a,
        ),
        (
            std::array::from_fn(|_| rng.next_u32()),
            std::array::from_fn(|_| rng.next_u32()),
            ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            rng.next_u32(),
            rng.next_u32(),
        ),
    ]
}

fn assert_batch_major_block(
    actual: (&[F128], &[F128], &[F128]),
    block: usize,
    n_blocks_log: usize,
    expected: &flock_core::circuit::boolean::ForwardTrace,
) {
    let packed = [
        pcs::pack_witness(&expected.z, blake3::K_LOG),
        pcs::pack_witness(&expected.a_z, blake3::K_LOG),
        pcs::pack_witness(&expected.b_z, blake3::K_LOG),
    ];
    for (actual, expected) in [actual.0, actual.1, actual.2].into_iter().zip(packed) {
        for (chunk, expected) in expected.into_iter().enumerate() {
            assert_eq!(actual[(chunk << n_blocks_log) + block], expected);
        }
    }
}

#[test]
#[ignore = "builds and compares large BLAKE3 artifacts; run explicitly"]
fn dsl_matches_legacy_relation_and_witness() {
    let dsl = blake3_circuit();
    let actual_r1cs = blake3_relation_projection(3);
    assert!(Arc::ptr_eq(&actual_r1cs, &blake3_relation_projection(3)));
    let expected_r1cs = blake3::build_block_r1cs(3);

    assert_relation_equal("legacy", &actual_r1cs, &expected_r1cs);
    drop(expected_r1cs);

    let identity = dsl
        .circuit()
        .lower(
            LoweringMode::RequireIdentityC,
            dsl.layout(),
            RowPlacement::Preserve,
        )
        .unwrap();
    assert!(identity.c_is_identity());
    assert!(identity.auxiliaries().is_empty());
    assert_eq!(identity.layout(), dsl.layout());
    let identity_r1cs = identity
        .to_block_r1cs(blake3::K_LOG, blake3::K_SKIP, 3)
        .unwrap();
    assert_relation_equal("required identity C", &identity_r1cs, &actual_r1cs);
    drop(identity_r1cs);

    for (name, direction, len, position, alignment_bits) in [
        (
            CV_PORT,
            PortDirection::Input,
            8 * blake3::WORD_BITS,
            blake3::CV_BASE,
            blake3::SLOT_BITS,
        ),
        (
            MESSAGE_PORT,
            PortDirection::Input,
            16 * blake3::WORD_BITS,
            blake3::M_BASE,
            blake3::SLOT_BITS,
        ),
        (
            COUNTER_PORT,
            PortDirection::Input,
            2 * blake3::WORD_BITS,
            blake3::T_LO_BASE,
            blake3::WORD_BITS,
        ),
        (
            BLOCK_LEN_PORT,
            PortDirection::Input,
            blake3::WORD_BITS,
            blake3::BLEN_BASE,
            blake3::WORD_BITS,
        ),
        (
            FLAGS_PORT,
            PortDirection::Input,
            blake3::WORD_BITS,
            blake3::FLAGS_BASE,
            blake3::WORD_BITS,
        ),
        (
            OUT_LO_PORT,
            PortDirection::Output,
            8 * blake3::WORD_BITS,
            blake3::OUT_LO_BASE,
            blake3::SLOT_BITS,
        ),
        (
            OUT_HI_PORT,
            PortDirection::Output,
            8 * blake3::WORD_BITS,
            blake3::OUT_HI_BASE,
            blake3::SLOT_BITS / 2,
        ),
    ] {
        let port = dsl.circuit().port(name).unwrap();
        assert_eq!(port.direction(), direction);
        assert_eq!(port.values().len(), len);
        assert_eq!(
            port.encoding(),
            PortEncoding::LittleEndianWord { alignment_bits }
        );
        for (offset, &value) in port.values().iter().enumerate() {
            assert_eq!(dsl.layout().value_position(value), Some(position + offset));
        }
    }
    assert_eq!(
        dsl.layout().value_position(dsl.circuit().one()),
        Some(blake3::Z_CONST_POS)
    );

    let plan = blake3_walk_projection();
    let identity_plan = identity.walk_plan().unwrap();
    assert_eq!(identity_plan.stats(), plan.stats());
    assert!(std::ptr::eq(plan, blake3_walk_projection()));
    assert!(plan.c_is_identity());
    assert_eq!(plan.stats().actions + 1, dsl.circuit().expression_count());
    assert!(plan.stats().max_live_temporaries < 1_000);
    assert!(plan.stats().action_bytes < 8 * 1024 * 1024);
    assert!(plan.stats().xor_term_bytes < 2 * 1024 * 1024);
    assert!(dsl.circuit().normalized_support_bytes() < 384 * 1024 * 1024);

    let cases = cases();
    let mut active_witness = None;
    for (case, &(cv, message, counter, block_len, flags)) in cases.iter().enumerate() {
        let actual_witness = dsl.evaluate_block(&cv, &message, counter, block_len, flags);
        let expected_witness =
            blake3::build_block_witness(&cv, &message, counter, block_len, flags);
        let identity_logical = identity
            .evaluate(&block_inputs(&cv, &message, counter, block_len, flags))
            .unwrap();
        assert_eq!(
            super::super::dsl::physical_witness(
                &identity_logical,
                identity.layout(),
                1 << blake3::K_LOG
            ),
            expected_witness,
        );
        let (cols, _) = dsl.generate_trace(&cv, &message, counter, block_len, flags);
        assert_eq!(
            cols.out_lo.into_iter().flatten().collect::<Vec<_>>(),
            expected_witness[blake3::OUT_LO_BASE..blake3::OUT_LO_BASE + 256]
        );
        assert_eq!(
            cols.out_hi.into_iter().flatten().collect::<Vec<_>>(),
            expected_witness[blake3::OUT_HI_BASE..blake3::OUT_HI_BASE + 256]
        );
        assert_eq!(actual_witness, expected_witness);
        assert_eq!(
            pcs::pack_witness(&actual_witness, blake3::K_LOG),
            pcs::pack_witness(&expected_witness, blake3::K_LOG)
        );
        let walked = plan
            .forward(
                &block_inputs(&cv, &message, counter, block_len, flags),
                blake3::K_LOG,
            )
            .expect("the structural walk must evaluate BLAKE3");
        assert_eq!(walked.z, actual_witness);
        assert_eq!(
            identity_plan
                .forward(
                    &block_inputs(&cv, &message, counter, block_len, flags),
                    blake3::K_LOG,
                )
                .unwrap(),
            walked
        );
        assert!(actual_r1cs.satisfies(&actual_witness.repeat(1 << 3)));

        assert_eq!(
            read_word(&actual_witness, blake3::T_LO_BASE),
            counter as u32
        );
        assert_eq!(
            read_word(&actual_witness, blake3::T_HI_BASE),
            (counter >> 32) as u32
        );
        assert_eq!(read_word(&actual_witness, blake3::BLEN_BASE), block_len);
        assert_eq!(read_word(&actual_witness, blake3::FLAGS_BASE), flags);
        let output = read_output(&actual_witness);
        assert_eq!(
            output,
            blake3::blake3_compress(&cv, &message, counter, block_len, flags)
        );
        if case == 1 {
            assert_eq!(output_bytes(&output), BLAKE3_EMPTY);
            active_witness = Some(actual_witness);
        }
    }

    // Standalone padding uses complete zero-input compressions so ONE is set.
    let active_witness = active_witness.unwrap();
    let zero_padding = dsl.evaluate_block(&[0; 8], &[0; 16], 0, 0, 0);
    let mut invalid_column_padding = active_witness.repeat(1 << 3);
    invalid_column_padding[blake3::USEFUL_BITS] = true;
    assert!(!actual_r1cs.satisfies(&invalid_column_padding));

    let mut partially_filled = active_witness;
    for _ in 1..1 << 3 {
        partially_filled.extend_from_slice(&zero_padding);
    }
    assert!(actual_r1cs.satisfies(&partially_filled));

    let (full_z, full_a, full_b, _) = blake3::generate_witness_batch_major(&cases, 3);
    for block in 0..1 << 3 {
        let (cv, message, counter, block_len, flags) = cases
            .get(block)
            .copied()
            .unwrap_or(([0; 8], [0; 16], 0, 0, 0));
        let walked = plan
            .forward(
                &block_inputs(&cv, &message, counter, block_len, flags),
                blake3::K_LOG,
            )
            .unwrap();
        assert_batch_major_block((&full_z, &full_a, &full_b), block, 3, &walked);
    }

    let partial_inputs = &cases[..3];
    let (partial_z, partial_a, partial_b, _) =
        blake3::generate_witness_batch_major_partial(partial_inputs, 3);
    for (block, &(cv, message, counter, block_len, flags)) in partial_inputs.iter().enumerate() {
        let walked = plan
            .forward(
                &block_inputs(&cv, &message, counter, block_len, flags),
                blake3::K_LOG,
            )
            .unwrap();
        assert_batch_major_block((&partial_z, &partial_a, &partial_b), block, 3, &walked);
    }
    let chunks_per_block = blake3::K / 128;
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
        .lincheck_circuit(blake3::K_LOG)
        .expect("BLAKE3 walk fits its base dimension");
    assert_eq!(adapter.const_pin_col(), const_pin);
    let legacy = blake3::Blake3LincheckCircuit;
    let alpha = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
    let eq: Vec<F128> = (0..blake3::K)
        .map(|i| F128::new(i as u64, (i as u64).rotate_left(17)))
        .collect();
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &eq),
        legacy.fold_alpha_batched(alpha, &eq)
    );
}

#[test]
#[ignore = "full m=22 Ligerito proof over the large BLAKE3 DSL artifact"]
fn dsl_relation_proves_and_verifies_with_walk_adapter() {
    let dsl = blake3_circuit();
    let r1cs = dsl.to_block_r1cs(8);
    let one_block = dsl.evaluate_block(&blake3::BLAKE3_IV, &[0; 16], 0, 0, EMPTY_FLAGS);
    let witness = one_block.repeat(1 << 8);
    assert!(r1cs.satisfies(&witness));

    let pcs_params = PcsParams {
        m: r1cs.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let mut prover_challenger = FsChallenger::new(b"blake3-dsl-proof-v0");
    let (proof, commitment, prover_claim) = prove_ligerito(
        &r1cs,
        pcs::pack_witness(&witness, r1cs.m),
        &pcs_params,
        &mut prover_challenger,
    );

    let plan = dsl.walk_plan();
    let adapter = plan
        .lincheck_circuit(blake3::K_LOG)
        .expect("BLAKE3 walk fits its base dimension");
    assert_eq!(adapter.const_pin_col(), r1cs.const_pin);
    let mut verifier_challenger = FsChallenger::new(b"blake3-dsl-proof-v0");
    let verifier_claim = verifier::verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &adapter,
        &pcs_params,
        &mut verifier_challenger,
    )
    .expect("the BLAKE3 DSL proof must verify through the walk adapter");
    assert_eq!(prover_claim, verifier_claim);
}

/// Run with:
/// `/usr/bin/time -v cargo +stable test --release -p flock-prover --lib \
/// r1cs_hashes::blake3_dsl::tests::performance_profile -- --ignored --exact --nocapture`
#[test]
#[ignore = "diagnostic Phase 5 profile; run alone with --nocapture"]
fn performance_profile() {
    let start = Instant::now();
    let dsl = blake3_circuit();
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
    let relation = blake3::build_block_r1cs(3);
    black_box(relation.statement_digest());
    let legacy_setup = start.elapsed();
    drop(relation);

    let input = (blake3::BLAKE3_IV, [0; 16], 0, 0, EMPTY_FLAGS);
    let inputs = block_inputs(&input.0, &input.1, input.2, input.3, input.4);
    let reference_witness = dsl.evaluate_block(&input.0, &input.1, input.2, input.3, input.4);
    let walked = plan.forward(&inputs, blake3::K_LOG).unwrap();
    let optimized_witness =
        blake3::build_block_witness(&input.0, &input.1, input.2, input.3, input.4);
    assert_eq!(walked.z, reference_witness);
    assert_eq!(optimized_witness, reference_witness);

    let reference_eval = median_time(3, || {
        black_box(dsl.evaluate_block(&input.0, &input.1, input.2, input.3, input.4));
    });
    let forward_walk = median_time(21, || {
        black_box(plan.forward(&inputs, blake3::K_LOG).unwrap());
    });
    let optimized_eval = median_time(21, || {
        black_box(blake3::build_block_witness(
            &input.0, &input.1, input.2, input.3, input.4,
        ));
    });

    let alpha = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
    let eq: Vec<F128> = (0..blake3::K)
        .map(|i| F128::new(i as u64, (i as u64).rotate_left(17)))
        .collect();
    let adapter = plan.lincheck_circuit(blake3::K_LOG).unwrap();
    let legacy = blake3::Blake3LincheckCircuit;
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &eq),
        legacy.fold_alpha_batched(alpha, &eq)
    );
    let reverse_walk = median_time(21, || {
        black_box(adapter.fold_alpha_batched(alpha, &eq));
    });
    let legacy_reverse = median_time(3, || {
        black_box(legacy.fold_alpha_batched(alpha, &eq));
    });

    println!("DSL build: {dsl_build:?}");
    println!("walk-plan build: {plan_build:?}");
    println!("DSL sparse lowering: {dsl_lower:?}");
    println!("legacy sparse setup: {legacy_setup:?}");
    println!("DSL reference evaluation: {reference_eval:?}");
    println!("structural forward walk: {forward_walk:?}");
    println!("optimized witness: {optimized_eval:?}");
    println!("structural reverse fold: {reverse_walk:?}");
    println!("legacy reverse fold: {legacy_reverse:?}");
    println!("walk stats: {:?}", plan.stats());
    println!(
        "normalized support payload: {} bytes",
        dsl.circuit().normalized_support_bytes()
    );
}
