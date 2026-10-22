use super::super::dsl::tests::{
    check_hash_lowering, check_hash_trace, prove_and_verify, read_word,
};
use super::*;

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

#[test]
#[ignore = "large hash relation and walk checks; run explicitly"]
fn dsl_outputs_and_walks_match_reference() {
    let dsl = blake3_circuit();
    let matrix = blake3_relation_projection(3);
    assert!(std::sync::Arc::ptr_eq(
        &matrix,
        &blake3_relation_projection(3)
    ));
    let plan = blake3_walk_projection();
    check_hash_lowering(dsl.circuit(), dsl.layout(), &matrix, plan);
    for (case, (cv, message, counter, block_len, flags)) in cases().into_iter().enumerate() {
        let (cols, _) = dsl.generate_trace(&cv, &message, counter, block_len, flags);
        let output: [u32; 16] = std::array::from_fn(|i| {
            read_word(if i < 8 {
                cols.out_lo[i]
            } else {
                cols.out_hi[i - 8]
            })
        });
        assert_eq!(
            output,
            flock_hash::blake3_compress(&cv, &message, counter, block_len, flags)
        );
        if case == 1 {
            assert_eq!(output_bytes(&output), BLAKE3_EMPTY);
        }
        let witness = dsl.evaluate_block(&cv, &message, counter, block_len, flags);
        let walked = plan
            .forward(
                &block_inputs(&cv, &message, counter, block_len, flags),
                blake3::K_LOG,
            )
            .unwrap();
        let output = dsl.circuit().column(OUT_LO_FIELD).unwrap();
        check_hash_trace(
            &matrix,
            walked,
            &witness,
            [
                dsl.layout().value_position(output.values[0]).unwrap(),
                dsl.layout().useful_bits(),
            ],
        );
    }
}

#[test]
#[ignore = "full m=22 Ligerito proof over the large BLAKE3 DSL artifact"]
fn dsl_relation_proves_and_verifies_with_walk_adapter() {
    let dsl = blake3_circuit();
    let r1cs = dsl.to_block_r1cs(8);
    let one_block = dsl.evaluate_block(&blake3::BLAKE3_IV, &[0; 16], 0, 0, EMPTY_FLAGS);
    let witness = one_block.repeat(1 << 8);
    prove_and_verify(&r1cs, &dsl.walk_plan(), &witness, b"blake3-dsl-proof-v0");
}
