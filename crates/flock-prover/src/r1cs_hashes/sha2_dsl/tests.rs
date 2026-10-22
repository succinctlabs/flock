use super::super::dsl::tests::{
    check_hash_lowering, check_hash_trace, prove_and_verify, read_word,
};
use super::*;

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

fn block_inputs(h_in: &[u32; 8], message: &[u32; 16]) -> Vec<bool> {
    let mut inputs = Vec::with_capacity((sha2::H_WORDS + sha2::M_WORDS) * sha2::WORD_BITS);
    for &word in h_in.iter().chain(message) {
        inputs.extend((0..sha2::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs
}

#[test]
#[ignore = "large hash relation and walk checks; run explicitly"]
fn dsl_outputs_and_walks_match_reference() {
    let dsl = sha256_circuit();
    let matrix = sha256_relation_projection(3);
    assert!(std::sync::Arc::ptr_eq(
        &matrix,
        &sha256_relation_projection(3)
    ));
    let plan = sha256_walk_projection();
    check_hash_lowering(dsl.circuit(), dsl.layout(), &matrix, plan);
    let mut rng = flock_core::test_rng::Rng::new(0x5A25_6D51);
    let mut abc = [0; 16];
    abc[0] = 0x6162_6380;
    abc[15] = 24;
    let cases = [
        ([0; 8], [0; 16]),
        (sha2::SHA256_IV, abc),
        (
            std::array::from_fn(|_| rng.next_u32()),
            std::array::from_fn(|_| rng.next_u32()),
        ),
    ];
    for (case, (h_in, message)) in cases.into_iter().enumerate() {
        let (cols, _) = dsl.generate_trace(&h_in, &message);
        let output = cols.h_out.map(read_word);
        assert_eq!(output, sha2::sha256_compress(&h_in, &message));
        if case == 1 {
            assert_eq!(output, SHA256_ABC);
        }
        let witness = dsl.evaluate_block(&h_in, &message);
        let walked = plan
            .forward(&block_inputs(&h_in, &message), sha2::K_LOG)
            .unwrap();
        let output = dsl.circuit().column(H_OUT_FIELD).unwrap();
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
#[ignore = "full m=22 Ligerito proof over the large SHA DSL artifact"]
fn dsl_relation_proves_and_verifies_with_walk_adapter() {
    let dsl = sha256_circuit();
    let r1cs = dsl.to_block_r1cs(7);
    let one_block = dsl.evaluate_block(&sha2::SHA256_IV, &[0u32; 16]);
    let witness = one_block.repeat(1 << 7);
    prove_and_verify(&r1cs, &dsl.walk_plan(), &witness, b"sha256-dsl-proof-v0");
}
