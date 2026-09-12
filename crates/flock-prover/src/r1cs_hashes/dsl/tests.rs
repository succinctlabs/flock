use crate::prover::prove_ligerito;
use flock_core::challenger::FsChallenger;
use flock_core::circuit::boolean::{ForwardTrace, WalkPlan};
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{self, PcsParams};
use flock_core::verifier;

use flock_core::r1cs::BlockR1cs;
pub(crate) fn read_word(bits: [bool; 32]) -> u32 {
    bits.into_iter()
        .enumerate()
        .fold(0, |word, (bit, value)| word | (u32::from(value) << bit))
}

pub(crate) fn check_hash_lowering(
    circuit: &flock_core::circuit::boolean::BooleanCircuit,
    layout: &flock_core::circuit::boolean::PhysicalLayout,
    matrix: &BlockR1cs,
    plan: &WalkPlan,
) {
    let identity = circuit.lower_identity_c().unwrap();
    assert!(identity.c_is_identity());
    assert!(identity.auxiliaries().is_empty());
    assert_eq!(identity.layout(), layout);
    let converted = identity
        .to_block_r1cs(matrix.k_log, matrix.k_skip, 3)
        .unwrap();
    assert_eq!(converted.statement_digest(), matrix.statement_digest());
    let alpha = F128::new(0x1234, 0x5678);
    let eq: Vec<_> = (0..1 << matrix.k_log)
        .map(|i| F128::new(i as u64, 7))
        .collect();
    assert_eq!(
        plan.lincheck_circuit(matrix.k_log)
            .unwrap()
            .fold_alpha_batched(alpha, &eq),
        matrix
            .sparse_lincheck_circuit()
            .fold_alpha_batched(alpha, &eq)
    );
}

pub(crate) fn check_hash_trace(
    matrix: &BlockR1cs,
    walked: ForwardTrace,
    witness: &[bool],
    corrupt_positions: [usize; 2],
) {
    assert_eq!(walked.z, witness);
    let blocks = 1 << (matrix.m - matrix.k_log);
    let witness = witness.repeat(blocks);
    assert_eq!(walked.a_z.repeat(blocks), matrix.apply_a(&witness));
    assert_eq!(walked.b_z.repeat(blocks), matrix.apply_b(&witness));
    assert_eq!(walked.c_z.repeat(blocks), matrix.apply_c(&witness));
    assert!(matrix.satisfies(&witness));
    for position in corrupt_positions {
        let mut invalid = witness.clone();
        invalid[position] ^= true;
        assert!(!matrix.satisfies(&invalid), "corrupted position {position}");
    }
}

pub(crate) fn prove_and_verify(
    matrix: &BlockR1cs,
    plan: &WalkPlan,
    witness: &[bool],
    domain: &[u8],
) {
    assert!(matrix.satisfies(witness));
    let params = PcsParams {
        m: matrix.m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let (proof, commitment, expected) = prove_ligerito(
        matrix,
        pcs::pack_witness(witness, matrix.m),
        &params,
        &mut FsChallenger::new(domain),
    );
    let adapter = plan.lincheck_circuit(matrix.k_log).unwrap();
    assert_eq!(adapter.const_pin_col(), matrix.const_pin);
    let actual = verifier::verify_ligerito(
        matrix,
        &commitment,
        &proof,
        &adapter,
        &params,
        &mut FsChallenger::new(domain),
    )
    .expect("hash proof must verify through the walk adapter");
    assert_eq!(actual, expected);
}
