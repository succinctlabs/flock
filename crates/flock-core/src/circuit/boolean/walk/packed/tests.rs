use super::*;
use crate::circuit::boolean::{BooleanCircuit, CircuitBuilder, WalkPlan};

fn identity_fixture() -> BooleanCircuit {
    let mut builder = CircuitBuilder::new();
    let input = builder.input_bits::<8>("input");
    let xor = builder.xor([input[0], input[1], input[2], input[3]]);
    let product = builder.and(xor, input[4]);
    let output = builder.xor([product, input[5], input[6], input[7]]);
    let output = builder.materialize(output);
    builder.output("output", [output]);
    builder.finish()
}

fn unpack(words: &[F128], batch_capacity: usize, block: usize, bits: usize) -> Vec<bool> {
    (0..bits)
        .map(|bit| {
            let word = words[(bit >> 7) * batch_capacity + block];
            if (bit & 127) < 64 {
                word.lo >> (bit & 63) & 1 == 1
            } else {
                word.hi >> (bit & 63) & 1 == 1
            }
        })
        .collect()
}

fn inputs(count: usize) -> Vec<Vec<bool>> {
    (0..count)
        .map(|block| (0..8).map(|bit| (block + bit * 3) & 1 == 1).collect())
        .collect()
}

fn assert_packed_error_preserves_destination(
    plan: &WalkPlan,
    inputs: &[Vec<bool>],
    k_log: usize,
    n_blocks_log: usize,
    expected: WalkError,
) {
    let mut z = vec![F128::ONE; 4];
    let mut a = z.clone();
    let mut b = z.clone();
    assert_eq!(
        plan.forward_batch_identity_c_into_slot(
            inputs,
            k_log,
            n_blocks_log,
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
            },
        ),
        Err(expected)
    );
    assert!(
        z.iter().chain(&a).chain(&b).all(|&word| word == F128::ONE),
        "validation errors must precede writes"
    );
}

#[test]
fn union_slot_output_matches_serial_walks_for_full_and_partial_batches() {
    let circuit = identity_fixture();
    let plan = circuit.walk_plan().unwrap();
    assert!(plan.c_is_identity());
    let k_log = 7;
    let block_capacity = 1 << k_log;
    let n_blocks_log = 2;
    let batch_capacity = 1 << n_blocks_log;
    let words = (block_capacity >> 7) * batch_capacity;

    for count in [batch_capacity, batch_capacity - 1] {
        let inputs = inputs(count);
        let mut z = vec![F128::ONE; words];
        let mut a = vec![F128::ONE; words];
        let mut b = vec![F128::ONE; words];
        plan.forward_batch_identity_c_into_slot(
            &inputs,
            k_log,
            n_blocks_log,
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
            },
        )
        .unwrap();

        for block in 0..count {
            let expected = plan.forward(&inputs[block], k_log).unwrap();
            assert_eq!(
                unpack(&z, batch_capacity, block, block_capacity),
                expected.z
            );
            assert_eq!(
                unpack(&a, batch_capacity, block, block_capacity),
                expected.a_z
            );
            assert_eq!(
                unpack(&b, batch_capacity, block, block_capacity),
                expected.b_z
            );
            assert_eq!(expected.c_z, expected.z);
        }
        for block in count..batch_capacity {
            assert!(
                unpack(&z, batch_capacity, block, block_capacity)
                    .iter()
                    .all(|&bit| !bit)
            );
            assert!(
                unpack(&a, batch_capacity, block, block_capacity)
                    .iter()
                    .all(|&bit| !bit)
            );
            assert!(
                unpack(&b, batch_capacity, block, block_capacity)
                    .iter()
                    .all(|&bit| !bit)
            );
        }
    }
}

#[test]
fn union_slot_elision_and_validation_follow_the_destination_contract() {
    let circuit = identity_fixture();
    let plan = circuit.walk_plan().unwrap();
    let k_log = 7;
    let n_blocks_log = 2;
    let batch_capacity = 1 << n_blocks_log;
    let inputs = inputs(3);
    let mut z = vec![F128::ONE; batch_capacity];
    let mut a = vec![F128::ONE; batch_capacity];
    let mut b = vec![F128::ONE; batch_capacity];
    plan.forward_batch_identity_c_into_slot(
        &inputs,
        k_log,
        n_blocks_log,
        SlotWitnessDest {
            z: &mut z,
            a: &mut a,
            b: &mut b,
            elide_padding_writes: true,
        },
    )
    .unwrap();
    assert_eq!(z[3], F128::ONE, "elided dummy row must stay untouched");
    assert_eq!(a[3], F128::ONE, "elided dummy row must stay untouched");
    assert_eq!(b[3], F128::ONE, "elided dummy row must stay untouched");

    let original = (z.clone(), a.clone(), b.clone());
    let error = plan.forward_batch_identity_c_into_slot(
        &inputs,
        k_log,
        n_blocks_log,
        SlotWitnessDest {
            z: &mut z[..3],
            a: &mut a,
            b: &mut b,
            elide_padding_writes: false,
        },
    );
    assert!(matches!(
        error,
        Err(WalkError::PackedDestinationLength { .. })
    ));
    assert_eq!((z, a, b), original, "validation must precede writes");

    let mut builder = CircuitBuilder::new();
    let input = builder.input();
    builder.assert_zero(input);
    let general_c = builder.finish().walk_plan().unwrap();
    let mut z = vec![F128::ONE; batch_capacity];
    let mut a = vec![F128::ONE; batch_capacity];
    let mut b = vec![F128::ONE; batch_capacity];
    assert_eq!(
        general_c.forward_batch_identity_c_into_slot(
            &[vec![false]],
            k_log,
            n_blocks_log,
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
            },
        ),
        Err(WalkError::IdentityCRequired)
    );
    assert!(z.iter().chain(&a).chain(&b).all(|&word| word == F128::ONE));
}

#[test]
fn union_slot_validation_boundaries_preserve_the_destination() {
    let plan = identity_fixture().walk_plan().unwrap();
    let valid = inputs(1);
    assert_packed_error_preserves_destination(&plan, &valid, 6, 2, WalkError::PackedKLog(6));

    let invalid_batch_log = usize::BITS as usize;
    assert_packed_error_preserves_destination(
        &plan,
        &valid,
        7,
        invalid_batch_log,
        WalkError::InvalidBatchLog(invalid_batch_log),
    );
    assert_packed_error_preserves_destination(
        &plan,
        &inputs(5),
        7,
        2,
        WalkError::BatchCount {
            capacity: 4,
            actual: 5,
        },
    );
    assert_packed_error_preserves_destination(
        &plan,
        &[vec![false; 7]],
        7,
        2,
        WalkError::InputCount {
            expected: 8,
            actual: 7,
        },
    );
}
