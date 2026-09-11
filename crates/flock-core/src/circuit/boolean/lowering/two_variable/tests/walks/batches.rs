use super::*;
use crate::union::SlotWitnessDest;

fn unpack(words: &[F128], block: usize) -> Vec<bool> {
    (0..256)
        .map(|bit| {
            let word = words[(bit / 128) * 4 + block];
            let limb = if bit % 128 < 64 { word.lo } else { word.hi };
            limb >> (bit % 64) & 1 != 0
        })
        .collect()
}

#[test]
fn converted_batches_match_sparse_results_and_preserve_failure_contracts() {
    let lowered = fixture(1, true, LoweringMode::RequireIdentityC);
    let plan = lowered.walk_plan().unwrap();
    let matrix = lowered.to_block_r1cs(8, 0, 0).unwrap();
    let inputs = valid_inputs(&lowered);
    let serial = plan.forward_batch(&inputs, 8).unwrap();
    let mut reused = ForwardTrace {
        z: vec![true; 2048],
        a_z: vec![true; 2048],
        b_z: vec![true; 2048],
        c_z: vec![true; 2048],
    };
    plan.forward_batch_into_parallel(&inputs, 8, &mut reused)
        .unwrap();
    assert_eq!(reused, serial);
    plan.forward_batch_into(&inputs[..1], 8, &mut reused)
        .unwrap();
    assert_eq!(reused, reference(&lowered, &matrix, &inputs[0]));
    for (block, input) in inputs.iter().enumerate() {
        let expected = reference(&lowered, &matrix, input);
        let range = block * 256..(block + 1) * 256;
        assert_eq!(&serial.z[range.clone()], expected.z);
        assert_eq!(&serial.a_z[range.clone()], expected.a_z);
        assert_eq!(&serial.b_z[range.clone()], expected.b_z);
        assert_eq!(&serial.c_z[range], expected.c_z);
    }
    let bad = vec![true, true, false];
    let invalid = [inputs[0].clone(), bad.clone(), bad];
    let expected_row = lowered.auxiliaries()[0].cancellation_row;
    let error = Err(WalkError::UnsatisfiedBatchRow {
        block: 1,
        row: expected_row,
    });
    assert_eq!(plan.forward_batch_into(&invalid, 8, &mut reused), error);
    assert_eq!(
        plan.forward_batch_into_parallel(&invalid, 8, &mut reused),
        error
    );
    plan.forward_batch_into_parallel(&inputs, 8, &mut reused)
        .unwrap();
    assert_eq!(reused, serial);
    let saved = reused.clone();
    assert!(
        plan.forward_batch_into_parallel(&[Vec::<bool>::new()], 8, &mut reused)
            .is_err()
    );
    assert_eq!(saved, reused);
    assert_eq!(
        plan.forward_batch::<Vec<bool>>(&[], 8).unwrap(),
        ForwardTrace::default()
    );
}

#[test]
fn converted_packed_batches_match_sparse_outputs_and_respect_elision() {
    let lowered = fixture(1, true, LoweringMode::RequireIdentityC);
    let plan = lowered.walk_plan().unwrap();
    let matrix = lowered.to_block_r1cs(8, 0, 0).unwrap();
    let inputs = valid_inputs(&lowered);
    assert_eq!(inputs.len(), 4);
    for count in [0, 3, 4] {
        for elide in [false, true] {
            let marker = F128::new(u64::MAX, u64::MAX);
            let mut z = vec![marker; 8];
            let mut a = z.clone();
            let mut b = z.clone();
            plan.forward_batch_identity_c_into_slot(
                &inputs[..count],
                8,
                2,
                SlotWitnessDest {
                    z: &mut z,
                    a: &mut a,
                    b: &mut b,
                    elide_padding_writes: elide,
                },
            )
            .unwrap();
            for (block, input) in inputs[..count].iter().enumerate() {
                let expected = reference(&lowered, &matrix, input);
                assert_eq!(unpack(&z, block), expected.z);
                assert_eq!(unpack(&a, block), expected.a_z);
                assert_eq!(unpack(&b, block), expected.b_z);
            }
            for block in count..4 {
                for words in [&z, &a, &b] {
                    assert_eq!(unpack(words, block), vec![elide; 256]);
                }
            }
        }
    }
    let mut z = vec![F128::ONE; 8];
    let mut a = z.clone();
    let mut b = z.clone();
    let error = plan.forward_batch_identity_c_into_slot(
        &[vec![true]],
        8,
        2,
        SlotWitnessDest {
            z: &mut z,
            a: &mut a,
            b: &mut b,
            elide_padding_writes: false,
        },
    );
    assert!(matches!(error, Err(WalkError::InputCount { .. })));
    assert!(
        z.iter()
            .chain(&a)
            .chain(&b)
            .all(|&value| value == F128::ONE)
    );
    let error = plan.forward_batch_identity_c_into_slot(
        &[vec![true, true, false]],
        8,
        2,
        SlotWitnessDest {
            z: &mut z,
            a: &mut a,
            b: &mut b,
            elide_padding_writes: false,
        },
    );
    assert_eq!(
        error,
        Err(WalkError::UnsatisfiedRow(
            lowered.auxiliaries()[0].cancellation_row
        ))
    );
}
