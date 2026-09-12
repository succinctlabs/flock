use super::*;
use crate::union::SlotWitnessDest;

fn identity_fixture() -> LoweredCircuit {
    support::circuit(8, 130, |b, cols| {
        let xor = b.xor(cols.input[..4].iter().copied());
        b.define_and(cols.witness[0], xor, cols.input[4]);
        for (index, &out) in cols.witness[1..].iter().enumerate() {
            let expr = b.xor2(cols.witness[0], cols.input[index % 8]);
            b.define_linear(out, expr);
        }
    })
    .lower(LoweringMode::Direct)
    .unwrap()
}

fn unpack(words: &[F128], block: usize, bits: usize) -> Vec<bool> {
    (0..bits)
        .map(|bit| {
            let word = words[(bit / 128) * 4 + block];
            let limb = if bit % 128 < 64 { word.lo } else { word.hi };
            limb >> (bit % 64) & 1 != 0
        })
        .collect()
}

#[test]
fn packed_batches_match_sparse_matrices_and_respect_elision() {
    for circuit in [
        identity_fixture(),
        batch_fixture(LoweringMode::RequireIdentityC),
    ] {
        let plan = circuit.walk_plan().unwrap();
        // The large fixture has live values beyond bit 128, not just padding.
        let k_log = capacity_log(plan.useful_bits()).max(7);
        let capacity = 1 << k_log;
        let matrix = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
        let valid = valid_inputs(&circuit);
        // Distinct patterns exercise both limbs and expose block permutations.
        let inputs: Vec<_> = (0..4)
            .map(|block| valid[(block * 73 + 0x35) % valid.len()].clone())
            .collect();
        if capacity > 128 {
            assert!(inputs.iter().any(|input| {
                reference(&circuit, &matrix, input).z[128..]
                    .iter()
                    .any(|&bit| bit)
            }));
        }
        for count in [4, 3, 0] {
            for elide in [false, true] {
                let marker = F128::new(u64::MAX, u64::MAX);
                let mut z = vec![marker; (capacity / 128) * 4];
                let mut a = z.clone();
                let mut b = z.clone();
                plan.forward_batch_identity_c_into_slot(
                    &inputs[..count],
                    k_log,
                    2,
                    SlotWitnessDest {
                        z: &mut z,
                        a: &mut a,
                        b: &mut b,
                        elide_padding_writes: elide,
                        dead_padding_unread: false,
                    },
                )
                .unwrap();
                for (block, input) in inputs[..count].iter().enumerate() {
                    let expected = reference(&circuit, &matrix, input);
                    assert_eq!(unpack(&z, block, capacity), expected.z);
                    assert_eq!(unpack(&a, block, capacity), expected.a_z);
                    assert_eq!(unpack(&b, block, capacity), expected.b_z);
                    assert_eq!(expected.c_z, expected.z);
                }
                for block in count..4 {
                    for words in [&z, &a, &b] {
                        assert_eq!(unpack(words, block, capacity), vec![elide; capacity]);
                    }
                }
            }
        }
    }
}
