use super::*;

fn concatenate(traces: &[ForwardTrace]) -> ForwardTrace {
    let mut result = ForwardTrace::default();
    for trace in traces {
        result.z.extend(&trace.z);
        result.a_z.extend(&trace.a_z);
        result.b_z.extend(&trace.b_z);
        result.c_z.extend(&trace.c_z);
    }
    result
}

#[test]
fn batches_match_sparse_matrices_and_reuse_storage() {
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        let circuit = batch_fixture(mode);
        let plan = circuit.walk_plan().unwrap();
        let matrix = circuit.to_block_r1cs(8, 0, 0).unwrap();
        check_transpose(&plan, &matrix);
        assert!(plan.forward(&[], 8).is_err());
        assert!(plan.forward(&[true, true, false], 8).is_err());
        let inputs = valid_inputs(&circuit);
        let traces: Vec<_> = inputs
            .iter()
            .map(|input| reference(&circuit, &matrix, input))
            .collect();
        let expected = concatenate(&traces);
        assert_eq!(plan.forward_batch(&inputs, 8).unwrap(), expected);
        assert_eq!(plan.forward_batch_parallel(&inputs, 8).unwrap(), expected);

        let mut destination = dirty_trace(expected.z.len());
        for count in [inputs.len(), 3, 1, 0] {
            let expected = concatenate(&traces[..count]);
            plan.forward_batch_into(&inputs[..count], 8, &mut destination)
                .unwrap();
            assert_eq!(destination, expected, "{mode:?}, serial count {count}");
            for values in [
                &mut destination.z,
                &mut destination.a_z,
                &mut destination.b_z,
                &mut destination.c_z,
            ] {
                values.fill(true);
            }
            plan.forward_batch_into_parallel(&inputs[..count], 8, &mut destination)
                .unwrap();
            assert_eq!(destination, expected, "{mode:?}, parallel count {count}");
        }
        plan.forward_into(&inputs[0], 8, &mut destination).unwrap();
        assert_eq!(destination, traces[0]);
    }
}
