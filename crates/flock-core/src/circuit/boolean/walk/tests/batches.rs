use super::*;

fn canonical_trace(
    circuit: &BooleanCircuit,
    r1cs: &crate::r1cs::BlockR1cs,
    inputs: &[bool],
    k_log: usize,
) -> ForwardTrace {
    let z = circuit.evaluate_r1cs(inputs, k_log).unwrap();
    ForwardTrace {
        a_z: r1cs.apply_a(&z),
        b_z: r1cs.apply_b(&z),
        c_z: r1cs.apply_c(&z),
        z,
    }
}

fn concatenate(traces: impl IntoIterator<Item = ForwardTrace>) -> ForwardTrace {
    let mut result = ForwardTrace::default();
    for mut trace in traces {
        result.z.append(&mut trace.z);
        result.a_z.append(&mut trace.a_z);
        result.b_z.append(&mut trace.b_z);
        result.c_z.append(&mut trace.c_z);
    }
    result
}

#[test]
fn forward_destinations_and_batches_match_the_canonical_evaluator() {
    let circuit = random_circuit(0xba7c_0001);
    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let r1cs = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
    let inputs: Vec<Vec<bool>> = (0..4)
        .map(|block| {
            (0..circuit.inputs().len())
                .map(|bit| ((block * 73 + 0x35) >> bit) & 1 == 1)
                .collect()
        })
        .collect();

    let expected_full = concatenate(
        inputs
            .iter()
            .map(|input| canonical_trace(&circuit, &r1cs, input, k_log)),
    );
    assert_eq!(plan.forward_batch(&inputs, k_log).unwrap(), expected_full);
    assert_eq!(
        plan.forward_batch_parallel(&inputs, k_log).unwrap(),
        expected_full
    );

    let total = inputs.len() * capacity;
    let dirty = || {
        let mut values = Vec::with_capacity(total);
        values.push(true);
        values
    };
    let mut destination = ForwardTrace {
        z: dirty(),
        a_z: dirty(),
        b_z: dirty(),
        c_z: dirty(),
    };
    let pointers = (
        destination.z.as_ptr(),
        destination.a_z.as_ptr(),
        destination.b_z.as_ptr(),
        destination.c_z.as_ptr(),
    );

    plan.forward_batch_into(&inputs, k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, expected_full);
    assert_eq!(
        pointers,
        (
            destination.z.as_ptr(),
            destination.a_z.as_ptr(),
            destination.b_z.as_ptr(),
            destination.c_z.as_ptr(),
        ),
        "sufficient destination capacity must be reused"
    );

    let expected_partial = concatenate(
        inputs[..3]
            .iter()
            .map(|input| canonical_trace(&circuit, &r1cs, input, k_log)),
    );
    plan.forward_batch_into(&inputs[..3], k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, expected_partial);
    plan.forward_batch_into_parallel(&inputs[..3], k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, expected_partial);

    let expected_one = canonical_trace(&circuit, &r1cs, &inputs[0], k_log);
    plan.forward_into(&inputs[0], k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, expected_one);
    assert_eq!(destination.z.len(), capacity);
}

#[test]
fn forward_batch_validates_inputs_and_reports_the_failing_block() {
    let circuit = random_circuit(0xba7c_0002);
    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let valid = vec![false; circuit.inputs().len()];
    let invalid = vec![false; circuit.inputs().len() - 1];
    let mut destination = ForwardTrace {
        z: vec![true; 3],
        a_z: vec![true; 3],
        b_z: vec![true; 3],
        c_z: vec![true; 3],
    };
    let original = destination.clone();
    assert!(matches!(
        plan.forward_batch_into(&[valid, invalid], k_log, &mut destination),
        Err(WalkError::InputCount { .. })
    ));
    assert_eq!(destination, original, "validation must precede writes");

    let circuit = support::circuit(1, 0, |b, cols| {
        b.assert_zero(cols.input[0]);
    });
    let row = circuit.rows()[2].id();
    let rejecting = circuit.walk_plan().unwrap();
    let inputs = [vec![false], vec![true], vec![true]];
    let k_log = capacity_log(rejecting.useful_bits());
    assert_eq!(
        rejecting.forward_batch(&inputs, k_log),
        Err(WalkError::UnsatisfiedBatchRow { block: 1, row })
    );
    assert_eq!(
        rejecting.forward_batch_parallel(&inputs, k_log),
        Err(WalkError::UnsatisfiedBatchRow { block: 1, row })
    );

    let dirty = ForwardTrace {
        z: vec![true; 3],
        a_z: vec![true; 3],
        b_z: vec![true; 3],
        c_z: vec![true; 3],
    };
    destination = dirty.clone();
    assert_eq!(
        rejecting.forward_into(&[true], k_log, &mut destination),
        Err(WalkError::UnsatisfiedRow(row))
    );

    destination = dirty.clone();
    assert_eq!(
        rejecting.forward_batch_into(&inputs, k_log, &mut destination),
        Err(WalkError::UnsatisfiedBatchRow { block: 1, row })
    );

    destination = dirty.clone();
    assert_eq!(
        rejecting.forward_batch_into_parallel(&inputs, k_log, &mut destination),
        Err(WalkError::UnsatisfiedBatchRow { block: 1, row })
    );

    let accepted = [vec![false], vec![false]];
    let expected = rejecting.forward_batch(&accepted, k_log).unwrap();
    rejecting
        .forward_batch_into(&accepted, k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, expected, "a later success restores validity");
}

#[test]
fn forward_batch_handles_empty_and_invalid_shapes() {
    let circuit = random_circuit(0xba7c_0003);
    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let empty: &[Vec<bool>] = &[];
    let dirty = ForwardTrace {
        z: vec![true],
        a_z: vec![true],
        b_z: vec![true],
        c_z: vec![true],
    };
    let mut destination = dirty.clone();

    plan.forward_batch_into(empty, k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, ForwardTrace::default());
    destination = dirty.clone();
    plan.forward_batch_into_parallel(empty, k_log, &mut destination)
        .unwrap();
    assert_eq!(destination, ForwardTrace::default());

    let input = vec![false; circuit.inputs().len()];
    for (invalid_k_log, expected) in [
        (
            0,
            WalkError::Capacity {
                required: plan.useful_bits(),
                actual: 1,
            },
        ),
        (
            usize::BITS as usize,
            WalkError::InvalidKLog(usize::BITS as usize),
        ),
    ] {
        destination = dirty.clone();
        assert_eq!(
            plan.forward_batch_into(
                std::slice::from_ref(&input),
                invalid_k_log,
                &mut destination
            ),
            Err(expected.clone())
        );
        assert_eq!(destination, dirty, "preflight errors preserve the trace");

        assert_eq!(
            plan.forward_batch_into_parallel(
                std::slice::from_ref(&input),
                invalid_k_log,
                &mut destination
            ),
            Err(expected)
        );
        assert_eq!(destination, dirty, "preflight errors preserve the trace");
    }
}
