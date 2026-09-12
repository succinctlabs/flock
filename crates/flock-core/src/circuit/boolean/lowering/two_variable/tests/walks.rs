use super::*;
use crate::circuit::boolean::{EvaluationError, ForwardTrace, WalkError};
use crate::field::F128;
use crate::r1cs::SparseBinaryMatrix;

mod batches;
mod transpose;

fn fixture(case: usize, mode: LoweringMode) -> LoweredCircuit {
    super::equivalence::fixture(case).lower(mode).unwrap()
}

fn reference(lowered: &LoweredCircuit, matrix: &BlockR1cs, input: &[bool]) -> ForwardTrace {
    let logical = lowered.evaluate(input).unwrap();
    let z = physical(lowered, &logical, 1 << matrix.k_log);
    ForwardTrace {
        a_z: matrix.apply_a(&z),
        b_z: matrix.apply_b(&z),
        c_z: matrix.apply_c(&z),
        z,
    }
}

fn valid_inputs(lowered: &LoweredCircuit) -> Vec<Vec<bool>> {
    (0..1 << lowered.inputs().len())
        .map(|n| bits(n, lowered.inputs().len()))
        .filter(|input| lowered.evaluate(input).is_ok())
        .collect()
}

fn sparse_transpose(matrix: &SparseBinaryMatrix, weights: &[F128]) -> Vec<F128> {
    let mut output = vec![F128::ZERO; matrix.num_cols];
    for (row, &weight) in matrix.rows.iter().zip(weights) {
        for &column in row {
            output[column] += weight;
        }
    }
    output
}

#[test]
fn selected_walks_match_reference_and_sparse_evaluation() {
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        for case in 0..7 {
            let lowered = fixture(case, mode);
            // Compile before emitting matrices: walks need only the shared relation view.
            let plan = lowered.walk_plan().unwrap();
            let matrix = lowered.to_block_r1cs(7, 0, 0).unwrap();
            assert_eq!(plan.c_is_identity(), matrix.c0_is_identity());
            let mut reused = ForwardTrace {
                z: vec![true; 256],
                a_z: vec![true; 256],
                b_z: vec![true; 256],
                c_z: vec![true; 256],
            };
            for input in (0..8).map(|n| bits(n, 3)) {
                match lowered.evaluate(&input) {
                    Ok(_) => {
                        plan.forward_into(&input, 7, &mut reused).unwrap();
                        assert_eq!(reused, reference(&lowered, &matrix, &input));
                        if plan.c_is_identity() {
                            assert_eq!(reused.c_z, reused.z);
                        }
                        for aux in lowered.auxiliaries() {
                            assert!(
                                !reused.z
                                    [lowered.layout().value_position(aux.cancellation).unwrap()]
                            );
                        }
                    }
                    Err(EvaluationError::UnsatisfiedRow(source)) => {
                        let Err(WalkError::UnsatisfiedRow(target)) =
                            plan.forward_into(&input, 7, &mut reused)
                        else {
                            panic!("failed source assertion must reject in the walk");
                        };
                        assert_eq!(lowered.source_row(target), Some(source));
                    }
                    Err(other) => panic!("unexpected reference error: {other}"),
                }
            }
            let saved = reused.clone();
            assert!(matches!(
                plan.forward_into(&[], 7, &mut reused),
                Err(WalkError::InputCount { .. })
            ));
            assert_eq!(reused, saved);
            assert!(matches!(
                plan.forward_into(&[false; 3], 0, &mut reused),
                Err(WalkError::Capacity { .. })
            ));
            assert_eq!(reused, saved);
        }
    }
}
