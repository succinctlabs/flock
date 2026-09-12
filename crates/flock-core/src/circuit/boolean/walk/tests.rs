use crate::circuit::boolean::tests::support;
use crate::circuit::boolean::{ForwardTrace, LoweredCircuit, LoweringMode, WalkError, WalkPlan};
use crate::field::F128;
use crate::lincheck::LincheckCircuit;
use crate::r1cs::{BlockR1cs, SparseBinaryMatrix};

fn capacity_log(required: usize) -> usize {
    required.next_power_of_two().trailing_zeros() as usize
}

fn weights(len: usize, seed: u64) -> Vec<F128> {
    crate::test_rng::Rng::new(seed).f128_vec(len)
}

fn sparse_transpose(matrix: &SparseBinaryMatrix, row_weights: &[F128]) -> Vec<F128> {
    assert_eq!(matrix.num_rows, row_weights.len());
    let mut result = vec![F128::ZERO; matrix.num_cols];
    for (row, &weight) in matrix.rows.iter().zip(row_weights) {
        for &column in row {
            result[column] += weight;
        }
    }
    result
}

fn check_transpose(plan: &WalkPlan, matrix: &BlockR1cs) {
    let [ea, eb, ec] = [41, 42, 43].map(|seed| weights(1 << matrix.k_log, seed));
    let mut expected = sparse_transpose(&matrix.a_0, &ea);
    for (matrix, weights) in [(&matrix.b_0, &eb), (&matrix.c_0, &ec)] {
        for (value, term) in expected.iter_mut().zip(sparse_transpose(matrix, weights)) {
            *value += term;
        }
    }
    assert_eq!(plan.transpose(&ea, &eb, &ec).unwrap(), expected);
    if plan.c_is_identity() {
        assert_eq!(plan.transpose_identity_c(&ea, &eb, &ec).unwrap(), expected);
    } else {
        assert_eq!(
            plan.transpose_identity_c(&ea, &eb, &ec),
            Err(WalkError::IdentityCRequired)
        );
    }
    let adapter = plan.lincheck_circuit(matrix.k_log).unwrap();
    let sparse = matrix.sparse_lincheck_circuit();
    let alpha = F128::new(0x1234, 0x5678);
    assert_eq!(adapter.n_cols(), sparse.n_cols());
    assert_eq!(adapter.const_pin_col(), matrix.const_pin);
    assert_eq!(
        adapter.fold_alpha_batched(alpha, &ea),
        sparse.fold_alpha_batched(alpha, &ea)
    );
}

fn dirty_trace(len: usize) -> ForwardTrace {
    ForwardTrace {
        z: vec![true; len],
        a_z: vec![true; len],
        b_z: vec![true; len],
        c_z: vec![true; len],
    }
}

fn batch_fixture(mode: LoweringMode) -> LoweredCircuit {
    support::circuit(3, 2, |b, cols| {
        let shared = b.xor2(cols.input[0], cols.input[2]);
        b.define_and(cols.witness[0], shared, cols.input[1]);
        b.constrain(cols.input[0], cols.input[1], cols.input[2]);
        b.define_linear(cols.witness[1], shared);
    })
    .lower(mode)
    .unwrap()
}

fn valid_inputs(circuit: &LoweredCircuit) -> Vec<Vec<bool>> {
    (0..1 << circuit.inputs().len())
        .map(|n| {
            (0..circuit.inputs().len())
                .map(|i| n >> i & 1 != 0)
                .collect::<Vec<_>>()
        })
        .filter(|input| circuit.evaluate(input).is_ok())
        .collect()
}

fn reference(circuit: &LoweredCircuit, matrix: &BlockR1cs, input: &[bool]) -> ForwardTrace {
    let values = circuit.evaluate(input).unwrap();
    let mut z = vec![false; 1 << matrix.k_log];
    for (&position, value) in circuit.layout().value_positions().iter().zip(values) {
        z[position] = value;
    }
    ForwardTrace {
        a_z: matrix.apply_a(&z),
        b_z: matrix.apply_b(&z),
        c_z: matrix.apply_c(&z),
        z,
    }
}

#[path = "tests/batches.rs"]
mod batches;
#[path = "tests/packed.rs"]
mod packed;
