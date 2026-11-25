use crate::circuit::boolean::tests::support;
use crate::circuit::boolean::{ForwardTrace, LoweredCircuit, LoweringMode, WalkError, WalkPlan};
use std::sync::Arc;

use crate::field::F128;
use crate::lincheck::{CscCircuit, LincheckCircuit, MatrixSide};
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
    check_split_marginals(plan, matrix, &[&ea, &eb, &ec]);
}

fn sparse_apply(matrix: &SparseBinaryMatrix, v: &[F128]) -> Vec<F128> {
    matrix
        .rows
        .iter()
        .map(|row| row.iter().fold(F128::ZERO, |acc, &column| acc + v[column]))
        .collect()
}

/// The fold's marginals: both matrices' column marginals at their own
/// weights in one reverse walk, both row marginals in one forward walk, and
/// a relation carrying the walk, whose lincheck circuit then walks every
/// marginal and agrees with the gathering one.
fn check_split_marginals(plan: &WalkPlan, matrix: &BlockR1cs, w: &[&[F128]; 3]) {
    let [ea, eb, ec] = *w;
    assert_eq!(
        plan.transpose_ab(ea, eb).unwrap(),
        (
            sparse_transpose(&matrix.a_0, ea),
            sparse_transpose(&matrix.b_0, eb)
        )
    );
    assert_eq!(
        plan.apply_ab(ec).unwrap(),
        (sparse_apply(&matrix.a_0, ec), sparse_apply(&matrix.b_0, ec))
    );
    assert!(matches!(
        plan.transpose_ab(ea, &eb[..eb.len() / 2]),
        Err(WalkError::WeightCount { .. })
    ));
    assert!(matches!(
        plan.apply_ab(&ec[..3]),
        Err(WalkError::InvalidTransposeSize(3))
    ));

    let walked = matrix.clone().with_walk(Arc::new(plan.clone()));
    let walking = walked.csc_lincheck_circuit();
    let gathering = CscCircuit::from_matrices(&matrix.a_0, &matrix.b_0);
    assert!(walking.walks() && !gathering.walks());
    let alpha = F128::new(0x9abc, 0xdef0);
    assert_eq!(
        walking.fold_alpha_batched(alpha, ea),
        gathering.fold_alpha_batched(alpha, ea)
    );
    assert_eq!(walking.fold_split(eb), gathering.fold_split(eb));
    let ws = [(ea, eb), (eb, eb), (ec, ea)];
    assert_eq!(walking.fold_split_many(&ws), gathering.fold_split_many(&ws));
    for (side, m) in [(MatrixSide::A, &matrix.a_0), (MatrixSide::B, &matrix.b_0)] {
        assert_eq!(walking.row_marginal(side, ec), Some(sparse_apply(m, ec)));
        assert_eq!(gathering.row_marginal(side, ec), None);
    }
}

/// A walk relabeled by a bijection of its positions is the walk of the
/// matrices relabeled the same way (rows and columns alike); a map that
/// leaves a read position without an image, or merges two, is refused.
#[test]
fn relabeled_walk_is_the_relabeled_relation() {
    let circuit = batch_fixture(LoweringMode::RequireIdentityC);
    let plan = circuit.walk_plan().unwrap();
    let matrix = circuit.to_block_r1cs(8, 0, 0).unwrap();
    let n = 1 << matrix.k_log;
    let useful = plan.useful_bits();
    let map: Vec<usize> = (0..useful).map(|p| useful - 1 - p).collect();
    let moved = plan.relabeled(&map, useful).unwrap();
    assert_eq!(moved.useful_bits(), useful);
    assert_eq!(moved.one_position(), map[plan.one_position()]);
    let relabel = |m: &SparseBinaryMatrix| {
        let mut rows = vec![Vec::new(); n];
        for (r, row) in m.rows.iter().enumerate() {
            if !row.is_empty() {
                rows[map[r]] = row.iter().map(|&c| map[c]).collect();
            }
        }
        SparseBinaryMatrix::new(n, n, rows)
    };
    let (a, b) = (relabel(&matrix.a_0), relabel(&matrix.b_0));
    let [ea, eb, ec] = [51, 52, 53].map(|seed| weights(n, seed));
    assert_eq!(
        moved.transpose_ab(&ea, &eb).unwrap(),
        (sparse_transpose(&a, &ea), sparse_transpose(&b, &eb))
    );
    assert_eq!(
        moved.apply_ab(&ec).unwrap(),
        (sparse_apply(&a, &ec), sparse_apply(&b, &ec))
    );

    let occupied = plan.occupied_positions();
    let mut unmapped = map.clone();
    unmapped[occupied[0]] = usize::MAX;
    assert_eq!(
        plan.relabeled(&unmapped, useful).unwrap_err(),
        WalkError::Relabel(occupied[0])
    );
    let mut merged = map.clone();
    merged[occupied[0]] = merged[occupied[1]];
    assert_eq!(
        plan.relabeled(&merged, useful).unwrap_err(),
        WalkError::Relabel(usize::MAX)
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
#[path = "tests/sliced.rs"]
mod sliced;
