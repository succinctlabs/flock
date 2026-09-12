use crate::circuit::boolean::tests::support::{self, TestSchema};
use crate::circuit::boolean::{
    BooleanCircuit, CircuitBuilder, ForwardTrace, LinearExpr, WalkError,
};
use crate::field::F128;
use crate::lincheck::LincheckCircuit;
use crate::r1cs::SparseBinaryMatrix;

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_circuit(seed: u64) -> BooleanCircuit {
    let mut state = seed;
    let steps: Vec<_> = (0..48)
        .map(|step| {
            let lhs = next(&mut state);
            let rhs = next(&mut state);
            let kind = next(&mut state) % 4;
            let check = (step % 11 == 0).then(|| next(&mut state));
            (lhs, rhs, kind, check)
        })
        .collect();
    let witnesses = 1 + steps.iter().filter(|step| step.2 >= 2).count();
    support::circuit(8, witnesses, |builder, cols| {
        let inputs = cols.input.iter().map(|var| var.0).collect::<Vec<_>>();
        let mut stored = cols.witness.iter().copied();
        let mut expressions: Vec<LinearExpr> = inputs.iter().map(|bit| bit.expr()).collect();

        for (lhs, rhs, kind, check) in steps {
            let lhs = expressions[lhs as usize % expressions.len()];
            let rhs = expressions[rhs as usize % expressions.len()];
            expressions.push(match kind {
                0 | 1 => builder.xor2(lhs, rhs),
                2 => {
                    let out = stored.next().unwrap();
                    builder.define_and(out, lhs, rhs);
                    out.into()
                }
                _ => {
                    let out = stored.next().unwrap();
                    builder.define_linear(out, lhs);
                    out.into()
                }
            });
            if let Some(index) = check {
                let expression = expressions[index as usize % expressions.len()];
                builder.constrain(expression, builder.one(), expression);
            }
        }
        builder.define_linear(stored.next().unwrap(), *expressions.last().unwrap());
        assert!(stored.next().is_none());
    })
}

fn capacity_log(required: usize) -> usize {
    required.next_power_of_two().trailing_zeros() as usize
}

fn weights(len: usize, seed: u64) -> Vec<F128> {
    let mut state = seed;
    (0..len)
        .map(|_| F128::new(next(&mut state), next(&mut state)))
        .collect()
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

fn add_three(mut a: Vec<F128>, b: &[F128], c: &[F128]) -> Vec<F128> {
    for ((value, &b), &c) in a.iter_mut().zip(b).zip(c) {
        *value += b;
        *value += c;
    }
    a
}

fn dot_bits(bits: &[bool], weights: &[F128]) -> F128 {
    bits.iter()
        .zip(weights)
        .filter(|(bit, _)| **bit)
        .fold(F128::ZERO, |sum, (_, &weight)| sum + weight)
}

#[path = "tests/batches.rs"]
mod batches;
#[path = "tests/transpose.rs"]
mod transpose;

#[test]
fn edge_case_xors_and_nested_general_c_match_sparse_walks() {
    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 2,
        witnesses: 2,
    };
    let cols = builder.reserve_columns(&schema);
    let inputs: Vec<_> = cols.input.iter().map(|var| var.0).collect();
    let empty = builder.xor(std::iter::empty::<LinearExpr>());
    let shared = builder.xor2(inputs[0], inputs[1]);
    let duplicate = builder.xor([shared, shared, inputs[1].expr()]);
    let single = builder.xor([duplicate]);
    let nested_c = builder.xor([shared, inputs[0].expr(), inputs[0].expr()]);
    let one = builder.one();
    builder.constrain(shared, one, nested_c);
    builder.define_linear(cols.witness[0], empty);
    builder.define_linear(cols.witness[1], single);
    let compiled = builder.finish_columns(schema);
    let inputs: Vec<_> = cols
        .input
        .iter()
        .map(|&var| support::resolved(&compiled, var))
        .collect();
    let circuit = compiled.circuit;
    let empty = support::expression(&circuit, empty);
    let duplicate = support::expression(&circuit, duplicate);
    let single = support::expression(&circuit, single);
    let nested_c = support::expression(&circuit, nested_c);
    let shared = support::expression(&circuit, shared);

    assert_eq!(circuit.support(empty.id()), Some(vec![]));
    assert_eq!(
        circuit.support(duplicate.id()),
        Some(vec![inputs[1].value_id()])
    );
    assert_eq!(
        circuit.support(single.id()),
        circuit.support(duplicate.id())
    );
    assert_eq!(circuit.support(nested_c.id()), circuit.support(shared.id()));

    let plan = circuit.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let capacity = 1 << k_log;
    let r1cs = circuit.to_block_r1cs(k_log, 0, 0).unwrap();
    for input in [[false, false], [false, true], [true, false], [true, true]] {
        let walked = plan.forward(&input, k_log).unwrap();
        assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
        assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
        assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));
    }

    let e_a = weights(capacity, 0xed6e_a001);
    let e_b = weights(capacity, 0xed6e_b001);
    let e_c = weights(capacity, 0xed6e_c001);
    let expected = add_three(
        sparse_transpose(&r1cs.a_0, &e_a),
        &sparse_transpose(&r1cs.b_0, &e_b),
        &sparse_transpose(&r1cs.c_0, &e_c),
    );
    assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);
}

#[test]
fn walks_match_sparse_oracles() {
    for seed in 0..16usize {
        let circuit = random_circuit(0xc01a_0000 + seed as u64);
        let layout = circuit.layout().unwrap();
        let plan = circuit.walk_plan_with_layout(&layout).unwrap();
        let k_log = capacity_log(plan.useful_bits());
        let capacity = 1 << k_log;
        let inputs: Vec<bool> = (0..circuit.inputs().len())
            .map(|index| (seed + index) & 1 == 1)
            .collect();
        let walked = plan.forward(&inputs, k_log).unwrap();
        assert_eq!(walked.z, circuit.evaluate_r1cs(&inputs, k_log).unwrap());
        assert_eq!(walked.z.len(), capacity);
        let r1cs = circuit
            .to_block_r1cs_with_layout(k_log, 0, 0, &layout)
            .unwrap();
        assert_eq!(walked.a_z, r1cs.apply_a(&walked.z));
        assert_eq!(walked.b_z, r1cs.apply_b(&walked.z));
        assert_eq!(walked.c_z, r1cs.apply_c(&walked.z));
        assert!(r1cs.satisfies(&walked.z));

        let e_a = weights(capacity, 0xa000 + seed as u64);
        let e_b = weights(capacity, 0xb000 + seed as u64);
        let e_c = weights(capacity, 0xc000 + seed as u64);
        let expected = add_three(
            sparse_transpose(&r1cs.a_0, &e_a),
            &sparse_transpose(&r1cs.b_0, &e_b),
            &sparse_transpose(&r1cs.c_0, &e_c),
        );
        assert_eq!(plan.transpose(&e_a, &e_b, &e_c).unwrap(), expected);
    }
}

#[test]
fn forward_walk_enforces_general_constraints() {
    let circuit = support::circuit(1, 0, |b, cols| {
        b.assert_zero(cols.input[0]);
    });
    let row = circuit.rows()[2].id();
    let plan = circuit.walk_plan().unwrap();
    let error = plan.forward(&[true], capacity_log(plan.useful_bits()));
    assert_eq!(error, Err(WalkError::UnsatisfiedRow(row)));
}

#[test]
fn deep_and_reused_xors_follow_structural_edges_and_liveness() {
    let deep = support::circuit(8, 1, |builder, cols| {
        let inputs: Vec<_> = cols.input.iter().map(|var| var.0).collect();
        let mut expression = inputs[0].expr();
        for index in 0..20_000 {
            expression = builder.xor2(expression, inputs[index % inputs.len()]);
        }
        builder.define_linear(cols.witness[0], expression);
    });
    let deep_plan = deep.walk_plan().unwrap();
    assert_eq!(deep_plan.stats().xor_nodes, 20_000);
    assert_eq!(deep_plan.stats().max_live_temporaries, 1);
    let deep_k_log = capacity_log(deep_plan.useful_bits());
    let deep_trace = deep_plan
        .forward(
            &[true, false, true, false, true, false, true, false],
            deep_k_log,
        )
        .unwrap();
    let deep_r1cs = deep.to_block_r1cs(deep_k_log, 0, 0).unwrap();
    assert_eq!(deep_trace.a_z, deep_r1cs.apply_a(&deep_trace.z));
    assert_eq!(deep_trace.b_z, deep_r1cs.apply_b(&deep_trace.z));
    assert_eq!(deep_trace.c_z, deep_r1cs.apply_c(&deep_trace.z));
    let deep_capacity = 1 << deep_k_log;
    let deep_a = weights(deep_capacity, 61);
    let deep_b = weights(deep_capacity, 62);
    let deep_c = weights(deep_capacity, 63);
    let deep_expected = add_three(
        sparse_transpose(&deep_r1cs.a_0, &deep_a),
        &sparse_transpose(&deep_r1cs.b_0, &deep_b),
        &sparse_transpose(&deep_r1cs.c_0, &deep_c),
    );
    assert_eq!(
        deep_plan.transpose(&deep_a, &deep_b, &deep_c).unwrap(),
        deep_expected
    );

    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 128,
        witnesses: 1,
    };
    let cols = builder.reserve_columns(&schema);
    let shared = builder.xor(cols.input.iter().copied());
    let one = builder.one();
    for _ in 0..128 {
        builder.constrain(shared, one, shared);
    }
    builder.define_linear(cols.witness[0], shared);
    let reused = builder.finish_columns(schema).circuit;
    let shared = support::expression(&reused, shared);
    let plan = reused.walk_plan().unwrap();
    let k_log = capacity_log(plan.useful_bits());
    let r1cs = reused.to_block_r1cs(k_log, 0, 0).unwrap();
    let sparse_nonzeros: usize = r1cs
        .a_0
        .rows
        .iter()
        .chain(&r1cs.b_0.rows)
        .chain(&r1cs.c_0.rows)
        .map(Vec::len)
        .sum();
    assert!(plan.stats().structural_edges * 20 < sparse_nonzeros);
    assert_eq!(plan.stats().max_live_temporaries, 1);
    assert_eq!(plan.expression_fanout(shared.id()), Some(128 * 2 + 1));
}
