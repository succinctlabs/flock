use super::*;
use support::TestSchema;

#[path = "tests/support.rs"]
pub(super) mod support;

#[path = "tests/placement.rs"]
mod placement;
#[path = "tests/relation.rs"]
mod relation;

#[test]
fn structural_dag_survives_normalization_and_cancellation() {
    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 2,
        witnesses: 0,
    };
    let cols = builder.reserve_columns(&schema);
    let (a, b) = (cols.input[0].0, cols.input[1].0);
    let ab = builder.xor([a.expr(), b.expr()]);
    let nested = builder.xor([ab, a.expr()]);
    let compiled = builder.finish_columns(schema);
    let a = support::resolved(&compiled, cols.input[0]);
    let b = support::resolved(&compiled, cols.input[1]);
    let circuit = compiled.circuit;
    let nested_id = support::expression(&circuit, nested).id();
    let ab_id = support::expression(&circuit, ab).id();

    assert_eq!(
        circuit.expression(nested_id),
        Some(&ExpressionNode::Xor(vec![ab_id, a.expr().id()]))
    );
    assert_eq!(circuit.support(nested_id), Some(vec![b.value_id()]));

    let expressions: Vec<_> = circuit.expressions().collect();
    assert_eq!(expressions.len(), circuit.expression_count());
    assert_eq!(
        expressions[nested_id.index()],
        circuit.expression(nested_id).unwrap()
    );

    let layout = circuit.layout().unwrap();
    assert_eq!(layout.value_positions(), &[0, 1, 2]);
    assert_eq!(layout.row_positions(), &[0, 1, 2]);
}

#[test]
fn materialization_adds_only_the_requested_boundary() {
    let [virtual_sum, stored_sum] = [false, true].map(|materialize| {
        support::circuit(2, 1 + usize::from(materialize), |b, cols| {
            let before = (b.value_count(), b.row_count());
            let sum = b.xor2(cols.input[0], cols.input[1]);
            assert_eq!((b.value_count(), b.row_count()), before);
            let lhs = if materialize {
                b.define_linear(cols.witness[0], sum);
                cols.witness[0].into()
            } else {
                sum
            };
            b.define_and(*cols.witness.last().unwrap(), lhs, cols.input[0]);
            assert_eq!(b.value_count(), before.0);
            assert_eq!(b.row_count(), before.1 + 1 + usize::from(materialize));
        })
    });
    assert_eq!(stored_sum.value_count(), virtual_sum.value_count() + 1);
    assert_eq!(stored_sum.row_count(), virtual_sum.row_count() + 1);
    for (circuit, first_kind) in [
        (&virtual_sum, RowKind::And),
        (&stored_sum, RowKind::Materialize),
    ] {
        let rows = &circuit.rows()[3..];
        assert_eq!(rows[0].kind(), first_kind);
        assert_eq!(
            circuit.support(rows[0].lhs()),
            Some(circuit.inputs().to_vec())
        );
        assert_eq!(rows.last().unwrap().kind(), RowKind::And);
        let matrix = circuit.to_block_r1cs(3, 0, 0).unwrap();
        let product = rows.last().unwrap().defined_value().unwrap();
        for input in [[false, false], [false, true], [true, false], [true, true]] {
            let witness = circuit.evaluate_r1cs(&input, 3).unwrap();
            let xor = input[0] ^ input[1];
            assert_eq!(witness[product.index()], xor & input[0]);
            assert_eq!(matrix.apply_a(&witness)[product.index()], xor);
            assert_eq!(matrix.apply_b(&witness)[product.index()], input[0]);
            if first_kind == RowKind::Materialize {
                assert_eq!(witness[rows[0].defined_value().unwrap().index()], xor);
            }
            assert!(matrix.satisfies(&witness));
        }
    }
    let copy = &stored_sum.rows()[3];
    assert_eq!(
        stored_sum.support(stored_sum.rows()[4].lhs()),
        Some(vec![copy.defined_value().unwrap()])
    );
    assert_ne!(
        virtual_sum.structure_digest(),
        stored_sum.structure_digest()
    );
    assert_ne!(
        virtual_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest(),
        stored_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest()
    );
}
