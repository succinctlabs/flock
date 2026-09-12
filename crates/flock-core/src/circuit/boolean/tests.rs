use super::*;
use support::{Fields, TestSchema};

#[path = "tests/support.rs"]
pub(super) mod support;

#[path = "tests/placement.rs"]
mod placement;
#[path = "tests/relation.rs"]
mod relation;

#[test]
fn materialization_is_explicit_and_exactly_one_row() {
    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 2,
        witnesses: 2,
    };
    let cols = builder.reserve_columns(&schema);
    let (a, b) = (cols.input[0].0, cols.input[1].0);
    let initial_values = builder.value_count();
    let initial_rows = builder.row_count();

    let sum = builder.xor([a.expr(), b.expr()]);
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows);

    let sum_bit = cols.witness[0];
    builder.define_linear(sum_bit, sum);
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows + 1);
    builder.define_and(cols.witness[1], sum_bit, a);
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows + 2);

    let compiled = builder.finish_columns(schema);
    let sum_bit = compiled.resolve(sum_bit).unwrap();
    let circuit = compiled.circuit;
    let defining_row = circuit
        .rows()
        .iter()
        .find(|row| row.defined_value() == Some(sum_bit))
        .unwrap();
    assert_eq!(defining_row.kind(), RowKind::Materialize);
}

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
fn changing_materialization_changes_only_the_requested_boundary() {
    fn build(materialize: bool) -> BooleanCircuit {
        support::circuit(2, 1 + usize::from(materialize), |builder, cols| {
            let (a, b) = (cols.input[0], cols.input[1]);
            let sum = builder.xor2(a, b);
            let lhs = if materialize {
                builder.define_linear(cols.witness[0], sum);
                cols.witness[0].into()
            } else {
                sum
            };
            builder.define_and(*cols.witness.last().unwrap(), lhs, a);
        })
    }

    let virtual_sum = build(false);
    let materialized_sum = build(true);
    assert_eq!(
        materialized_sum.value_count(),
        virtual_sum.value_count() + 1
    );
    assert_eq!(materialized_sum.row_count(), virtual_sum.row_count() + 1);
    assert_eq!(
        materialized_sum
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Materialize)
            .count(),
        1
    );
    let virtual_product = &virtual_sum.rows()[3];
    assert_eq!(virtual_product.kind(), RowKind::And);
    assert_eq!(
        virtual_sum.support(virtual_product.lhs()),
        Some(vec![virtual_sum.inputs()[0], virtual_sum.inputs()[1]])
    );
    let copy = &materialized_sum.rows()[3];
    let materialized_product = &materialized_sum.rows()[4];
    assert_eq!(copy.kind(), RowKind::Materialize);
    assert_eq!(materialized_product.kind(), RowKind::And);
    assert_eq!(
        materialized_sum.support(copy.lhs()),
        Some(vec![
            materialized_sum.inputs()[0],
            materialized_sum.inputs()[1]
        ])
    );
    assert_eq!(
        materialized_sum.support(materialized_product.lhs()),
        Some(vec![copy.defined_value().unwrap()])
    );
    assert_ne!(
        virtual_sum.structure_digest(),
        materialized_sum.structure_digest()
    );
    assert_ne!(
        virtual_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest(),
        materialized_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest()
    );
}

#[test]
fn ripple_carry_example_keeps_linear_carries_virtual() {
    fn full_adder(
        builder: &mut CircuitBuilder,
        a: LinearExpr,
        b: LinearExpr,
        carry_in: LinearExpr,
        products: [Var; 2],
    ) -> (LinearExpr, LinearExpr) {
        let a_xor_b = builder.xor([a, b]);
        let sum = builder.xor([a_xor_b, carry_in]);
        builder.define_and(products[0], a, b);
        builder.define_and(products[1], carry_in, a_xor_b);
        let carry_out = builder.xor(products);
        (sum, carry_out)
    }

    let compiled = support::compile(5, 7, |builder, cols| {
        let input = &cols.input;
        let products = &cols.witness;
        let (sum_0, carry_0) = full_adder(
            builder,
            input[0].into(),
            input[2].into(),
            input[4].into(),
            [products[0], products[1]],
        );
        let rows_after_low_bit = builder.row_count();
        let (sum_1, carry_1) = full_adder(
            builder,
            input[1].into(),
            input[3].into(),
            carry_0,
            [products[2], products[3]],
        );
        // Only the two nonlinear products need rows; the carry stays an expression.
        assert_eq!(builder.row_count(), rows_after_low_bit + 2);
        builder.define_linear(products[4], sum_0);
        builder.define_linear(products[5], sum_1);
        builder.define_linear(products[6], carry_1);
    });
    let [sum_0, sum_1, carry_1] = compiled.columns().witness[4..] else {
        unreachable!()
    };
    let circuit = compiled.circuit;
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    assert!(r1cs.c0_is_identity());

    for input in 0u8..32 {
        let bits: [bool; 5] = std::array::from_fn(|i| (input >> i) & 1 == 1);
        let witness = circuit.evaluate_r1cs(&bits, 4).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output = u8::from(witness[sum_0.index()])
            | (u8::from(witness[sum_1.index()]) << 1)
            | (u8::from(witness[carry_1.index()]) << 2);
        let a_value = (input & 1) | (((input >> 1) & 1) << 1);
        let b_value = ((input >> 2) & 1) | (((input >> 3) & 1) << 1);
        let carry_value = (input >> 4) & 1;
        assert_eq!(output, a_value + b_value + carry_value);
    }
}

#[test]
fn word_helpers_are_virtual_until_defined() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("input", ColumnRole::Input, 8, 1),
            ("output", ColumnRole::Output, 8, 1),
        ]),
        |builder, cols| {
            let input: [_; 8] = cols[0].clone().try_into().unwrap();
            let values_before = builder.value_count();
            let rows_before = builder.row_count();
            let rotated = builder.rotate_right(input, 2);
            let shifted = builder.shift_right(input, 3);
            let mixed = builder.xor2_words(rotated, shifted);
            assert_eq!(builder.value_count(), values_before);
            assert_eq!(builder.row_count(), rows_before);

            for (&output, value) in cols[1].iter().zip(mixed) {
                builder.define_linear(output, value);
            }
            assert_eq!(builder.value_count(), values_before);
            assert_eq!(builder.row_count(), rows_before + 8);
        },
    );
    let output = compiled.columns().remove(1);
    let circuit = compiled.circuit;
    assert_eq!(circuit.column("input").unwrap().alignment_bits, 1);
    assert_eq!(circuit.column("output").unwrap().alignment_bits, 1);
    let r1cs = circuit.to_block_r1cs(5, 0, 0).unwrap();

    for value in 0u8..=u8::MAX {
        let input_bits: [bool; 8] = std::array::from_fn(|i| value >> i & 1 == 1);
        let witness = circuit.evaluate_r1cs(&input_bits, 5).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output_value = output.iter().enumerate().fold(0u8, |acc, (i, bit)| {
            acc | (u8::from(witness[bit.index()]) << i)
        });
        assert_eq!(output_value, value.rotate_right(2) ^ (value >> 3));
    }
}

#[test]
fn fixed_words_are_named_and_constrained() {
    struct FixedByte;
    impl ColumnSchema for FixedByte {
        type Cols<T> = [T; 8];
        fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
            std::array::from_fn(|i| {
                v.bit(&format!("iv.{i}"), ColumnRole::Fixed(0xa5 >> i & 1 == 1))
            })
        }
    }
    let compiled = CircuitBuilder::compile(FixedByte, |_, _| {});
    let fixed = compiled.columns();
    let circuit = compiled.circuit;
    assert_eq!(
        circuit.column("iv.0").unwrap().role,
        ColumnRole::Fixed(true)
    );
    assert_eq!(circuit.column("iv.0").unwrap().alignment_bits, 1);
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    let witness = circuit.evaluate_r1cs(&[], 4).unwrap();
    assert!(r1cs.satisfies(&witness));
    for (i, bit) in fixed.iter().enumerate() {
        assert_eq!(witness[bit.index()], 0xa5 >> i & 1 == 1);
    }
    let mut mutated = witness;
    mutated[fixed[3].index()] ^= true;
    assert!(!r1cs.satisfies(&mutated));
}

#[test]
fn alignment_is_part_of_the_structure_digest() {
    fn word(alignment_bits: usize) -> BooleanCircuit {
        CircuitBuilder::compile(
            Fields(vec![("value", ColumnRole::Input, 8, alignment_bits)]),
            |_, _| {},
        )
        .circuit
    }

    let unaligned_word = word(1);
    let aligned_word = word(8);
    assert_ne!(
        unaligned_word.structure_digest(),
        aligned_word.structure_digest()
    );
}

#[test]
fn structure_and_relation_digests_are_deterministic_and_golden() {
    fn build() -> BooleanCircuit {
        CircuitBuilder::compile(
            Fields(vec![
                ("input", ColumnRole::Input, 2, 1),
                ("sum", ColumnRole::Output, 1, 1),
            ]),
            |builder, cols| {
                let input = &cols[0];
                let sum = builder.xor2(input[0], input[1]);
                builder.define_linear(cols[1][0], sum);
                builder.assert_zero_product(input[0], input[1]);
            },
        )
        .circuit
    }

    let first = build();
    let second = build();
    assert_eq!(first.structure_digest(), second.structure_digest());
    let first_layout = first.layout().unwrap();
    let second_layout = second.layout().unwrap();
    assert_eq!(first_layout, second_layout);
    assert_eq!(
        first.structure_layout_digest(&first_layout).unwrap(),
        second.structure_layout_digest(&second_layout).unwrap()
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        second.to_block_r1cs(3, 0, 0).unwrap().statement_digest()
    );
    assert_eq!(
        first.structure_digest(),
        [
            230, 55, 97, 65, 189, 233, 127, 33, 212, 92, 20, 80, 247, 149, 232, 116, 18, 123, 152,
            202, 167, 2, 99, 99, 2, 214, 88, 23, 113, 95, 196, 87,
        ]
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        [
            106, 16, 156, 236, 111, 130, 214, 34, 173, 41, 61, 220, 52, 226, 61, 162, 41, 193, 169,
            235, 128, 224, 120, 125, 227, 239, 50, 114, 147, 168, 88, 148,
        ]
    );
}
