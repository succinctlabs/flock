use super::*;

mod alignment;
mod configuration;
use crate::circuit::boolean::{ColumnVisitor, ExpressionNode};

struct TestSchema;
struct TestCols<T> {
    output: T,
    late: T,
    input: T,
}

impl ColumnSchema for TestSchema {
    type Cols<T> = TestCols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> TestCols<V::Value> {
        TestCols {
            output: v.bit("output", ColumnRole::Output),
            late: v.bit("late", ColumnRole::Witness),
            input: v.bit("input", ColumnRole::Input),
        }
    }
}

fn define(b: &mut CircuitBuilder, cols: &TestCols<Var>) {
    b.define_linear(cols.late, cols.input);
    b.define_and(cols.output, cols.late, cols.input);
}

#[test]
fn retained_handles_resolve_and_all_final_references_use_definition_order() {
    let mut b = CircuitBuilder::new();
    let cols = b.reserve_columns(&TestSchema);
    let stale = cols.output.0.value_id();
    define(&mut b, &cols);
    let compiled = b.finish_columns(TestSchema);
    let resolved = compiled.columns();
    assert_eq!(resolved.input.index(), 1);
    assert_eq!(resolved.late.index(), 2);
    assert_eq!(resolved.output.index(), 3);
    assert_eq!(compiled.resolve(cols.output), Some(resolved.output));
    let layout = compiled.circuit().layout().unwrap();
    assert_eq!(layout.value_position(stale), None);
    assert_eq!(layout.value_position(resolved.output), Some(3));
    assert_eq!(
        compiled.circuit().column("output").unwrap().values,
        &[resolved.output]
    );
    assert_eq!(
        compiled
            .circuit()
            .definition_row(resolved.output)
            .unwrap()
            .index(),
        3
    );
    for (index, row) in compiled.circuit().rows.iter().enumerate() {
        assert_eq!(row.defined_value.unwrap().index(), index);
    }
    let (row, values) = compiled.evaluate(|cols| *cols.input = true).unwrap();
    assert!(row.input && row.late && row.output);
    assert_eq!(values, [true, true, true, true]);
    let other = CircuitBuilder::compile(TestSchema, define);
    assert_eq!(
        compiled.circuit().structure_digest(),
        other.circuit().structure_digest()
    );
    assert_eq!(other.resolve(cols.output), None);
    assert!(compiled.unused_values().is_empty());
}

#[test]
fn canonical_schema_relation_matches_expected_equations() {
    let compiled = CircuitBuilder::compile(TestSchema, define);
    let new = compiled.circuit().to_block_r1cs(2, 0, 0).unwrap();
    // ONE, input, copied input, then copied input AND input.
    assert_eq!(new.a_0.rows, vec![vec![0], vec![1], vec![1], vec![2]]);
    assert_eq!(new.b_0.rows, vec![vec![0], vec![0], vec![0], vec![1]]);
    assert_eq!(new.c_0.rows, vec![vec![0], vec![1], vec![2], vec![3]]);
    assert_eq!(compiled.columns().input.index(), 1);
    assert_eq!(compiled.columns().output.index(), 3);
    for input in [false, true] {
        let (_, values) = compiled.evaluate(|cols| *cols.input = input).unwrap();
        assert_eq!(values, vec![true, input, input, input]);
        let walked = compiled
            .circuit()
            .walk_plan()
            .unwrap()
            .forward(&[input], 2)
            .unwrap();
        assert_eq!(walked.z, values);
        assert_eq!(walked.a_z, new.apply_a(&values));
        assert_eq!(walked.b_z, new.apply_b(&values));
        assert_eq!(walked.c_z, new.apply_c(&values));
    }
}

#[test]
#[should_panic(expected = "structural read before definition")]
fn cancelled_forward_dependency_is_rejected() {
    let mut b = CircuitBuilder::new();
    let cols = b.reserve_columns(&TestSchema);
    // Even late XOR late must not be scheduled before late's definition.
    b.xor([
        LinearExpr::from(cols.late),
        cols.input.into(),
        cols.late.into(),
    ]);
}

#[test]
#[should_panic(expected = "materialized value is missing its defining row")]
fn missing_definition_is_rejected() {
    let mut b = CircuitBuilder::new();
    b.reserve_columns(&TestSchema);
    b.finish_columns(TestSchema);
}

#[test]
#[should_panic(expected = "column defined twice")]
fn duplicate_definition_is_rejected() {
    let mut b = CircuitBuilder::new();
    let cols = b.reserve_columns(&TestSchema);
    b.define_linear(cols.late, cols.input);
    b.define_linear(cols.late, cols.input);
}

#[test]
#[should_panic(expected = "expression belongs to another circuit builder")]
fn foreign_handle_is_rejected() {
    let mut b = CircuitBuilder::new();
    let cols = b.reserve_columns(&TestSchema);
    let mut other = CircuitBuilder::new();
    let foreign = other.reserve_columns(&TestSchema);
    b.define_linear(cols.late, foreign.input);
}

#[test]
#[should_panic(expected = "columns must be reserved on a fresh builder")]
fn columns_cannot_be_reserved_twice() {
    let mut b = CircuitBuilder::new();
    b.reserve_columns(&TestSchema);
    b.reserve_columns(&TestSchema);
}

#[test]
fn unused_values_are_reported_without_pruning() {
    let compiled = CircuitBuilder::compile(TestSchema, |b, cols| {
        b.define_linear(cols.late, cols.input);
        b.define_linear(cols.output, cols.input);
    });
    assert_eq!(compiled.unused_values(), [("late", 0)]);
    assert_eq!(compiled.circuit().value_count(), 4);
    assert_eq!(
        compiled
            .circuit()
            .expressions()
            .filter(|e| matches!(e, ExpressionNode::Value(_)))
            .count(),
        4
    );
}

#[test]
fn fixed_columns_and_general_constraints_survive_remapping() {
    struct FixedSchema;
    impl ColumnSchema for FixedSchema {
        type Cols<T> = [T; 3];
        fn columns<V: ColumnVisitor>(&self, v: &mut V) -> [V::Value; 3] {
            [
                v.bit("output", ColumnRole::Output),
                v.bit("input", ColumnRole::Input),
                v.bit("fixed", ColumnRole::Fixed(true)),
            ]
        }
    }
    let compiled = CircuitBuilder::compile(FixedSchema, |b, cols| {
        let expression = b.xor2(cols[1], cols[2]);
        b.define_linear(cols[0], expression);
        b.assert_zero(cols[0]);
    });
    let (row, mut values) = compiled.evaluate(|cols| *cols[1] = true).unwrap();
    assert_eq!(row, [false, true, true]);
    assert!(compiled.evaluate(|_| {}).is_err());
    values.resize(8, false);
    let r1cs = compiled.circuit().to_block_r1cs(3, 0, 0).unwrap();
    assert!(r1cs.satisfies(&values));
    values[compiled.columns()[2].index()] = false;
    assert!(!r1cs.satisfies(&values));
}

#[test]
fn malformed_schema_fields_are_rejected() {
    struct Bad<const CASE: u8>;
    impl<const CASE: u8> ColumnSchema for Bad<CASE> {
        type Cols<T> = [T; 2];
        fn columns<V: ColumnVisitor>(&self, v: &mut V) -> [V::Value; 2] {
            [
                v.bit(if CASE == 0 { "" } else { "a" }, ColumnRole::Input),
                v.bit(
                    if CASE == 1 { "a" } else { "b" },
                    if CASE == 2 {
                        ColumnRole::Advice("")
                    } else {
                        ColumnRole::Input
                    },
                ),
            ]
        }
    }
    assert!(std::panic::catch_unwind(|| CircuitBuilder::compile(Bad::<0>, |_, _| {})).is_err());
    assert!(std::panic::catch_unwind(|| CircuitBuilder::compile(Bad::<1>, |_, _| {})).is_err());
    assert!(std::panic::catch_unwind(|| CircuitBuilder::compile(Bad::<2>, |_, _| {})).is_err());
}
