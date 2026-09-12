use super::*;

use crate::circuit::boolean::ColumnVisitor;
use crate::circuit::boolean::tests::support::Fields;

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
    let other = CircuitBuilder::compile(TestSchema, define);
    assert_eq!(
        compiled.circuit().structure_digest(),
        other.circuit().structure_digest()
    );
    assert_eq!(other.resolve(cols.output), None);
    assert!(compiled.unused_values().is_empty());

    // Independent equations: ONE, input, copied input, copied input AND input.
    let matrix = compiled.circuit().to_block_r1cs(2, 0, 0).unwrap();
    assert_eq!(*matrix.a_0.rows, vec![vec![0], vec![1], vec![1], vec![2]]);
    assert_eq!(*matrix.b_0.rows, vec![vec![0], vec![0], vec![0], vec![1]]);
    assert_eq!(*matrix.c_0.rows, vec![vec![0], vec![1], vec![2], vec![3]]);
    for input in [false, true] {
        let (cols, values) = compiled
            .evaluate(|cols| {
                *cols.input = input;
                *cols.late = !input;
                *cols.output = !input;
            })
            .unwrap();
        assert_eq!([cols.input, cols.late, cols.output], [input; 3]);
        assert_eq!(values, [true, input, input, input]);
        assert!(matrix.satisfies(&values));
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
fn fixed_columns_and_general_constraints_survive_remapping() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("output", ColumnRole::Output, 1, 1),
            ("input", ColumnRole::Input, 1, 1),
            ("one", ColumnRole::Fixed(true), 1, 1),
            ("zero", ColumnRole::Fixed(false), 1, 1),
        ]),
        |b, cols| {
            let expression = b.xor3(cols[1][0], cols[2][0], cols[3][0]);
            b.define_linear(cols[0][0], expression);
            b.assert_zero(cols[0][0]);
        },
    );
    // Only input is supplied; writes to output and both fixed bits are ignored.
    let (cols, mut values) = compiled
        .evaluate(|mut cols| {
            *cols[0][0] = true;
            *cols[1][0] = true;
            *cols[2][0] = false;
            *cols[3][0] = true;
        })
        .unwrap();
    assert_eq!(cols, [vec![false], vec![true], vec![true], vec![false]]);
    assert!(compiled.evaluate(|_| {}).is_err());
    values.resize(8, false);
    let matrix = compiled.circuit().to_block_r1cs(3, 0, 0).unwrap();
    assert!(matrix.satisfies(&values));
    for (field, role) in [(2, ColumnRole::Fixed(true)), (3, ColumnRole::Fixed(false))] {
        assert_eq!(compiled.schema()[field].role, role);
        let position = compiled.columns()[field][0].index();
        values[position] ^= true;
        assert!(!matrix.satisfies(&values));
        values[position] ^= true;
    }
}
