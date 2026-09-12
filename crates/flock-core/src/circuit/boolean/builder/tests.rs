use super::*;
use crate::circuit::boolean::tests::support::TestSchema;

#[test]
#[should_panic(expected = "stored normalized support disagrees with structural DAG")]
fn validation_rejects_corrupt_normalized_support() {
    let mut builder = CircuitBuilder::new();
    let cols = builder.reserve_columns(&TestSchema {
        inputs: 1,
        witnesses: 0,
    });
    let input = cols.input[0];
    let expression = builder.xor([input]);
    builder.expressions[expression.id.index].support.clear();
    builder.validate();
}

#[test]
#[should_panic(expected = "materialized value has multiple defining rows")]
fn validation_rejects_duplicate_definition() {
    let mut builder = CircuitBuilder::new();
    builder.reserve_columns(&TestSchema {
        inputs: 1,
        witnesses: 0,
    });
    builder.rows[1].defined_value = Some(builder.one.value);
    builder.validate();
}

#[test]
#[should_panic(expected = "row reads a value before it is defined")]
fn validation_rejects_read_before_definition() {
    let mut builder = CircuitBuilder::new();
    let cols = builder.reserve_columns(&TestSchema {
        inputs: 2,
        witnesses: 2,
    });
    let (a, b) = (cols.input[0], cols.input[1]);
    let (early, late) = (cols.witness[0].0, cols.witness[1].0);
    builder.define_and(cols.witness[0], a, b);
    builder.define_and(cols.witness[1], a, b);
    let early_row = builder
        .rows
        .iter_mut()
        .find(|row| row.defined_value == Some(early.value))
        .unwrap();
    early_row.lhs = late.expression;
    builder.validate();
}

#[test]
#[should_panic(expected = "row reads a value before it is defined")]
fn validation_checks_cancelled_structural_dependencies() {
    let mut builder = CircuitBuilder::new();
    let cols = builder.reserve_columns(&TestSchema {
        inputs: 1,
        witnesses: 2,
    });
    let a = cols.input[0];
    let (early, late) = (cols.witness[0].0, cols.witness[1].0);
    builder.define_linear(cols.witness[0], a);
    builder.define_linear(cols.witness[1], a);
    let cancelled = builder.xor3(late, a, late);
    let row = builder
        .rows
        .iter_mut()
        .find(|row| row.defined_value == Some(early.value))
        .unwrap();
    row.lhs = cancelled.id;
    builder.validate();
}
