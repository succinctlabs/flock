use super::*;

#[test]
#[should_panic(expected = "stored normalized support disagrees with structural DAG")]
fn validation_rejects_corrupt_normalized_support() {
    let mut builder = CircuitBuilder::new();
    let input = builder.input();
    let expression = builder.xor([input]);
    builder.expressions[expression.id.index].support.clear();
    builder.finish();
}

#[test]
#[should_panic(expected = "materialized value has multiple defining rows")]
fn validation_rejects_duplicate_definition() {
    let mut builder = CircuitBuilder::new();
    builder.input();
    builder.rows[1].defined_value = Some(builder.one.value);
    builder.finish();
}

#[test]
#[should_panic(expected = "row reads a value before it is defined")]
fn validation_rejects_read_before_definition() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let early = builder.and(a, b);
    let late = builder.and(a, b);
    let early_row = builder
        .rows
        .iter_mut()
        .find(|row| row.defined_value == Some(early.value))
        .unwrap();
    early_row.lhs = late.expression;
    builder.finish();
}

#[test]
#[should_panic(expected = "row reads a value before it is defined")]
fn validation_checks_cancelled_structural_dependencies() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let early = builder.materialize(a);
    let late = builder.materialize(a);
    let cancelled = builder.xor3(late, a, late);
    let row = builder
        .rows
        .iter_mut()
        .find(|row| row.defined_value == Some(early.value))
        .unwrap();
    row.lhs = cancelled.id;
    builder.finish();
}
