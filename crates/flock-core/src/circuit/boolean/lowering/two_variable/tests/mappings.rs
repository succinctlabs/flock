use super::*;
use crate::circuit::boolean::{
    ColumnRole, ColumnSchema, ColumnVisitor, InteractionDirection, InteractionField,
    InteractionScope, Var,
};

struct Schema;
struct Cols<T> {
    input: T,
    advice: T,
    left: T,
    right: T,
}

impl ColumnSchema for Schema {
    type Cols<T> = Cols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Cols<V::Value> {
        Cols {
            input: v.bit("input", ColumnRole::Input),
            advice: v.bit("advice", ColumnRole::Advice("claimed-bit")),
            left: v.bit("parent.left.out", ColumnRole::Witness),
            right: v.bit("parent.right.out", ColumnRole::Output),
        }
    }
}

fn eval(b: &mut CircuitBuilder, cols: &Cols<Var>) {
    b.operation(
        "Parent",
        &[cols.left, cols.right],
        [
            ("input", vec![cols.input.into()]),
            ("advice", vec![cols.advice.into()]),
        ],
        |b| {
            let left = b.operation(
                "Copy",
                &[cols.left],
                [
                    ("input", vec![cols.input.into()]),
                    ("advice", vec![cols.advice.into()]),
                ],
                |b| {
                    b.define_linear(cols.left, cols.input);
                    b.constrain(cols.left, b.one(), cols.advice);
                    [cols.left.into()]
                },
            );
            b.operation("Copy", &[cols.right], [("input", left.to_vec())], |b| {
                b.define_linear(cols.right, left[0]);
                [cols.right.into()]
            })
        },
    );
    b.interaction(
        "test",
        "output",
        InteractionDirection::Send,
        [InteractionField::columns("value", 1, [cols.right])],
        [b.one()],
        b.column_selector(cols.input),
        InteractionScope::new("Nested", 0),
    );
}

#[test]
fn maps_nested_operations_columns_advice_and_interactions_without_changing_the_schema() {
    let compiled = CircuitBuilder::compile(Schema, eval);
    let source = compiled.circuit();
    let lowered = source.lower_identity_c().unwrap();
    assert_eq!(
        lowered.inputs(),
        source
            .inputs()
            .iter()
            .map(|&id| lowered.mapped_value(id).unwrap())
            .collect::<Vec<_>>()
    );
    for (old, new) in source.schema().iter().zip(lowered.schema()) {
        assert_eq!(old.name, new.name);
        assert_eq!(old.role, new.role);
        assert_eq!(old.alignment_bits, new.alignment_bits);
        assert_eq!(
            new.values,
            old.values
                .iter()
                .map(|&id| lowered.mapped_value(id).unwrap())
                .collect::<Vec<_>>()
        );
    }
    for operation in compiled.operations() {
        let mapped: Vec<_> = source.rows()[operation.rows.clone()]
            .iter()
            .flat_map(|row| lowered.mapped_rows(row.id()).unwrap())
            .collect();
        let extra = source.rows()[operation.rows.clone()]
            .iter()
            .filter(|row| row.kind() == RowKind::Constraint)
            .count();
        assert_eq!(mapped.len(), operation.rows.len() + extra);
        for word in operation
            .inputs
            .iter()
            .chain(std::iter::once(&operation.output))
        {
            for &expr in &word.expressions {
                assert!(
                    lowered
                        .expression(lowered.mapped_expression(expr).unwrap())
                        .is_some()
                );
            }
        }
        for &column in &operation.columns {
            assert!(lowered.mapped_value(column).is_some());
        }
    }
    let (old, new) = (&source.interactions()[0], &lowered.interactions()[0]);
    assert_eq!(old.channel(), new.channel());
    assert_eq!(old.kind(), new.kind());
    assert_eq!(old.direction(), new.direction());
    assert_eq!(old.scope(), new.scope());
    assert_eq!(
        new.selector(),
        lowered.mapped_value(old.selector()).unwrap()
    );
    assert_eq!(
        new.multiplicity(),
        old.multiplicity()
            .iter()
            .map(|&v| lowered.mapped_value(v).unwrap())
            .collect::<Vec<_>>()
    );
    for (old, new) in old.message().iter().zip(new.message()) {
        assert_eq!(old.name(), new.name());
        assert_eq!(old.encoding(), new.encoding());
        assert_eq!(
            new.values(),
            old.values()
                .iter()
                .map(|&v| lowered.mapped_value(v).unwrap())
                .collect::<Vec<_>>()
        );
    }
    for value in [false, true] {
        let (cols, witness) = compiled
            .evaluate(|cols| {
                *cols.input = value;
                *cols.advice = value;
            })
            .unwrap();
        let extension = lowered.evaluate(&[value, value]).unwrap();
        assert_eq!(lowered.project(&extension).unwrap(), witness);
        assert_eq!(
            extension[lowered
                .mapped_value(compiled.columns().right)
                .unwrap()
                .index()],
            cols.right
        );
        assert_eq!(
            old.effective_multiplicity_bits(&witness),
            new.effective_multiplicity_bits(&extension)
        );
    }
    let foreign = CircuitBuilder::compile(Schema, eval);
    assert!(lowered.mapped_value(foreign.circuit().one()).is_none());
    assert!(
        lowered
            .mapped_rows(foreign.circuit().rows()[0].id())
            .is_none()
    );
    assert!(
        lowered
            .mapped_expression(foreign.circuit().rows()[0].lhs())
            .is_none()
    );
    assert!(lowered.mapped_value(lowered.one()).is_none());
    assert!(lowered.mapped_rows(lowered.rows()[0].id()).is_none());
    assert!(lowered.source_row(source.rows()[0].id()).is_none());
    assert!(lowered.layout().value_position(source.one()).is_none());
}
