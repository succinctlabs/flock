use crate::circuit::boolean::tests::support::Fields;
use crate::circuit::boolean::{
    CircuitBuilder, ColumnRole, ColumnSchema, ColumnVisitor, InteractionDirection,
    InteractionField, InteractionScope, Var,
};

struct Nested {
    direct_field: bool,
}

const PAIR: Nested = Nested {
    direct_field: false,
};

struct Cols<T> {
    input: T,
    left: T,
    right: T,
    direct: Option<T>,
}

impl ColumnSchema for Nested {
    type Cols<T> = Cols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Cols<V::Value> {
        Cols {
            input: v.bit("input", ColumnRole::Input),
            left: v.bit("left", ColumnRole::Witness),
            right: v.bit("right", ColumnRole::Witness),
            direct: self
                .direct_field
                .then(|| v.bit("direct", ColumnRole::Witness)),
        }
    }
}

fn eval(b: &mut CircuitBuilder, cols: &Cols<Var>) {
    let owned: Vec<_> = [cols.left, cols.right]
        .into_iter()
        .chain(cols.direct)
        .collect();
    b.operation(
        "Parent",
        &owned,
        [("input", vec![cols.input.into()])],
        |b| {
            // Define children in reverse schema order to exercise finalization remapping.
            let right = b.operation(
                "Not",
                &[cols.right],
                [("input", vec![cols.input.into()])],
                |b| {
                    let value = b.xor2(cols.input, b.one());
                    b.define_linear(cols.right, value);
                    [cols.right.into()]
                },
            );
            let left = b.operation("Copy", &[cols.left], [("input", right.to_vec())], |b| {
                b.define_linear(cols.left, right[0]);
                [cols.left.into()]
            });
            if let Some(out) = cols.direct {
                b.define_linear(out, left[0]);
            }
            left
        },
    );
}

fn check_nested(direct_field: bool) {
    let compiled = CircuitBuilder::compile(Nested { direct_field }, eval);
    let cols = compiled.columns();
    let ops = compiled.operations();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0].kind, "Not");
    assert_eq!(ops[1].kind, "Copy");
    assert_eq!(ops[2].kind, "Parent");
    assert_eq!(ops[0].columns, [cols.right]);
    assert_eq!(ops[1].columns, [cols.left]);
    assert_eq!(
        ops[2].columns,
        [cols.left, cols.right]
            .into_iter()
            .chain(cols.direct)
            .collect::<Vec<_>>()
    );
    // ONE and input rows precede the child definitions.
    assert_eq!(ops[0].rows, 2..3);
    assert_eq!(ops[1].rows, 3..4);
    assert_eq!(ops[2].rows, 2..4 + usize::from(direct_field));
    assert!(cols.right.index() < cols.left.index());
    assert_eq!(ops[0].output.expressions, ops[1].inputs[0].expressions);
    assert_eq!(ops[1].output.expressions, ops[2].output.expressions);
    for op in ops {
        for word in op.inputs.iter().chain(std::iter::once(&op.output)) {
            for &expression in &word.expressions {
                assert!(compiled.circuit().support(expression).is_some());
            }
        }
    }
    let relation = compiled.circuit().to_block_r1cs(3, 0, 0).unwrap();
    for input in [false, true] {
        let (trace, mut witness) = compiled.evaluate(|cols| *cols.input = input).unwrap();
        assert_eq!(trace.left, !input);
        assert_eq!(trace.right, !input);
        assert_eq!(trace.direct, direct_field.then_some(!input));
        witness.resize(8, false);
        assert!(relation.satisfies(&witness));
    }
}

#[test]
fn parent_can_own_two_child_operations() {
    check_nested(false);
}

#[test]
fn parent_can_start_with_nested_fields_before_a_direct_field() {
    check_nested(true);
}

#[test]
#[should_panic(expected = "operation columns already defined")]
fn nested_parent_cannot_be_reused() {
    CircuitBuilder::compile(Nested { direct_field: true }, |b, cols| {
        eval(b, cols);
        eval(b, cols);
    });
}

#[test]
#[should_panic(expected = "operation result must be nonempty")]
fn variable_width_operation_cannot_return_an_empty_word() {
    use crate::circuit::boolean::LinearExpr;

    CircuitBuilder::compile(
        Fields(vec![("copy.out", ColumnRole::Witness, 1, 1)]),
        |b, cols| {
            b.operation("Copy", &cols[0], [], |b| {
                b.define_linear(cols[0][0], b.one());
                Vec::<LinearExpr>::new()
            });
        },
    );
}

#[test]
fn repeated_kinds_can_define_parts_of_one_field_in_any_order() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("input", ColumnRole::Input, 1, 1),
            ("outputs", ColumnRole::Witness, 2, 1),
        ]),
        |b, cols| {
            for &output in cols[1].iter().rev() {
                b.operation(
                    "Copy",
                    &[output],
                    [("input", vec![cols[0][0].into()])],
                    |b| {
                        b.define_linear(output, cols[0][0]);
                        [output.into()]
                    },
                );
            }
        },
    );
    let cols = compiled.columns();
    let ops = compiled.operations();
    assert_eq!(ops.len(), 2);
    assert!(ops.iter().all(|op| op.kind == "Copy"));
    assert_eq!(ops[0].columns, [cols[1][1]]);
    assert_eq!(ops[1].columns, [cols[1][0]]);
    assert_eq!(ops[0].rows, 2..3);
    assert_eq!(ops[1].rows, 3..4);
    for input in [false, true] {
        let (cols, values) = compiled.evaluate(|mut cols| *cols[0][0] = input).unwrap();
        assert_eq!(cols[1], [input, input]);
        assert!(
            compiled
                .circuit()
                .to_block_r1cs(2, 0, 0)
                .unwrap()
                .satisfies(&values)
        );
    }
}

#[test]
#[should_panic(expected = "duplicate operation column")]
fn duplicate_columns_are_rejected() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation("Copy", &[cols.left, cols.left], [], |_| [cols.input.into()]);
    });
}

#[test]
#[should_panic(expected = "expression belongs to another circuit builder")]
fn foreign_columns_are_rejected() {
    CircuitBuilder::compile(PAIR, |_, foreign| {
        CircuitBuilder::compile(PAIR, |b, cols| {
            b.operation("Copy", &[foreign.left], [], |_| [cols.input.into()]);
        });
    });
}

#[test]
#[should_panic(expected = "operation columns already defined")]
fn supplied_columns_cannot_be_owned() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation("Copy", &[cols.input], [], |_| [cols.input.into()]);
    });
}

#[test]
#[should_panic(expected = "operation left columns undefined")]
fn every_owned_column_must_be_defined() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation(
            "Copy",
            &[cols.left, cols.right],
            [("input", vec![cols.input.into()])],
            |b| {
                b.define_linear(cols.left, cols.input);
                [cols.left.into()]
            },
        );
    });
}

#[test]
#[should_panic(expected = "operation defined a column it does not own")]
fn definitions_outside_the_owned_columns_are_rejected() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation(
            "Copy",
            &[cols.left],
            [("input", vec![cols.input.into()])],
            |b| {
                b.define_linear(cols.left, cols.input);
                b.define_linear(cols.right, cols.input);
                [cols.left.into()]
            },
        );
    });
}

#[test]
#[should_panic(expected = "operation reads an undeclared argument")]
fn canceled_reads_still_require_an_argument() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation("Zero", &[cols.left], [], |b| {
            let zero = b.xor2(cols.input, cols.input);
            b.define_linear(cols.left, zero);
            [cols.left.into()]
        });
    });
}

#[test]
#[should_panic(expected = "interaction reads an undeclared operation argument")]
fn interaction_reads_require_an_argument() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation("Send", &[cols.left], [], |b| {
            b.define_linear(cols.left, b.zero());
            b.interaction(
                "test",
                "bit",
                InteractionDirection::Send,
                [InteractionField::column_bits("value", [cols.input])],
                [b.one()],
                b.selector(b.one()),
                InteractionScope::new("Test", 0),
            );
            [cols.left.into()]
        });
    });
}

#[test]
#[should_panic(expected = "operation defined a column it does not own")]
fn parent_must_own_its_child_definitions() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation(
            "Parent",
            &[cols.left],
            [("input", vec![cols.input.into()])],
            |b| {
                eval(b, cols);
                [cols.left.into()]
            },
        );
    });
}
