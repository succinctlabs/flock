use crate::circuit::boolean::{CircuitBuilder, ColumnRole, ColumnSchema, ColumnVisitor, Var};

struct Nested {
    direct_field: bool,
}

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
            left: v.bit("parent.left.out", ColumnRole::Witness),
            right: v.bit("parent.right.out", ColumnRole::Witness),
            direct: self
                .direct_field
                .then(|| v.bit("parent.out", ColumnRole::Witness)),
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
    assert_eq!(ops[0].name, "parent.right");
    assert_eq!(ops[1].name, "parent.left");
    assert_eq!(ops[2].name, "parent");
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
#[should_panic(expected = "operation must own its exact schema instance")]
fn parent_cannot_omit_its_direct_field() {
    CircuitBuilder::compile(Nested { direct_field: true }, |b, cols| {
        b.operation("Incomplete", &[cols.left, cols.right], [], |_| {
            [cols.input.into()]
        });
    });
}

#[test]
#[should_panic(expected = "operation columns already defined")]
fn nested_parent_cannot_be_reused() {
    CircuitBuilder::compile(Nested { direct_field: true }, |b, cols| {
        eval(b, cols);
        eval(b, cols);
    });
}
