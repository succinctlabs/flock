use crate::circuit::boolean::{CircuitBuilder, ColumnRole, ColumnSchema, ColumnVisitor, Var};

struct Nested {
    direct_field: bool,
}

const PAIR: Nested = Nested {
    direct_field: false,
};

struct Cols<T> {
    input: T,
    children: [T; 2],
    direct: Option<T>,
}

impl ColumnSchema for Nested {
    type Cols<T> = Cols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Cols<V::Value> {
        Cols {
            input: v.bit("input", ColumnRole::Input),
            children: v.word("children", ColumnRole::Witness),
            direct: self
                .direct_field
                .then(|| v.bit("direct", ColumnRole::Witness)),
        }
    }
}

fn eval(b: &mut CircuitBuilder, cols: &Cols<Var>) {
    let owned: Vec<_> = cols.children.into_iter().chain(cols.direct).collect();
    b.operation(
        "Parent",
        &owned,
        [("input", vec![cols.input.into()])],
        |b| {
            let mut value = cols.input.into();
            // Same kind and witness field, but reverse definition order.
            for &out in cols.children.iter().rev() {
                value = b.operation("Not", &[out], [("input", vec![value])], |b| {
                    let complement = b.xor2(value, b.one());
                    b.define_linear(out, complement);
                    [out.into()]
                })[0];
            }
            if let Some(out) = cols.direct {
                b.define_linear(out, value);
            }
            [value]
        },
    );
}

#[test]
fn nested_and_repeated_operations_keep_ownership_and_definition_order() {
    for direct_field in [false, true] {
        let compiled = CircuitBuilder::compile(Nested { direct_field }, eval);
        let cols = compiled.columns();
        let ops = compiled.operations();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].kind, "Not");
        assert_eq!(ops[1].kind, "Not");
        assert_eq!(ops[2].kind, "Parent");
        assert_eq!(ops[0].columns, [cols.children[1]]);
        assert_eq!(ops[1].columns, [cols.children[0]]);
        assert_eq!(
            ops[2].columns,
            cols.children
                .into_iter()
                .chain(cols.direct)
                .collect::<Vec<_>>()
        );
        // ONE and input rows precede the child definitions.
        assert_eq!(ops[0].rows, 2..3);
        assert_eq!(ops[1].rows, 3..4);
        assert_eq!(ops[2].rows, 2..4 + usize::from(direct_field));
        assert!(cols.children[1].index() < cols.children[0].index());
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
            assert_eq!(trace.children[0], input);
            assert_eq!(trace.children[1], !input);
            assert_eq!(trace.direct, direct_field.then_some(input));
            witness.resize(8, false);
            assert!(relation.satisfies(&witness));
        }
    }
}

#[test]
#[should_panic(expected = "operation defined a column it does not own")]
fn definitions_outside_the_owned_columns_are_rejected() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation(
            "Copy",
            &[cols.children[0]],
            [("input", vec![cols.input.into()])],
            |b| {
                b.define_linear(cols.children[0], cols.input);
                b.define_linear(cols.children[1], cols.input);
                [cols.children[0].into()]
            },
        );
    });
}

#[test]
#[should_panic(expected = "operation reads an undeclared argument")]
fn canceled_reads_still_require_an_argument() {
    CircuitBuilder::compile(PAIR, |b, cols| {
        b.operation("Zero", &[cols.children[0]], [], |b| {
            let zero = b.xor2(cols.input, cols.input);
            b.define_linear(cols.children[0], zero);
            [cols.children[0].into()]
        });
    });
}
