use super::*;

struct Repeated {
    rows: usize,
}

struct Cols<T> {
    output: T,
    input: T,
}

impl ColumnSchema for Repeated {
    type Cols<T> = Vec<Cols<T>>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        (0..self.rows)
            .map(|row| Cols {
                output: v.bit(&format!("row-{row}.output"), ColumnRole::Output),
                input: v.bit(&format!("row-{row}.input"), ColumnRole::Input),
            })
            .collect()
    }
}

fn eval(b: &mut CircuitBuilder, rows: &[Cols<Var>]) {
    for cols in rows {
        let complement = b.xor2(b.one(), cols.input);
        b.define_linear(cols.output, complement);
    }
}

#[test]
fn fixed_configuration_drives_reservation_resolution_and_supply() {
    for count in [0, 1, 3, 8] {
        let compiled = CircuitBuilder::compile(Repeated { rows: count }, |b, rows| eval(b, rows));
        let inputs = compiled.inputs(|rows| {
            for (i, cols) in rows.into_iter().enumerate() {
                *cols.input = i % 2 == 0;
            }
        });
        assert_eq!(inputs, (0..count).map(|i| i % 2 == 0).collect::<Vec<_>>());
        let (rows, values) = compiled
            .evaluate(|rows| {
                for (i, cols) in rows.into_iter().enumerate() {
                    *cols.input = i % 2 == 0;
                    // A write to a derived field is not supplied as an input.
                    *cols.output = i % 2 == 0;
                }
            })
            .unwrap();
        assert_eq!(values, compiled.circuit().evaluate(&inputs).unwrap());
        assert_eq!(rows.len(), count);
        assert_eq!(compiled.columns().len(), count);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.input, i % 2 == 0);
            assert_eq!(row.output, i % 2 != 0);
        }
        let (inactive, _) = compiled.evaluate(|_| {}).unwrap();
        assert!(inactive.iter().all(|row| row.output));
    }
}

#[test]
#[should_panic(expected = "schema traversal changed")]
fn finalization_rejects_changed_configuration() {
    let mut b = CircuitBuilder::new();
    let cols = b.reserve_columns(&Repeated { rows: 2 });
    eval(&mut b, &cols);
    b.finish_columns(Repeated { rows: 1 });
}
