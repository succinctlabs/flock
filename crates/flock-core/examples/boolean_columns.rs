// Shared by the core and prover examples. Run with:
// cargo run -p flock-core --example boolean_columns

use flock_core::circuit::boolean::{
    CircuitBuilder, ColumnRole, ColumnSchema, ColumnVisitor, CompiledColumns, EvaluationError,
    InteractionDirection, InteractionField, InteractionScope, LinearExpr as Expr, Var,
};

struct Inputs<T> {
    a: [T; 4],
    b: [T; 4],
    c: [T; 4],
    active: T,
}

struct Add4<T> {
    carry_product: [T; 3],
}

struct Witness<T> {
    first: Add4<T>,
    second: Add4<T>,
    inactive: T,
}

struct Cols<T> {
    inputs: Inputs<T>,
    claimed_sum: [T; 4],
    witness: Witness<T>,
    output: [T; 4],
}

struct DoubleAdd;

impl ColumnSchema for DoubleAdd {
    type Cols<T> = Cols<T>;

    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Cols<V::Value> {
        Cols {
            inputs: Inputs {
                a: v.word("inputs.a", ColumnRole::Input),
                b: v.word("inputs.b", ColumnRole::Input),
                c: v.word("inputs.c", ColumnRole::Input),
                active: v.bit("inputs.active", ColumnRole::Input),
            },
            claimed_sum: v.word("claimed_sum", ColumnRole::Advice("wrapping-sum-4")),
            witness: Witness {
                first: Add4 {
                    carry_product: v.word("witness.first.carry_product", ColumnRole::Witness),
                },
                second: Add4 {
                    carry_product: v.word("witness.second.carry_product", ColumnRole::Witness),
                },
                inactive: v.bit("witness.inactive", ColumnRole::Witness),
            },
            output: v.word("output", ColumnRole::Output),
        }
    }
}

impl Add4<Var> {
    fn eval(b: &mut CircuitBuilder, a: [Expr; 4], rhs: [Expr; 4], cols: &Self) -> [Expr; 4] {
        b.operation(
            "Add4",
            &cols.carry_product,
            [("a", a.to_vec()), ("b", rhs.to_vec())],
            |b| {
                let mut carry = b.zero();
                std::array::from_fn(|i| {
                    let sum = b.xor3(a[i], rhs[i], carry);
                    if i < 3 {
                        let lhs = b.xor2(a[i], carry);
                        let rhs = b.xor2(rhs[i], carry);
                        b.define_and(cols.carry_product[i], lhs, rhs);
                        carry = b.xor2(carry, cols.carry_product[i]);
                    }
                    sum
                })
            },
        )
    }
}

fn msb<T: Copy>(word: &[T; 4]) -> T {
    word[3]
}

impl DoubleAdd {
    fn eval(b: &mut CircuitBuilder, cols: &Cols<Var>) {
        let first = Add4::eval(
            b,
            cols.inputs.a.map(Into::into),
            cols.inputs.b.map(Into::into),
            &cols.witness.first,
        );
        let sum = Add4::eval(
            b,
            first,
            cols.inputs.c.map(Into::into),
            &cols.witness.second,
        );
        for (output, expression) in cols.output.into_iter().zip(sum) {
            b.define_linear(output, expression);
        }
        let inactive = b.xor2(b.one(), cols.inputs.active);
        b.define_linear(cols.witness.inactive, inactive);

        let active = b.column_selector(cols.inputs.active);
        for i in 0..3 {
            b.assert_when_eq(active, cols.claimed_sum[i], cols.output[i]);
        }
        b.assert_when_eq(active, msb(&cols.claimed_sum), msb(&cols.output));
        b.interaction(
            "example-output",
            "sum",
            InteractionDirection::Send,
            [InteractionField::columns("value", 4, cols.output)],
            [b.one()],
            active,
            InteractionScope::new("DoubleAdd", 0),
        );
    }
}

#[derive(Clone, Copy)]
struct Event {
    a: u8,
    b: u8,
    c: u8,
}

type TraceRow = (Cols<bool>, Vec<bool>);

fn generate_trace(
    compiled: &CompiledColumns<DoubleAdd>,
    events: &[Option<Event>],
) -> Result<Vec<TraceRow>, EvaluationError> {
    events
        .iter()
        .map(|event| {
            compiled.evaluate(|cols| {
                if let Some(event) = event {
                    write_word(cols.inputs.a, event.a);
                    write_word(cols.inputs.b, event.b);
                    write_word(cols.inputs.c, event.c);
                    *cols.inputs.active = true;
                    write_word(
                        cols.claimed_sum,
                        event.a.wrapping_add(event.b).wrapping_add(event.c),
                    );
                }
                // An absent event supplies zero inputs/advice; definitions still run.
            })
        })
        .collect()
}

fn write_word(word: [&mut bool; 4], value: u8) {
    for (i, bit) in word.into_iter().enumerate() {
        *bit = value >> i & 1 != 0;
    }
}

fn read_word(word: [bool; 4]) -> u8 {
    word.into_iter()
        .enumerate()
        .fold(0, |value, (i, bit)| value | (u8::from(bit) << i))
}

fn main() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let trace = generate_trace(&compiled, &[Some(Event { a: 7, b: 9, c: 3 }), None]).unwrap();
    assert_eq!(read_word(trace[0].0.output), 3);
    assert!(trace[1].0.witness.inactive);
    println!("7 + 9 + 3 mod 16 = {}", read_word(trace[0].0.output));
}

#[cfg(test)]
#[path = "boolean_columns/tests.rs"]
mod tests;
