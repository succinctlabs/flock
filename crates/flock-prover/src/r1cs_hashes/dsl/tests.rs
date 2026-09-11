use super::*;
use flock_core::circuit::boolean::ColumnSchema;

use flock_core::r1cs::BlockR1cs;
use std::time::{Duration, Instant};

pub(crate) fn median_time(iterations: usize, mut operation: impl FnMut()) -> Duration {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[iterations / 2]
}

#[track_caller]
pub(crate) fn assert_relation_equal(context: &str, actual: &BlockR1cs, expected: &BlockR1cs) {
    assert_eq!(actual.m, expected.m);
    assert_eq!(actual.k_log, expected.k_log);
    assert_eq!(actual.k_skip, expected.k_skip);
    assert_eq!(actual.useful_bits, expected.useful_bits);
    assert_eq!(actual.layout, expected.layout);
    assert_eq!(actual.const_pin, expected.const_pin);
    for (side, actual, expected) in [
        ("A", &actual.a_0.rows, &expected.a_0.rows),
        ("B", &actual.b_0.rows, &expected.b_0.rows),
        ("C", &actual.c_0.rows, &expected.c_0.rows),
    ] {
        assert_eq!(actual.len(), expected.len(), "{context}: {side} row count");
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual, expected, "{context}: {side} differs at row {row}");
        }
    }
    assert_eq!(actual.statement_digest(), expected.statement_digest());
}

struct Fixture(u32);
struct Cols<T> {
    input: [[T; 32]; 4],
    two: Add32<T>,
    three: Add3<T>,
    four: Add4<T>,
    constant: AddConst32<T>,
    output: [[T; 32]; 4],
}

impl ColumnSchema for Fixture {
    type Cols<T> = Cols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Cols<V::Value> {
        Cols {
            input: words(v, "input", ColumnRole::Input, 1),
            two: Add32::columns(v, "two"),
            three: Add3::columns(v, "three"),
            four: Add4::columns(v, "four"),
            constant: AddConst32::columns(v, "constant", self.0),
            output: words(v, "output", ColumnRole::Output, 1),
        }
    }
}

#[test]
fn shared_additions_match_wrapping_arithmetic_including_affine_constants() {
    let mut rng = flock_core::test_rng::Rng::new(0xadd3_0032);
    for k in [
        0,
        1,
        2,
        4,
        0x4000_0000,
        0x8000_0000,
        0xffff_ffff,
        0x428a_2f98,
    ] {
        let compiled = CircuitBuilder::compile(Fixture(k), |b, cols| {
            let [x, y, z, w] = cols.input.map(|word| word.map(Into::into));
            let sums = [
                cols.two.eval(b, x, y),
                cols.three.eval(b, x, y, z),
                cols.four.eval(b, x, y, z, w),
                cols.constant.eval(b, k, x),
            ];
            for (cols, sum) in cols.output.iter().zip(sums) {
                materialize(b, cols, sum);
            }
        });
        let bits = compiled.circuit().row_count().next_power_of_two();
        let sparse = compiled
            .circuit()
            .to_block_r1cs(bits.trailing_zeros() as usize, 0, 0)
            .unwrap();
        for i in 0..100 {
            let input = match i {
                0 => [0; 4],
                1 => [u32::MAX; 4],
                _ => std::array::from_fn(|_| rng.next_u32()),
            };
            let (cols, mut witness) = compiled
                .evaluate(|cols| populate_words(cols.input, &input))
                .unwrap();
            let [x, y, z, w] = input;
            let expected = [
                x.wrapping_add(y),
                x.wrapping_add(y).wrapping_add(z),
                x.wrapping_add(y).wrapping_add(z).wrapping_add(w),
                x.wrapping_add(k),
            ];
            let actual = cols.output.map(|word| {
                word.into_iter()
                    .enumerate()
                    .fold(0u32, |n, (i, bit)| n | (u32::from(bit) << i))
            });
            assert_eq!(actual, expected);
            witness.resize(bits, false);
            assert!(sparse.satisfies(&witness));
        }
    }
}
