use super::*;
use flock_core::circuit::boolean::ColumnSchema;

use flock_core::r1cs::BlockR1cs;
pub(crate) fn read_word(bits: [bool; 32]) -> u32 {
    bits.into_iter()
        .enumerate()
        .fold(0, |word, (bit, value)| word | (u32::from(value) << bit))
}

pub(crate) fn check_hash_lowering(
    circuit: &flock_core::circuit::boolean::BooleanCircuit,
    layout: &flock_core::circuit::boolean::PhysicalLayout,
    matrix: &BlockR1cs,
) {
    for column in circuit
        .schema()
        .iter()
        .filter(|column| column.role != ColumnRole::Witness)
    {
        let start = layout.value_position(column.values[0]).unwrap();
        assert_eq!(start % column.alignment_bits, 0);
        for (offset, &value) in column.values.iter().enumerate() {
            assert_eq!(layout.value_position(value), Some(start + offset));
        }
    }
    let identity = circuit.lower_identity_c().unwrap();
    assert!(identity.c_is_identity());
    assert!(identity.auxiliaries().is_empty());
    assert_eq!(identity.layout(), layout);
    let converted = identity
        .to_block_r1cs(matrix.k_log, matrix.k_skip, 3)
        .unwrap();
    assert_eq!(converted.statement_digest(), matrix.statement_digest());
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
