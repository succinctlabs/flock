use super::*;
use crate::circuit::boolean::tests::support;

pub(super) fn fixture(case: usize) -> BooleanCircuit {
    support::circuit(3, 2, |b, cols| {
        let [x, y, s] = cols.input.as_slice() else {
            unreachable!()
        };
        let (x, y, s) = (*x, *y, *s);
        let product = cols.witness[0];
        b.define_and(product, x, y);
        let values = b.value_count();
        match case {
            0 => {} // Definition-only path.
            1 => {
                b.constrain(x, y, s);
            }
            2 => {
                b.assert_zero_product(x, y);
            }
            3 => {
                b.constrain(x, y, b.one());
            }
            4 => {
                let shared = b.xor2(x, s);
                let canceled = b.xor2(shared, shared);
                b.constrain(shared, y, canceled);
            }
            5 => {
                b.assert_when_eq(b.column_selector(x), product, s);
            }
            6 => {
                b.constrain(x, y, s);
                let result = b.xor2(x, s);
                b.constrain(product, b.one(), result);
            }
            _ => unreachable!(),
        }
        assert_eq!(b.value_count(), values);
        // A definition after an assertion exercises distinct execution/placement order.
        b.define_linear(cols.witness[1], product);
    })
}

// Independent formulas for ONE, inputs, product, copied product, and assertions.
fn accepts(case: usize, z: &[bool]) -> bool {
    let [one, x, y, s, product, output]: [bool; 6] = z.try_into().unwrap();
    one && product == (x & y)
        && output == product
        && match case {
            0 => true,
            1 => (x & y) == s,
            2 => !(x & y),
            3 => x & y,
            4 => !((x ^ s) & y),
            5 => !x || product == s,
            6 => (x & y) == s && product == (x ^ s),
            _ => unreachable!(),
        }
}

#[test]
fn exhaustive_source_and_lowered_witnesses_match_independent_sparse_relations() {
    for case in 0..7 {
        let source = fixture(case);
        let placement = source.layout().unwrap();
        let lowered = source.lower_identity_c().unwrap();
        let q = source
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Constraint)
            .count();
        assert_eq!(lowered.value_count(), source.value_count() + 2 * q);
        assert_eq!(lowered.rows().len(), lowered.value_count());
        assert_eq!(lowered.auxiliaries().len(), q);
        let k_log = lowered
            .layout()
            .useful_bits()
            .next_power_of_two()
            .trailing_zeros() as usize;
        let capacity = 1 << k_log;
        let original = source
            .to_block_r1cs_with_layout(k_log, 0, 0, &placement)
            .unwrap();
        assert_eq!(original.c0_is_identity(), q == 0);
        let direct = source.lower(LoweringMode::Direct).unwrap();
        let converted = lowered.to_block_r1cs(k_log, 0, 0).unwrap();
        assert!(direct.auxiliaries().is_empty());
        if q == 0 {
            assert_eq!(lowered.layout(), direct.layout());
            assert_eq!(converted.statement_digest(), original.statement_digest());
        }
        assert!(converted.c0_is_identity());
        assert_eq!(
            converted.const_pin,
            Some(lowered.layout().value_position(lowered.one()).unwrap())
        );
        let source_accepts = |logical: &[bool]| {
            let mut z = vec![false; capacity];
            for (&position, &bit) in placement.value_positions().iter().zip(logical) {
                z[position] = bit;
            }
            z[original.const_pin.unwrap()] && original.satisfies(&z)
        };
        for n in 0..1 << source.value_count() {
            let z = bits(n, source.value_count());
            let expected = accepts(case, &z);
            assert_eq!(source_accepts(&z), expected, "case {case}, source {n}");
            assert_eq!(direct.accepts(&z), expected);
            assert_eq!(direct.extend(&z), Some(z.clone()));
            let extension = lowered.extend(&z).unwrap();
            assert_eq!(lowered.project(&extension).unwrap(), z);
            for choices in 0..1 << q {
                let mut candidate = extension.clone();
                for (i, aux) in lowered.auxiliaries().iter().enumerate() {
                    candidate[aux.cancellation.index()] = choices >> i & 1 != 0;
                }
                assert_eq!(
                    lowered.accepts(&candidate),
                    expected,
                    "case {case}, source {n}, t {choices}"
                );
            }
        }
        // Enumerate y as well as t: forged products cannot satisfy the lowered relation.
        for n in 0..1 << lowered.value_count() {
            let z = bits(n, lowered.value_count());
            let physical = physical(&lowered, &z, capacity);
            let sparse_holds =
                physical[converted.const_pin.unwrap()] && converted.satisfies(&physical);
            assert_eq!(lowered.accepts(&z), sparse_holds);
            if sparse_holds {
                assert!(
                    accepts(case, &lowered.project(&z).unwrap()),
                    "case {case}, lowered {n}"
                );
            }
        }
        for n in 0..1 << source.inputs().len() {
            let input: [bool; 3] = std::array::from_fn(|i| n >> i & 1 != 0);
            let [x, y, s] = input;
            let expected = accepts(case, &[true, x, y, s, x & y, x & y]);
            assert_eq!(
                source.evaluate(&input).is_ok(),
                expected,
                "case {case}, input {n}"
            );
            assert_eq!(direct.evaluate(&input), source.evaluate(&input));
            match source.evaluate(&input) {
                Ok(z) => {
                    let actual = lowered.evaluate(&input).unwrap();
                    assert_eq!(actual, lowered.extend(&z).unwrap());
                    assert!(lowered.accepts(&actual));
                }
                Err(error) => assert_eq!(lowered.evaluate(&input).unwrap_err(), error),
            }
        }
        assert!(lowered.extend(&[]).is_none());
        assert!(lowered.project(&[]).is_none());
        assert!(!lowered.accepts(&[]));
        assert!(lowered.evaluate(&[]).is_err());
    }
}
