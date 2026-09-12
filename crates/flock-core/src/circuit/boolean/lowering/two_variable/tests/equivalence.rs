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
        // A definition after an assertion exercises distinct execution/placement order.
        b.define_linear(cols.witness[1], product);
    })
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
        let extra_support: usize = source
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Constraint)
            .map(|row| 4 + source.support(row.result()).unwrap().len())
            .sum();
        assert_eq!(
            lowered.normalized_support_terms(),
            source.normalized_support_terms() + extra_support
        );
        assert_eq!(
            lowered.normalized_support_bytes(),
            source.normalized_support_bytes() + extra_support * std::mem::size_of::<ValueIndex>()
        );
        let k_log = lowered
            .layout()
            .useful_bits()
            .next_power_of_two()
            .trailing_zeros() as usize;
        let capacity = 1 << k_log;
        let original = source
            .to_block_r1cs_with_layout(k_log, 0, 0, &placement)
            .unwrap();
        let direct = source.lower(LoweringMode::Direct).unwrap();
        assert_eq!(
            direct.normalized_support_bytes(),
            source.normalized_support_bytes()
        );
        assert_eq!(
            direct
                .to_block_r1cs(k_log, 0, 0)
                .unwrap()
                .statement_digest(),
            original.statement_digest()
        );
        let converted = lowered.to_block_r1cs(k_log, 0, 0).unwrap();
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
            let expected = source_accepts(&z);
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
                assert!(source_accepts(&lowered.project(&z).unwrap()));
            }
        }
        for n in 0..1 << source.inputs().len() {
            let input = bits(n, source.inputs().len());
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

#[test]
fn extension_does_not_repair_derived_values_or_cancel_multiple_failures() {
    let source = fixture(6);
    let lowered = source.lower_identity_c().unwrap();
    let mut z = source.evaluate(&[false, false, false]).unwrap();
    let product = source
        .rows()
        .iter()
        .find(|row| row.kind() == RowKind::And)
        .unwrap()
        .defined_value()
        .unwrap();
    z[product.index()] = true;
    let extension = lowered.extend(&z).unwrap();
    assert_eq!(lowered.project(&extension).unwrap(), z);
    assert!(!lowered.accepts(&extension));

    // Both source assertions fail; one cannot cancel out the other.
    let z = vec![true, false, false, true, false, false];
    let extension = lowered.extend(&z).unwrap();
    let matrix = lowered.to_block_r1cs(4, 0, 0).unwrap();
    let witness = physical(&lowered, &extension, 16);
    let (a, b, c) = (
        matrix.apply_a(&witness),
        matrix.apply_b(&witness),
        matrix.apply_c(&witness),
    );
    for aux in lowered.auxiliaries() {
        let row = lowered.layout().row_position(aux.cancellation_row).unwrap();
        assert_ne!(a[row] & b[row], c[row]);
    }
    assert!(!lowered.accepts(&extension));
}

#[test]
fn retained_xor_dag_and_assertion_provenance_are_inspectable() {
    let source = fixture(4);
    let lowered = source.lower_identity_c().unwrap();
    for row in source.rows() {
        let mapped: Vec<_> = lowered.mapped_rows(row.id()).unwrap().collect();
        assert_eq!(
            mapped.len(),
            if row.kind() == RowKind::Constraint {
                2
            } else {
                1
            }
        );
        for &id in &mapped {
            assert_eq!(lowered.source_row(id), Some(row.id()));
        }
        for expr in [row.lhs(), row.rhs(), row.result()] {
            let mapped = lowered.mapped_expression(expr).unwrap();
            assert!(lowered.expression(expr).is_none());
            match source.expression(expr).unwrap() {
                ExpressionNode::Zero => {
                    assert_eq!(lowered.expression(mapped), Some(&ExpressionNode::Zero))
                }
                ExpressionNode::Value(value) => assert_eq!(
                    lowered.expression(mapped),
                    Some(&ExpressionNode::Value(
                        lowered.mapped_value(*value).unwrap()
                    ))
                ),
                ExpressionNode::Xor(terms) => assert_eq!(
                    lowered.expression(mapped),
                    Some(&ExpressionNode::Xor(
                        terms
                            .iter()
                            .map(|&term| lowered.mapped_expression(term).unwrap())
                            .collect()
                    ))
                ),
            }
        }
    }
    for aux in lowered.auxiliaries() {
        assert_eq!(
            lowered.rows()[aux.product_row.index()].defined_value(),
            Some(aux.product)
        );
        assert_eq!(
            lowered.rows()[aux.cancellation_row.index()].defined_value(),
            None
        );
    }
}
