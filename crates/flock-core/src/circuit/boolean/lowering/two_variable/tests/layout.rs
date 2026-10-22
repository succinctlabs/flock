use super::*;
use crate::circuit::boolean::ColumnRole;
use crate::circuit::boolean::tests::support::Fields;

#[test]
fn identity_lowering_keeps_alignment_and_checks_all_padding() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("input", ColumnRole::Input, 2, 4),
            ("output", ColumnRole::Output, 1, 1),
        ]),
        |b, cols| {
            b.constrain(cols[0][0], cols[0][1], b.zero());
            b.define_linear(cols[1][0], cols[0][0]);
        },
    );
    let source = compiled.circuit();
    let placement = source.layout().unwrap();
    let lowered = source.lower_identity_c().unwrap();
    assert_eq!(
        &lowered.layout().value_positions()[..source.value_count()],
        placement.value_positions()
    );
    let relation = lowered.to_block_r1cs(4, 0, 0).unwrap();
    assert!(relation.c0_is_identity());
    assert_eq!(relation.const_pin, Some(0));
    let logical = lowered.evaluate(&[true, false]).unwrap();
    let witness = physical(&lowered, &logical, 16);
    assert!(relation.satisfies(&witness));
    let plan = lowered.walk_plan().unwrap();
    assert_eq!(plan.forward(&[true, false], 4).unwrap().z, witness);
    for position in 0..16 {
        if !lowered.layout().value_positions().contains(&position) {
            let mut invalid = witness.clone();
            invalid[position] = true;
            assert!(!relation.satisfies(&invalid), "padding {position}");
        }
    }
}

#[test]
fn capacity_and_dimension_errors_remain_explicit() {
    let lowered = super::equivalence::fixture(1).lower_identity_c().unwrap();
    assert!(matches!(
        lowered.to_block_r1cs(2, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 9,
            actual: 4
        })
    ));
    assert!(lowered.to_block_r1cs(4, 0, 0).unwrap().c0_is_identity());
    assert!(matches!(
        lowered.to_block_r1cs(3, 4, 0),
        Err(R1csBuildError::InvalidKSkip { .. })
    ));
    assert!(matches!(
        lowered.to_block_r1cs(4, 0, usize::MAX),
        Err(R1csBuildError::DimensionOverflow)
    ));
    assert!(matches!(
        lowered.to_block_r1cs(usize::MAX, 0, 0),
        Err(R1csBuildError::InvalidKLog(_))
    ));
}
