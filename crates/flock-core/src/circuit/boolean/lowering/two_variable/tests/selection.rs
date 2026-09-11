use super::*;

fn fixture(assertion: bool) -> BooleanCircuit {
    let mut b = CircuitBuilder::new();
    let [x, y] = b.input_bits("input");
    if assertion {
        b.constrain(x, y, x);
    }
    b.and(x, y);
    b.finish()
}

#[test]
fn explicit_modes_preserve_their_relation_and_witness_contracts() {
    let source = fixture(true);
    let placement = PhysicalLayout::source_order(&source);
    let original = source
        .to_block_r1cs_with_layout(3, 0, 0, &placement)
        .unwrap();
    for policy in [RowPlacement::Preserve, RowPlacement::AllowReordering] {
        let direct = source
            .lower(LoweringMode::Direct, &placement, policy)
            .unwrap();
        assert!(!direct.c_is_identity());
        assert!(direct.auxiliaries().is_empty());
        assert_eq!(direct.layout(), &placement);
        let matrix = direct.to_block_r1cs(3, 0, 0).unwrap();
        assert_eq!(matrix.a_0.rows, original.a_0.rows);
        assert_eq!(matrix.b_0.rows, original.b_0.rows);
        assert_eq!(matrix.c_0.rows, original.c_0.rows);
        assert_eq!(matrix.statement_digest(), original.statement_digest());
        for n in 0..1 << source.value_count() {
            let z = bits(n, source.value_count());
            assert_eq!(direct.extend(&z), Some(z.clone()));
            let mut padded = z.clone();
            padded.resize(8, false);
            assert_eq!(direct.accepts(&z), z[0] && original.satisfies(&padded));
        }
        for n in 0..4 {
            assert_eq!(direct.evaluate(&bits(n, 2)), source.evaluate(&bits(n, 2)));
        }
    }
    assert!(matches!(
        source.lower(
            LoweringMode::RequireIdentityC,
            &placement,
            RowPlacement::Preserve
        ),
        Err(LayoutError::IdentityCRequired)
    ));
    let lowered = source
        .lower(
            LoweringMode::RequireIdentityC,
            &placement,
            RowPlacement::AllowReordering,
        )
        .unwrap();
    assert!(lowered.c_is_identity());
    assert_eq!(lowered.auxiliaries().len(), 1);
    assert!(lowered.to_block_r1cs(3, 0, 0).unwrap().c0_is_identity());
    let failed = source
        .rows()
        .iter()
        .find(|row| row.kind() == RowKind::Constraint)
        .unwrap()
        .id();
    assert_eq!(
        lowered.evaluate(&[true, false]),
        Err(crate::circuit::boolean::EvaluationError::UnsatisfiedRow(
            failed
        ))
    );
}

#[test]
fn definition_only_reordering_requires_explicit_permission() {
    let source = fixture(false);
    // All 24 value permutations, with rows independently placed beyond them.
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let positions = [a, b, c, d];
                    if (0..4).any(|i| positions[..i].contains(&positions[i])) {
                        continue;
                    }
                    let mut layout = source.layout();
                    for (i, row) in source.rows().iter().enumerate() {
                        layout
                            .place_value(row.defined_value().unwrap(), positions[i])
                            .unwrap();
                        layout.place_row(row.id(), i + 4).unwrap();
                    }
                    let layout = layout.finish().unwrap();
                    assert!(matches!(
                        source.lower(
                            LoweringMode::RequireIdentityC,
                            &layout,
                            RowPlacement::Preserve
                        ),
                        Err(LayoutError::IdentityCRequired)
                    ));
                    let lowered = source
                        .lower(
                            LoweringMode::RequireIdentityC,
                            &layout,
                            RowPlacement::AllowReordering,
                        )
                        .unwrap();
                    assert!(lowered.c_is_identity());
                    assert!(lowered.auxiliaries().is_empty());
                    assert_eq!(lowered.layout().value_positions(), &positions);
                    assert_eq!(lowered.layout().row_positions(), &positions);
                    let matrix = lowered.to_block_r1cs(3, 0, 0).unwrap();
                    assert_eq!(matrix.const_pin, Some(a));
                    for input in 0..4 {
                        let z = lowered.evaluate(&bits(input, 2)).unwrap();
                        assert!(matrix.satisfies(&physical(&lowered, &z, 8)));
                    }
                }
            }
        }
    }
    let layout = source.layout().finish().unwrap();
    let kept = source
        .lower(
            LoweringMode::RequireIdentityC,
            &layout,
            RowPlacement::Preserve,
        )
        .unwrap();
    assert_eq!(kept.layout(), &layout);
    assert!(kept.auxiliaries().is_empty());
}

#[test]
fn capacity_growth_overflow_and_foreign_placements_fail_explicitly() {
    let source = fixture(true);
    let mut layout = source.layout();
    for (i, row) in source.rows().iter().enumerate() {
        if let Some(value) = row.defined_value() {
            layout.place_value(value, value.index()).unwrap();
        }
        layout
            .place_row(
                row.id(),
                if row.kind() == RowKind::Constraint {
                    7
                } else {
                    i
                },
            )
            .unwrap();
    }
    let layout = layout.finish().unwrap();
    let direct = source
        .lower(LoweringMode::Direct, &layout, RowPlacement::Preserve)
        .unwrap();
    assert!(direct.to_block_r1cs(3, 0, 0).is_ok());
    let lowered = source.lower_identity_c(&layout).unwrap();
    assert!(matches!(
        lowered.to_block_r1cs(3, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 10,
            actual: 8
        })
    ));
    assert!(lowered.to_block_r1cs(4, 0, 0).unwrap().c0_is_identity());
    assert!(matches!(
        lowered.to_block_r1cs(4, 5, 0),
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
    let mut overflowing = layout.clone();
    overflowing.useful_bits = usize::MAX;
    assert!(matches!(
        source.lower_identity_c(&overflowing),
        Err(LayoutError::PositionOverflow)
    ));
    let foreign = fixture(true).layout().finish().unwrap();
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        assert!(matches!(
            source.lower(mode, &foreign, RowPlacement::AllowReordering),
            Err(LayoutError::WrongCircuit)
        ));
    }
}
