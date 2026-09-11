use super::*;

#[test]
fn identity_placement_preserves_aligned_ports_and_relocated_one() {
    let mut b = CircuitBuilder::new();
    let input = b.input_word_aligned::<2>("input", 4);
    let assertion = b.constrain(input[0], input[1], b.zero());
    let output = b.materialize(input[0]);
    b.output("output", [output]);
    let source = b.finish();
    let mut placement = source.layout();
    placement.place_definition(source.one(), 3).unwrap();
    placement.place_port("input", 4).unwrap();
    placement.place_definition(output.value_id(), 0).unwrap();
    placement.place_row(assertion, 7).unwrap();
    let placement = placement.finish().unwrap();
    let lowered = source.lower_identity_c(&placement).unwrap();
    assert_eq!(lowered.layout().useful_bits(), placement.useful_bits() + 2);
    for (index, &position) in placement.value_positions().iter().enumerate() {
        assert_eq!(lowered.layout().value_positions()[index], position);
    }
    let aux = &lowered.auxiliaries()[0];
    assert_eq!(lowered.layout().value_position(aux.product), Some(8));
    assert_eq!(lowered.layout().value_position(aux.cancellation), Some(9));
    let relation = lowered.to_block_r1cs(4, 0, 0).unwrap();
    assert!(relation.c0_is_identity());
    assert_eq!(relation.const_pin, Some(3));
    let logical = lowered.evaluate(&[true, false]).unwrap();
    let witness = physical(&lowered, &logical, 16);
    assert!(relation.satisfies(&witness));
    for position in [1, 2, 6, 7, 10, 15] {
        let mut invalid = witness.clone();
        invalid[position] = true;
        assert!(
            !relation.satisfies(&invalid),
            "hole/padding {position} must be constrained"
        );
    }
    assert!(matches!(
        lowered.to_block_r1cs(3, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 10,
            actual: 8
        })
    ));
    assert!(source.lower_identity_c(lowered.layout()).is_err());
}

#[test]
fn existing_identity_layout_needs_no_auxiliaries_or_relation_changes() {
    let mut b = CircuitBuilder::new();
    let [x, y] = b.input_bits("input");
    b.and(x, y);
    let source = b.finish();
    let mut placement = source.layout();
    for (i, row) in source.rows().iter().enumerate() {
        placement
            .place_definition(row.defined_value().unwrap(), 7 - i)
            .unwrap();
    }
    let placement = placement.finish().unwrap();
    let lowered = source.lower_identity_c(&placement).unwrap();
    assert!(lowered.auxiliaries().is_empty());
    assert_eq!(lowered.layout(), &placement); // Physical data, not runtime circuit IDs.
    let original = source
        .to_block_r1cs_with_layout(3, 0, 0, &placement)
        .unwrap();
    let converted = lowered.to_block_r1cs(3, 0, 0).unwrap();
    assert_eq!(original.a_0.rows, converted.a_0.rows);
    assert_eq!(original.b_0.rows, converted.b_0.rows);
    assert_eq!(original.c_0.rows, converted.c_0.rows);
    assert_eq!(original.statement_digest(), converted.statement_digest());
    for n in 0..4 {
        assert_eq!(lowered.evaluate(&bits(n, 2)), source.evaluate(&bits(n, 2)));
    }
}
