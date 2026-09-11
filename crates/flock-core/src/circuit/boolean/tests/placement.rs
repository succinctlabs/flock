use super::*;

#[test]
fn named_ports_and_layout_stay_out_of_the_arithmetic_definition() {
    // This is the intended authoring style: named typed boundaries and
    // direct bit/expression operands, with no logical or physical IDs.
    let mut builder = CircuitBuilder::new();
    let a = builder.input_bits::<2>("a");
    let b = builder.input_bits::<2>("b");
    let sum_0_expr = builder.xor2(a[0], b[0]);
    let sum_1_expr = builder.xor2(a[1], b[1]);
    let sum = [
        builder.materialize(sum_0_expr),
        builder.materialize(sum_1_expr),
    ];
    builder.output("sum", sum);
    let circuit = builder.finish();

    assert_eq!(circuit.port("a").unwrap().direction(), PortDirection::Input);
    assert_eq!(
        circuit.port("sum").unwrap().direction(),
        PortDirection::Output
    );

    // Compatibility placement is a separate concern. Reserving the first
    // four positions and moving ports does not change the circuit code.
    let mut layout = circuit.layout();
    layout.reserve(0..4).unwrap();
    layout.place_port("a", 8).unwrap();
    layout.place_port("sum", 12).unwrap();
    let layout = layout.finish().unwrap();
    assert_eq!(layout.value_position(a[0].value_id()), Some(8));
    assert_eq!(layout.value_position(a[1].value_id()), Some(9));
    assert_eq!(layout.value_position(sum[0].value_id()), Some(12));
    assert_eq!(layout.value_position(sum[1].value_id()), Some(13));
    assert_eq!(layout.useful_bits(), 14);

    let r1cs = circuit.to_block_r1cs_with_layout(4, 0, 0, &layout).unwrap();
    let witness = circuit
        .evaluate_r1cs_with_layout(&[true, false, false, true], 4, &layout)
        .unwrap();
    assert!(r1cs.c0_is_identity());
    assert!(r1cs.satisfies(&witness));
    assert!(witness[12]);
    assert!(witness[13]);
    assert_eq!(r1cs.const_pin, layout.value_position(circuit.one()));
    assert_eq!(r1cs.c_0.rows[14], vec![14]);
    assert_eq!(r1cs.c_0.rows[15], vec![15]);
}

#[test]
fn word_ports_enforce_contiguous_aligned_layout() {
    let mut builder = CircuitBuilder::new();
    let word = builder.input_word_aligned::<8>("word", 8);
    let bits = builder.input_bits::<8>("bits");
    let circuit = builder.finish();

    assert_eq!(
        circuit.port("word").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 8 }
    );
    assert_eq!(circuit.port("bits").unwrap().encoding(), PortEncoding::Bits);
    assert!(matches!(
        circuit.to_block_r1cs(5, 0, 0),
        Err(R1csBuildError::InvalidLayout(
            LayoutError::MisalignedPort {
                ref name,
                start: 1,
                alignment_bits: 8,
            }
        )) if name == "word"
    ));
    assert!(matches!(
        circuit.walk_plan(),
        Err(LayoutError::MisalignedPort {
            ref name,
            start: 1,
            alignment_bits: 8,
        }) if name == "word"
    ));

    let mut layout = circuit.layout();
    assert!(matches!(
        layout.place_port("word", 4),
        Err(LayoutError::MisalignedPort {
            ref name,
            start: 4,
            alignment_bits: 8,
        }) if name == "word"
    ));
    layout.place_port("word", 8).unwrap();
    layout.place_port("bits", 24).unwrap();
    let layout = layout.finish().unwrap();
    for (offset, bit) in word.iter().enumerate() {
        assert_eq!(layout.value_position(bit.value_id()), Some(8 + offset));
    }
    assert_eq!(layout.value_position(bits[0].value_id()), Some(24));

    let mut scattered = circuit.layout();
    for (offset, bit) in word.iter().enumerate() {
        scattered
            .place_definition(bit.value_id(), 8 + 2 * offset)
            .unwrap();
    }
    assert!(matches!(
        scattered.finish(),
        Err(LayoutError::NonContiguousPort {
            ref name,
            offset: 1,
            expected: 9,
            actual: 10,
        }) if name == "word"
    ));
}

#[test]
fn layout_rejects_overlaps_and_cross_circuit_reuse() {
    let mut first = CircuitBuilder::new();
    let bits = first.input_bits::<2>("bits");
    let first = first.finish();
    let mut layout = first.layout();
    layout.place_definition(bits[0].value_id(), 7).unwrap();
    assert!(matches!(
        layout.place_definition(bits[1].value_id(), 7),
        Err(LayoutError::PositionOccupied {
            kind: PositionKind::Column,
            position: 7,
            ..
        })
    ));

    let first_layout = layout.finish().unwrap();
    let mut second_builder = CircuitBuilder::new();
    let second_bit = second_builder.input();
    let second = second_builder.finish();
    let mut second_layout = second.layout();
    assert!(matches!(
        second_layout.place_value(bits[0].value_id(), 0),
        Err(LayoutError::WrongCircuit)
    ));
    let foreign_row = first.definition_row(bits[0].value_id()).unwrap();
    assert!(matches!(
        second_layout.place_row(foreign_row, 0),
        Err(LayoutError::WrongCircuit)
    ));
    assert_eq!(
        PhysicalLayout::source_order(&second).value_position(bits[0].value_id()),
        None
    );
    assert_eq!(second.expression(bits[0].expr().id()), None);
    second_layout
        .place_definition(second_bit.value_id(), 0)
        .unwrap();
    assert!(matches!(
        second.to_block_r1cs_with_layout(3, 0, 0, &first_layout),
        Err(R1csBuildError::InvalidLayout(LayoutError::WrongCircuit))
    ));
}

#[test]
fn sha_sized_layouts_are_deterministic() {
    const N: usize = 25_500;
    fn build() -> (BooleanCircuit, Vec<Bit>) {
        let mut builder = CircuitBuilder::new();
        let bits = (0..N).map(|_| builder.input()).collect();
        (builder.finish(), bits)
    }

    let (first, first_bits) = build();
    let automatic = first.layout().finish().unwrap();
    assert_eq!(automatic.useful_bits(), N + 1);
    let mut first_layout = first.layout();
    for (i, bit) in first_bits.into_iter().enumerate() {
        first_layout
            .place_definition(bit.value_id(), N - 1 - i)
            .unwrap();
    }
    let first_layout = first_layout.finish().unwrap();

    let (second, second_bits) = build();
    let mut second_layout = second.layout();
    for (i, bit) in second_bits.into_iter().enumerate() {
        second_layout
            .place_definition(bit.value_id(), N - 1 - i)
            .unwrap();
    }
    let second_layout = second_layout.finish().unwrap();
    assert_eq!(first_layout, second_layout);
    assert_eq!(first_layout.useful_bits(), N + 1);
}

#[test]
fn rejects_unrepresentable_total_dimension() {
    let circuit = CircuitBuilder::new().finish();
    assert!(matches!(
        circuit.to_block_r1cs(0, 0, usize::BITS as usize),
        Err(R1csBuildError::DimensionOverflow)
    ));
}

#[test]
fn capacity_boundaries_are_exact_across_backends() {
    use crate::field::F128;

    let mut builder = CircuitBuilder::new();
    for _ in 0..7 {
        builder.input();
    }
    let circuit = builder.finish();
    let inputs = [false; 7];
    let plan = circuit.walk_plan().unwrap();

    assert!(circuit.evaluate_r1cs(&inputs, 3).is_ok());
    assert!(circuit.to_block_r1cs(3, 0, 0).is_ok());
    assert!(plan.forward(&inputs, 3).is_ok());
    let weights = vec![F128::ZERO; 8];
    assert!(plan.transpose(&weights, &weights, &weights).is_ok());

    assert!(matches!(
        circuit.evaluate_r1cs(&inputs, 2),
        Err(EvaluationError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    assert!(matches!(
        circuit.to_block_r1cs(2, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    assert!(matches!(
        plan.forward(&inputs, 2),
        Err(WalkError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
    let too_small = vec![F128::ZERO; 4];
    assert!(matches!(
        plan.transpose(&too_small, &too_small, &too_small),
        Err(WalkError::Capacity {
            required: 8,
            actual: 4,
        })
    ));
}
