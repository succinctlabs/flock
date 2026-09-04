use super::*;
use crate::circuit::boolean::{CircuitBuilder, PortOrigin};

#[test]
fn interface_metadata_is_typed_inspectable_and_allocation_free() {
    let mut builder = CircuitBuilder::new();
    let active = builder.input_selector("active");

    let advice = builder.component("load-byte.row-0", |builder| {
        let advice = builder.advice_word::<8>("selected-byte", "sp1.load-byte.selected-byte/v1");
        let values = builder.value_count();
        let rows = builder.row_count();
        let expressions = builder.expression_count();
        builder.interaction(
            "memory",
            "read",
            InteractionDirection::Receive,
            [InteractionField::little_endian("value", 8, advice)],
            [active.bit()],
            active,
            InteractionScope::new("load-byte", 0),
        );
        assert_eq!(builder.value_count(), values);
        assert_eq!(builder.row_count(), rows);
        assert_eq!(builder.expression_count(), expressions);
        advice
    });
    let circuit = builder.finish();

    assert_eq!(
        circuit.port("selected-byte").unwrap().origin(),
        &PortOrigin::Advice {
            advice_type: "sp1.load-byte.selected-byte/v1".to_owned(),
        }
    );
    let component = &circuit.components()[0];
    assert_eq!(component.name(), "load-byte.row-0");
    assert!(component.values().contains(&advice[0].value_id().index()));

    let interaction = &circuit.interactions()[0];
    assert_eq!(interaction.channel(), "memory");
    assert_eq!(interaction.kind(), "read");
    assert_eq!(interaction.direction(), InteractionDirection::Receive);
    assert_eq!(interaction.scope(), &InteractionScope::new("load-byte", 0));
    assert_eq!(interaction.component(), Some(0));
    assert_eq!(interaction.selector(), active.bit().value_id());
    assert_eq!(interaction.multiplicity(), &[active.bit().value_id()]);
    assert_eq!(interaction.message()[0].values().len(), 8);
}

#[test]
fn interaction_multiplicity_is_gated_by_its_selector() {
    let mut builder = CircuitBuilder::new();
    let active = builder.input_selector("active");
    let multiplicity = builder.input_bits::<2>("multiplicity");
    builder.interaction(
        "test",
        "unequal-activation",
        InteractionDirection::Send,
        [InteractionField::bits("value", [active.bit()])],
        multiplicity,
        active,
        InteractionScope::new("test", 0),
    );
    let circuit = builder.finish();
    let interaction = &circuit.interactions()[0];

    let selector_off = circuit.evaluate(&[false, true, true]).unwrap();
    assert_eq!(
        interaction.effective_multiplicity_bits(&selector_off),
        [false, false]
    );

    let selector_on = circuit.evaluate(&[true, true, false]).unwrap();
    assert_eq!(
        interaction.effective_multiplicity_bits(&selector_on),
        [true, false]
    );
}

#[test]
fn guarded_equality_obeys_the_selector() {
    let mut builder = CircuitBuilder::new();
    let active = builder.input_selector("active");
    let lhs = builder.input();
    let rhs = builder.input();
    builder.assert_when_eq(active, lhs, rhs);
    let circuit = builder.finish();
    let r1cs = circuit.to_block_r1cs(3, 0, 0).unwrap();

    assert!(r1cs.satisfies(&circuit.evaluate_r1cs(&[false, false, true], 3).unwrap()));
    assert!(r1cs.satisfies(&circuit.evaluate_r1cs(&[true, true, true], 3).unwrap()));
    assert!(matches!(
        circuit.evaluate_r1cs(&[true, false, true], 3),
        Err(crate::circuit::boolean::EvaluationError::UnsatisfiedRow(_))
    ));
}

#[test]
fn advice_origin_does_not_change_the_local_relation_identity() {
    fn build(advice: bool) -> crate::circuit::boolean::BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        if advice {
            builder.advice_word::<8>("value", "test/v1");
        } else {
            builder.input_word::<8>("value");
        }
        builder.finish()
    }

    let witness = build(false);
    let advice = build(true);
    assert_eq!(witness.structure_digest(), advice.structure_digest());
    assert_eq!(
        witness.to_block_r1cs(4, 0, 0).unwrap().statement_digest(),
        advice.to_block_r1cs(4, 0, 0).unwrap().statement_digest()
    );
}

#[test]
#[should_panic(expected = "advice type must not be empty")]
fn empty_advice_type_is_rejected() {
    CircuitBuilder::new().advice_bits::<1>("advice", "");
}

#[test]
#[should_panic(expected = "interaction message must not be empty")]
fn empty_interaction_message_is_rejected() {
    let mut builder = CircuitBuilder::new();
    let active = builder.input_selector("active");
    builder.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
    builder.finish();
}

#[test]
#[should_panic(expected = "interaction element width must be nonzero")]
fn malformed_interaction_encoding_is_rejected() {
    let mut builder = CircuitBuilder::new();
    let active = builder.input_selector("active");
    let value = builder.input();
    builder.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [InteractionField::little_endian("value", 0, [value])],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
    builder.finish();
}

#[test]
fn remaining_malformed_interaction_shapes_are_rejected() {
    #[derive(Clone, Copy, Debug)]
    enum Malformed {
        EmptyChannel,
        EmptyKind,
        EmptyScope,
        EmptyMultiplicity,
        EmptyFieldName,
        EmptyField,
        DuplicateField,
        NonDivisibleWidth,
    }

    let cases = [
        (
            Malformed::EmptyChannel,
            "interaction channel must not be empty",
        ),
        (Malformed::EmptyKind, "interaction kind must not be empty"),
        (
            Malformed::EmptyScope,
            "interaction chip scope must not be empty",
        ),
        (
            Malformed::EmptyMultiplicity,
            "interaction multiplicity must not be empty",
        ),
        (
            Malformed::EmptyFieldName,
            "interaction field name must not be empty",
        ),
        (Malformed::EmptyField, "interaction field is empty"),
        (Malformed::DuplicateField, "duplicate interaction field"),
        (
            Malformed::NonDivisibleWidth,
            "interaction field width is not a multiple",
        ),
    ];

    for (case, expected) in cases {
        let panic = std::panic::catch_unwind(|| {
            let mut builder = CircuitBuilder::new();
            let active = builder.input_selector("active");
            let value = builder.input();
            let channel = if matches!(case, Malformed::EmptyChannel) {
                ""
            } else {
                "memory"
            };
            let kind = if matches!(case, Malformed::EmptyKind) {
                ""
            } else {
                "read"
            };
            let scope = if matches!(case, Malformed::EmptyScope) {
                ""
            } else {
                "load-byte"
            };
            let multiplicity = if matches!(case, Malformed::EmptyMultiplicity) {
                Vec::new()
            } else {
                vec![active.bit()]
            };
            let fields = match case {
                Malformed::EmptyFieldName => vec![InteractionField::bits("", [value])],
                Malformed::EmptyField => {
                    vec![InteractionField::bits("value", std::iter::empty())]
                }
                Malformed::DuplicateField => vec![
                    InteractionField::bits("value", [value]),
                    InteractionField::bits("value", [value]),
                ],
                Malformed::NonDivisibleWidth => {
                    vec![InteractionField::little_endian("value", 2, [value])]
                }
                _ => vec![InteractionField::bits("value", [value])],
            };
            builder.interaction(
                channel,
                kind,
                InteractionDirection::Receive,
                fields,
                multiplicity,
                active,
                InteractionScope::new(scope, 0),
            );
            builder.finish();
        })
        .expect_err("malformed interaction should panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        assert!(
            message.contains(expected),
            "{case:?} panicked with unexpected message: {message}"
        );
    }
}

#[test]
#[should_panic(expected = "expression belongs to another circuit builder")]
fn interaction_rejects_cross_circuit_values() {
    let mut first = CircuitBuilder::new();
    let foreign = first.input();

    let mut second = CircuitBuilder::new();
    let active = second.input_selector("active");
    second.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [InteractionField::bits("value", [foreign])],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
}
