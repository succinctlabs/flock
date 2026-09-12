use super::*;
use crate::circuit::boolean::tests::support::{self, Fields, TestSchema};
use crate::circuit::boolean::{CircuitBuilder, ColumnRole};

#[test]
fn interface_metadata_is_typed_inspectable_and_allocation_free() {
    let compiled = CircuitBuilder::compile(
        Fields(vec![
            ("active", ColumnRole::Input, 1, 1),
            (
                "selected-byte",
                ColumnRole::Advice("sp1.load-byte.selected-byte/v1"),
                8,
                1,
            ),
        ]),
        |builder, cols| {
            let active = builder.column_selector(cols[0][0]);
            let values = builder.value_count();
            let rows = builder.row_count();
            let expressions = builder.expression_count();
            builder.interaction(
                "memory",
                "read",
                InteractionDirection::Receive,
                [InteractionField::columns(
                    "value",
                    8,
                    cols[1].iter().copied(),
                )],
                [active.bit()],
                active,
                InteractionScope::new("load-byte", 0),
            );
            assert_eq!(builder.value_count(), values);
            assert_eq!(builder.row_count(), rows);
            assert_eq!(builder.expression_count(), expressions);
        },
    );
    let active = compiled.columns()[0][0];
    let circuit = compiled.circuit;

    assert_eq!(
        circuit.column("selected-byte").unwrap().role,
        ColumnRole::Advice("sp1.load-byte.selected-byte/v1")
    );
    let interaction = &circuit.interactions()[0];
    assert_eq!(interaction.channel(), "memory");
    assert_eq!(interaction.kind(), "read");
    assert_eq!(interaction.direction(), InteractionDirection::Receive);
    assert_eq!(interaction.scope(), &InteractionScope::new("load-byte", 0));
    assert_eq!(interaction.selector(), active);
    assert_eq!(interaction.multiplicity(), &[active]);
    assert_eq!(interaction.message()[0].values().len(), 8);
}

#[test]
fn interaction_multiplicity_is_gated_by_its_selector() {
    let circuit = support::circuit(3, 0, |builder, cols| {
        let active = builder.column_selector(cols.input[0]);
        let multiplicity = [cols.input[1].0, cols.input[2].0];
        builder.interaction(
            "test",
            "unequal-activation",
            InteractionDirection::Send,
            [InteractionField::bits("value", [active.bit()])],
            multiplicity,
            active,
            InteractionScope::new("test", 0),
        );
    });
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
    let circuit = support::circuit(3, 0, |builder, cols| {
        builder.assert_when_eq(
            builder.column_selector(cols.input[0]),
            cols.input[1],
            cols.input[2],
        );
    });
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
        let role = if advice {
            ColumnRole::Advice("test/v1")
        } else {
            ColumnRole::Input
        };
        CircuitBuilder::compile(Fields(vec![("value", role, 8, 1)]), |_, _| {}).circuit
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
#[should_panic(expected = "interaction message must not be empty")]
fn empty_interaction_message_is_rejected() {
    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 2,
        witnesses: 0,
    };
    let cols = builder.reserve_columns(&schema);
    let active = builder.column_selector(cols.input[0]);
    builder.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
    builder.finish_columns(schema);
}

#[test]
#[should_panic(expected = "interaction element width must be nonzero")]
fn malformed_interaction_encoding_is_rejected() {
    let mut builder = CircuitBuilder::new();
    let schema = TestSchema {
        inputs: 2,
        witnesses: 0,
    };
    let cols = builder.reserve_columns(&schema);
    let active = builder.column_selector(cols.input[0]);
    let value = cols.input[1];
    builder.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [InteractionField::columns("value", 0, [value])],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
    builder.finish_columns(schema);
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
            let schema = TestSchema {
                inputs: 2,
                witnesses: 0,
            };
            let cols = builder.reserve_columns(&schema);
            let active = builder.column_selector(cols.input[0]);
            let value = cols.input[1];
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
                Malformed::EmptyFieldName => vec![InteractionField::column_bits("", [value])],
                Malformed::EmptyField => {
                    vec![InteractionField::column_bits("value", std::iter::empty())]
                }
                Malformed::DuplicateField => vec![
                    InteractionField::column_bits("value", [value]),
                    InteractionField::column_bits("value", [value]),
                ],
                Malformed::NonDivisibleWidth => {
                    vec![InteractionField::columns("value", 2, [value])]
                }
                _ => vec![InteractionField::column_bits("value", [value])],
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
            builder.finish_columns(schema);
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
    let foreign = first
        .reserve_columns(&TestSchema {
            inputs: 1,
            witnesses: 0,
        })
        .input[0];

    let mut second = CircuitBuilder::new();
    let cols = second.reserve_columns(&TestSchema {
        inputs: 1,
        witnesses: 0,
    });
    let active = second.column_selector(cols.input[0]);
    second.interaction(
        "memory",
        "read",
        InteractionDirection::Receive,
        [InteractionField::column_bits("value", [foreign])],
        [active.bit()],
        active,
        InteractionScope::new("load-byte", 0),
    );
}
