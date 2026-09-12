use super::*;

#[path = "tests/downstream.rs"]
mod downstream;
#[path = "tests/identity.rs"]
mod identity;

#[test]
fn exhaustive_composed_addition_and_advice_checks() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    // ONE + 17 supplied bits + 6 carry products + 1 inactive bit + 4 output bits.
    assert_eq!(compiled.circuit().value_count(), 29);
    assert_eq!(compiled.circuit().row_count(), 33);
    let r1cs = compiled.circuit().to_block_r1cs(6, 0, 0).unwrap();
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..16 {
                let trace = generate_trace(&compiled, &[Some(Event { a, b, c })]).unwrap();
                let (cols, logical) = &trace[0];
                assert_eq!(read_word(cols.output), (a + b + c) & 15);
                assert!(!cols.witness.inactive);
                let mut witness = logical.clone();
                witness.resize(64, false);
                assert!(r1cs.satisfies(&witness));
                for value in compiled.columns().output {
                    witness[value.index()] ^= true;
                    assert!(!r1cs.satisfies(&witness));
                    witness[value.index()] ^= true;
                }
                let invalid = compiled.evaluate(|cols| {
                    write_word(cols.inputs.a, a);
                    write_word(cols.inputs.b, b);
                    write_word(cols.inputs.c, c);
                    *cols.inputs.active = true;
                    write_word(cols.claimed_sum, ((a + b + c) & 15) ^ 1);
                });
                assert!(invalid.is_err());
            }
        }
    }
}

#[test]
fn inactive_invocations_are_evaluated_and_interactions_are_disabled() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let trace = generate_trace(&compiled, &[Some(Event { a: 3, b: 4, c: 5 }), None]).unwrap();
    assert_eq!(read_word(trace[0].0.output), 12);
    assert_eq!(read_word(trace[1].0.output), 0);
    assert!(trace[1].0.witness.inactive);
    let interaction = &compiled.circuit().interactions()[0];
    assert_eq!(interaction.effective_multiplicity_bits(&trace[0].1), [true]);
    assert_eq!(
        interaction.effective_multiplicity_bits(&trace[1].1),
        [false]
    );
    assert_eq!(interaction.message()[0].values(), compiled.columns().output);
    for advice in 0..16 {
        let (cols, _) = compiled
            .evaluate(|cols| write_word(cols.claimed_sum, advice))
            .unwrap();
        assert!(cols.witness.inactive);
        assert_eq!(read_word(cols.claimed_sum), advice);
    }
}

#[test]
fn operation_bindings_retain_virtual_results_and_exact_columns() {
    let compiled = CircuitBuilder::compile(DoubleAdd, DoubleAdd::eval);
    let operations = compiled.operations();
    assert_eq!(operations.len(), 2);
    assert_eq!(operations[0].kind, "Add4");
    assert_eq!(operations[1].kind, "Add4");
    assert_eq!(operations[0].rows.len(), 3);
    assert_eq!(operations[1].rows.len(), 3);
    assert_eq!(
        operations[0].output.expressions,
        operations[1].inputs[0].expressions
    );
    assert_eq!(
        operations[0].columns,
        compiled.columns().witness.first.carry_product
    );
    assert_eq!(
        operations[1].columns,
        compiled.columns().witness.second.carry_product
    );
    for operation in operations {
        for word in operation
            .inputs
            .iter()
            .chain(std::iter::once(&operation.output))
        {
            assert_eq!(word.expressions.len(), 4);
            for &expression in &word.expressions {
                assert!(compiled.circuit().support(expression).is_some());
            }
        }
    }
    // The marker intentionally has no consumer; it must still be computed.
    assert_eq!(compiled.unused_values(), [("witness.inactive", 0)]);
}
