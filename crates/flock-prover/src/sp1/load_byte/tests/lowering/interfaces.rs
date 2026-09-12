use super::*;

#[test]
fn converted_outputs_and_interactions_preserve_source_bindings() {
    let chip = LoadByteCircuit::build(2);
    let lowered = chip.lower(LoweringMode::RequireIdentityC).unwrap();
    assert_eq!(chip.circuit().schema().len(), lowered.schema().len());
    assert_eq!(
        chip.circuit().interactions().len(),
        lowered.interactions().len()
    );
    for opcode in [LoadByteOpcode::Lb, LoadByteOpcode::Lbu] {
        for offset in 0..8 {
            let event = event(opcode, 0x1_0000, offset, 0xc342_8101_ff80_7f00);
            let trace = chip.generate_trace(&[event]).unwrap();
            let values = lowered.evaluate(&chip.honest_inputs(&[event])).unwrap();
            let projected = lowered.project(&values).unwrap();
            assert_eq!(projected, trace.values);
            assert_eq!(chip.output(&projected, 0), event.result());
            assert_eq!(chip.output(&projected, 1), 0);
            for (old, new) in chip.circuit().schema().iter().zip(lowered.schema()) {
                assert_eq!(old.name, new.name);
                assert_eq!(old.role, new.role);
                assert_eq!(old.alignment_bits, new.alignment_bits);
                assert_eq!(old.values.len(), new.values.len());
                for (&old, &new) in old.values.iter().zip(&new.values) {
                    assert_eq!(lowered.mapped_value(old), Some(new));
                    assert_eq!(trace.values[old.index()], values[new.index()]);
                }
            }
            for (old, new) in chip
                .circuit()
                .interactions()
                .iter()
                .zip(lowered.interactions())
            {
                assert_eq!(old.channel(), new.channel());
                assert_eq!(old.kind(), new.kind());
                assert_eq!(old.direction(), new.direction());
                assert_eq!(old.scope(), new.scope());
                assert_eq!(lowered.mapped_value(old.selector()), Some(new.selector()));
                assert_eq!(
                    old.effective_multiplicity_bits(&trace.values),
                    new.effective_multiplicity_bits(&values)
                );
                assert_eq!(
                    new.multiplicity(),
                    old.multiplicity()
                        .iter()
                        .map(|&v| lowered.mapped_value(v).unwrap())
                        .collect::<Vec<_>>()
                );
                assert_eq!(old.message().len(), new.message().len());
                for (old, new) in old.message().iter().zip(new.message()) {
                    assert_eq!(old.name(), new.name());
                    assert_eq!(old.encoding(), new.encoding());
                    assert_eq!(
                        new.values(),
                        old.values()
                            .iter()
                            .map(|&v| lowered.mapped_value(v).unwrap())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        decode(&trace.values, old.values()),
                        decode(&values, new.values())
                    );
                }
            }
        }
    }
}
