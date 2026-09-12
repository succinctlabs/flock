use super::*;
use flock_core::circuit::boolean::{ColumnRole, InteractionEncoding};

#[test]
fn every_byte_offset_and_signed_result_match_the_native_semantics() {
    let chip = LoadByteCircuit::build(1);
    for opcode in [LoadByteOpcode::Lb, LoadByteOpcode::Lbu] {
        for offset in 0..8 {
            for byte in 0..256u64 {
                let event = event(opcode, 0x1_0000, offset, byte << (8 * offset));
                let trace = chip.generate_trace(&[event]).unwrap();
                assert_eq!(chip.output(&trace.values, 0), event.result());
                assert_eq!(
                    trace.rows[0].sign,
                    opcode == LoadByteOpcode::Lb && byte >= 128
                );
            }
        }
    }
}

#[test]
fn typed_trace_and_bindings_match_the_canonical_witness() {
    let chip = LoadByteCircuit::build(4);
    let events = [
        event(LoadByteOpcode::Lb, 0x1_0000, 7, 0x8070_6050_4030_2010),
        event(LoadByteOpcode::Lbu, 0x2_0000, 0, 0xff),
    ];
    let trace = chip.generate_trace(&events).unwrap();
    assert_eq!(
        trace.values,
        chip.circuit()
            .evaluate(&chip.honest_inputs(&events))
            .unwrap()
    );
    for (row, cols) in trace.rows.iter().enumerate() {
        assert_eq!(cols.real, row < events.len());
        let expected = events.get(row).map_or(0, |event| event.result());
        let result = cols
            .result
            .iter()
            .enumerate()
            .fold(0u64, |value, (bit, &set)| value | (u64::from(set) << bit));
        assert_eq!(result, expected);
        assert_eq!(chip.output(&trace.values, row), expected);
    }
    for (row, cols) in chip.columns().iter().enumerate() {
        let memory = &chip.circuit().interactions()[row * 2];
        let register = &chip.circuit().interactions()[row * 2 + 1];
        assert_eq!(memory.selector(), cols.real);
        assert_eq!(memory.message()[0].values()[..3], cols.aligned_low);
        assert_eq!(memory.message()[0].values()[3..], cols.address.value[3..]);
        assert_eq!(register.message()[0].encoding(), InteractionEncoding::Bits);
        assert_eq!(register.message()[1].values(), cols.result);
    }
    let schema = chip.compiled().schema();
    assert_eq!(
        schema
            .iter()
            .filter(|field| field.role == ColumnRole::Advice(ADVICE_TYPE))
            .count(),
        4
    );
    assert_eq!(chip.compiled().operations().len(), 12);
    assert_eq!(chip.circuit().value_count(), 1 + 550 * 4);
    assert_eq!(chip.circuit().row_count(), 1 + 576 * 4 + 3);
}

#[test]
fn every_real_row_pattern_obeys_the_prefix_constraint() {
    let chip = LoadByteCircuit::build(4);
    let event = event(LoadByteOpcode::Lbu, 0x1_0000, 0, 0x42);
    for mask in 0..16usize {
        let rows: Vec<_> = (0..4)
            .map(|i| (mask >> i & 1 != 0).then_some(event))
            .collect();
        let prefix = mask & (mask + 1) == 0;
        let inputs = chip.encode_rows(&rows);
        assert_eq!(chip.circuit().evaluate(&inputs).is_ok(), prefix);
    }
}

#[test]
fn shape_and_event_counts_are_checked() {
    assert!(std::panic::catch_unwind(|| LoadByteCircuit::build(0)).is_err());
    let chip = LoadByteCircuit::build(1);
    let event = event(LoadByteOpcode::Lbu, 0x1_0000, 0, 0);
    assert!(std::panic::catch_unwind(|| chip.generate_trace(&[event, event])).is_err());
    assert!(std::panic::catch_unwind(|| chip.encode_rows(&[])).is_err());
}
