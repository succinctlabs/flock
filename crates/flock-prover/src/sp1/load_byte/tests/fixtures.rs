use super::*;

// Test-only population deliberately skips assertions so invalid inputs produce
// candidates. Each relation must reject these without relying on its evaluator.
pub(super) fn candidate(circuit: &BooleanCircuit, inputs: &[bool]) -> Vec<bool> {
    let mut values = vec![false; circuit.value_count()];
    values[circuit.one().index()] = true;
    assert_eq!(inputs.len(), circuit.inputs().len());
    for (&value, &bit) in circuit.inputs().iter().zip(inputs) {
        values[value.index()] = bit;
    }
    for row in circuit.rows() {
        if matches!(row.kind(), RowKind::And | RowKind::Materialize) {
            let eval = |expr| {
                circuit
                    .support(expr)
                    .unwrap()
                    .iter()
                    .fold(false, |sum, value| sum ^ values[value.index()])
            };
            values[row.defined_value().unwrap().index()] = eval(row.lhs()) & eval(row.rhs());
        }
    }
    values
}

pub(super) fn advice_inputs(chip: &LoadByteCircuit, active: bool, advice: u64) -> Vec<bool> {
    assert_eq!(chip.capacity(), 1);
    chip.compiled().inputs(|mut rows| {
        let cols = rows.remove(0);
        *cols.is_lbu = active;
        *cols.b[16] = true;
        for (i, bit) in cols.selected_byte.into_iter().enumerate() {
            *bit = advice >> i & 1 != 0;
        }
    })
}

pub(super) fn input_cases(chip: &LoadByteCircuit) -> Vec<(Vec<bool>, bool)> {
    assert_eq!(chip.capacity(), 2);
    let row = event(LoadByteOpcode::Lb, 0x1_0000, 7, 0x8070_6050_4030_2010);
    let valid = chip.honest_inputs(&[row]);
    let non_prefix = chip.encode_rows(&[None, Some(row)]);
    let bad_address = chip.honest_inputs(&[event(LoadByteOpcode::Lb, 0xffff, 0, 0)]);
    let high_address = chip.honest_inputs(&[event(LoadByteOpcode::Lbu, 1 << 48, 0, 0)]);
    let mut bad_advice = valid.clone();
    let advice = chip.columns()[0].selected_byte[7];
    let advice_input = chip
        .circuit()
        .inputs()
        .iter()
        .position(|&value| value == advice)
        .unwrap();
    bad_advice[advice_input] ^= true;
    let mut both_selectors = valid.clone();
    both_selectors[0] = true;
    both_selectors[1] = true;
    vec![
        (valid, true),
        (chip.honest_inputs(&[]), true),
        (non_prefix, false),
        (bad_address, false),
        (high_address, false),
        (bad_advice, false),
        (both_selectors, false),
    ]
}
