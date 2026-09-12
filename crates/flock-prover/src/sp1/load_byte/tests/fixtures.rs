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

// None means rejection; Some gives the expected output of the first row.
pub(super) fn input_cases(chip: &LoadByteCircuit) -> Vec<(String, Vec<bool>, Option<u64>)> {
    assert_eq!(chip.capacity(), 2);
    let row = event(LoadByteOpcode::Lb, 0x1_0000, 7, 0x8070_6050_4030_2010);
    let valid = chip.honest_inputs(&[row]);
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
    let mut cases = vec![
        ("signed".into(), valid, Some(row.result())),
        ("padding".into(), chip.honest_inputs(&[]), Some(0)),
        (
            "non-prefix".into(),
            chip.encode_rows(&[None, Some(row)]),
            None,
        ),
        ("bad advice".into(), bad_advice, None),
        ("both selectors".into(), both_selectors, None),
    ];
    let memory = 0xc342_8101_ff80_7f00;
    for (name, opcode, b, c, valid) in [
        ("below guard", LoadByteOpcode::Lb, 0xffff, 0, false),
        ("above range", LoadByteOpcode::Lbu, 1 << 48, 0, false),
        ("lower bound", LoadByteOpcode::Lbu, 1 << 16, 0, true),
        ("upper bound", LoadByteOpcode::Lbu, (1 << 48) - 1, 0, true),
        ("lower carry", LoadByteOpcode::Lbu, 0xffff, 1, true),
        (
            "47-bit carry",
            LoadByteOpcode::Lb,
            0x7fff_ffff_ffff,
            1,
            true,
        ),
        ("64-bit wrap", LoadByteOpcode::Lbu, u64::MAX, 0x1_0001, true),
        ("upper carry", LoadByteOpcode::Lbu, (1 << 48) - 1, 1, false),
    ] {
        let row = event(opcode, b, c, memory);
        cases.push((
            name.into(),
            chip.honest_inputs(&[row]),
            valid.then(|| row.result()),
        ));
    }
    for opcode in [LoadByteOpcode::Lb, LoadByteOpcode::Lbu] {
        for offset in 0..8 {
            let row = event(opcode, 0x1_0000, offset, memory);
            cases.push((
                format!("{opcode:?} offset {offset}"),
                chip.honest_inputs(&[row]),
                Some(row.result()),
            ));
        }
    }
    cases
}
