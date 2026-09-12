use super::*;
use crate::circuit::boolean::tests::support;

fn fixture() -> LoweredCircuit {
    let source = support::circuit(1, 1, |b, cols| {
        b.assert_zero(cols.input[0]);
        b.define_linear(cols.witness[0], cols.input[0]);
    });
    source.lower_identity_c().unwrap()
}

#[test]
#[should_panic(expected = "a cancellation check must not define t")]
fn cancellation_cannot_be_presented_as_a_definition() {
    let mut lowered = fixture();
    let aux = &lowered.auxiliaries[0];
    lowered.rows[aux.cancellation_row.index].defined_value = Some(aux.cancellation);
    lowered.validate();
}
