use super::*;

#[test]
fn constant_is_pinned_and_padding_is_forced_to_zero() {
    use crate::challenger::FsChallenger;
    use crate::field::F128;
    use crate::lincheck::{self, LincheckCircuit, QuirkyPoint, SkipPoint};

    let circuit = support::circuit(1, 1, |b, cols| {
        b.define_linear(cols.witness[0], cols.input[0]);
    });
    let r1cs = circuit.to_block_r1cs(3, 0, 0).unwrap();

    assert_eq!(r1cs.const_pin, Some(circuit.one().index()));
    assert!(r1cs.c0_is_identity());
    assert!(r1cs.a_0.rows[3].is_empty());
    assert!(r1cs.b_0.rows[3].is_empty());

    let mut witness = circuit.evaluate_r1cs(&[true], 3).unwrap();
    assert!(r1cs.satisfies(&witness));
    witness[3] = true;
    assert!(!r1cs.satisfies(&witness));

    let lincheck_r1cs = circuit.to_block_r1cs(3, 0, 3).unwrap();
    let lincheck_circuit = lincheck_r1cs.sparse_lincheck_circuit();
    assert_eq!(lincheck_circuit.const_pin_col(), lincheck_r1cs.const_pin);
    let point = QuirkyPoint {
        z_skip: SkipPoint::Phi8(F128::ZERO),
        x_inner_rest: vec![F128::ZERO; lincheck_r1cs.k_log],
        x_outer: vec![F128::ZERO; lincheck_r1cs.n_log()],
    };
    let all_zero_witness = vec![0u8; 1 << (lincheck_r1cs.m - 3)];

    for pin in [None, lincheck_r1cs.const_pin] {
        let circuit = lincheck::SparseMatrixCircuit::new(&lincheck_r1cs.a_0, &lincheck_r1cs.b_0)
            .with_const_pin(pin);
        let challenger = || FsChallenger::new(b"dsl-const-pin");
        let (proof, _) = lincheck::prove(
            &all_zero_witness,
            lincheck_r1cs.m,
            lincheck_r1cs.k_log,
            0,
            &circuit,
            &point,
            &mut challenger(),
        );
        let verified = lincheck::verify(
            lincheck_r1cs.m,
            lincheck_r1cs.k_log,
            0,
            &circuit,
            &point,
            F128::ZERO,
            F128::ZERO,
            &proof,
            &mut challenger(),
        );
        // An all-zero witness is valid only without the statement-level ONE pin.
        assert_eq!(verified.is_ok(), pin.is_none());
    }
}
