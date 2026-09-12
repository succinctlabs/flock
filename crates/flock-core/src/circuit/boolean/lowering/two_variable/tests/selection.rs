use super::*;
use crate::circuit::boolean::tests::support;

fn fixture(assertion: bool) -> BooleanCircuit {
    support::circuit(2, 1, |b, cols| {
        let (x, y) = (cols.input[0], cols.input[1]);
        if assertion {
            b.constrain(x, y, x);
        }
        b.define_and(cols.witness[0], x, y);
    })
}

#[test]
fn modes_preserve_equations_or_enforce_identity_c() {
    for assertion in [false, true] {
        let source = fixture(assertion);
        let direct = source.lower(LoweringMode::Direct).unwrap();
        let original = source.to_block_r1cs(3, 0, 0).unwrap();
        assert!(direct.auxiliaries().is_empty());
        assert_eq!(
            direct.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
            original.statement_digest()
        );
        let lowered = source.lower_identity_c().unwrap();
        assert!(lowered.c_is_identity());
        assert_eq!(lowered.auxiliaries().len(), usize::from(assertion));
        assert!(lowered.to_block_r1cs(3, 0, 0).unwrap().c0_is_identity());
        if !assertion {
            assert_eq!(lowered.layout(), direct.layout());
            assert_eq!(
                lowered.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
                original.statement_digest()
            );
        }
        for n in 0..1 << source.value_count() {
            let z = bits(n, source.value_count());
            assert_eq!(direct.extend(&z), Some(z.clone()));
            let mut padded = z.clone();
            padded.resize(8, false);
            assert_eq!(direct.accepts(&z), z[0] && original.satisfies(&padded));
        }
        for n in 0..4 {
            let inputs = bits(n, 2);
            assert_eq!(direct.evaluate(&inputs), source.evaluate(&inputs));
            assert_eq!(
                lowered.evaluate(&inputs).is_ok(),
                source.evaluate(&inputs).is_ok()
            );
        }
    }
}

#[test]
fn capacity_and_dimension_errors_remain_explicit() {
    let lowered = fixture(true).lower_identity_c().unwrap();
    assert!(matches!(
        lowered.to_block_r1cs(2, 0, 0),
        Err(R1csBuildError::Capacity {
            required: 7,
            actual: 4
        })
    ));
    assert!(lowered.to_block_r1cs(3, 0, 0).unwrap().c0_is_identity());
    assert!(matches!(
        lowered.to_block_r1cs(3, 4, 0),
        Err(R1csBuildError::InvalidKSkip { .. })
    ));
    assert!(matches!(
        lowered.to_block_r1cs(3, 0, usize::MAX),
        Err(R1csBuildError::DimensionOverflow)
    ));
    assert!(matches!(
        lowered.to_block_r1cs(usize::MAX, 0, 0),
        Err(R1csBuildError::InvalidKLog(_))
    ));
}
