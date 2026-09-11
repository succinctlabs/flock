use super::*;
use crate::lincheck::LincheckCircuit;

#[test]
fn transposes_and_lincheck_match_independent_sparse_projections() {
    let mut rng = crate::test_rng::Rng::new(0xcafe_a110);
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        for case in 0..7 {
            for permuted in [false, true] {
                let lowered = fixture(case, permuted, mode);
                let plan = lowered.walk_plan().unwrap();
                let matrix = lowered.to_block_r1cs(7, 0, 0).unwrap();
                let [ea, eb, ec] = std::array::from_fn(|_| rng.f128_vec(128));
                let mut expected = sparse_transpose(&matrix.a_0, &ea);
                let b = sparse_transpose(&matrix.b_0, &eb);
                let c = sparse_transpose(&matrix.c_0, &ec);
                for i in 0..128 {
                    expected[i] += b[i] + c[i];
                }
                assert_eq!(plan.transpose(&ea, &eb, &ec).unwrap(), expected);
                if plan.c_is_identity() {
                    assert_eq!(plan.transpose_identity_c(&ea, &eb, &ec).unwrap(), expected);
                } else {
                    assert_eq!(
                        plan.transpose_identity_c(&ea, &eb, &ec),
                        Err(WalkError::IdentityCRequired)
                    );
                }
                let adapter = plan.lincheck_circuit(7).unwrap();
                let sparse = matrix.sparse_lincheck_circuit();
                let alpha = rng.f128();
                assert_eq!(adapter.const_pin_col(), matrix.const_pin);
                assert_eq!(adapter.n_cols(), sparse.n_cols());
                assert_eq!(
                    adapter.fold_alpha_batched(alpha, &ea),
                    sparse.fold_alpha_batched(alpha, &ea)
                );
            }
        }
    }
}

#[test]
fn cancellation_boundaries_remain_in_row_side_walking_even_when_honestly_zero() {
    let lowered = fixture(6, true, LoweringMode::RequireIdentityC);
    let plan = lowered.walk_plan().unwrap();
    let matrix = lowered.to_block_r1cs(7, 0, 0).unwrap();
    let zero = vec![F128::ZERO; 128];
    for aux in lowered.auxiliaries() {
        let mut weights = zero.clone();
        weights[lowered.layout().row_position(aux.cancellation_row).unwrap()] =
            F128::new(0x1234, 0xabcd);
        let expected = sparse_transpose(&matrix.b_0, &weights);
        assert_eq!(
            expected[lowered.layout().value_position(aux.cancellation).unwrap()],
            F128::new(0x1234, 0xabcd)
        );
        assert_eq!(plan.transpose(&zero, &weights, &zero).unwrap(), expected);
        assert_eq!(
            plan.transpose_identity_c(&zero, &weights, &zero).unwrap(),
            expected
        );
    }
    let mut witness = lowered.evaluate(&[false; 3]).unwrap();
    for aux in lowered.auxiliaries() {
        witness[aux.cancellation.index()] = true;
    }
    assert!(lowered.accepts(&witness));
    let physical = physical(&lowered, &witness, 128);
    assert!(matrix.satisfies(&physical));
    let b = matrix.apply_b(&physical);
    for aux in lowered.auxiliaries() {
        assert!(b[lowered.layout().row_position(aux.cancellation_row).unwrap()]);
    }
}
