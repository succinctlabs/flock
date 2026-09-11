use super::*;

fn source() -> BooleanCircuit {
    let mut b = CircuitBuilder::new();
    let [a, c] = b.input_bits::<2>("input");
    let product = b.and(a, c);
    b.output("product", [product]);
    b.assert_zero(a);
    b.assert_zero(c);
    b.finish()
}

#[test]
fn virtual_nonzero_c_is_checked_without_reinterpreting_the_source() {
    let mut b = CircuitBuilder::new();
    let [a, c] = b.input_bits::<2>("input");
    let not_a = b.xor2(b.one(), a);
    let result = b.xor2(a, c);
    b.constrain(not_a, c, result);
    let source = b.finish();
    let checker = source.identity_checker();
    for a in [false, true] {
        for c in [false, true] {
            let witness = checker.extend(&[true, a, c]).unwrap();
            assert_eq!(checker.accepts(&witness), ((!a) & c) == (a ^ c));
            assert_eq!(checker.accepts(&witness), source.evaluate(&[a, c]).is_ok());
        }
    }
}

#[test]
fn extension_and_projection_preserve_every_source_witness() {
    let source = source();
    let checker = source.identity_checker();
    let direct = source.to_block_r1cs(3, 0, 0).unwrap();
    let unbound = checker.unbound_circuit().to_block_r1cs(4, 0, 0).unwrap();
    assert!(unbound.c0_is_identity());
    let mut failures = [false; 3];
    for mask in 0..1usize << source.value_count() {
        let logical: Vec<_> = (0..source.value_count())
            .map(|i| mask >> i & 1 != 0)
            .collect();
        let mut padded = logical.clone();
        padded.resize(8, false);
        if !logical[source.one().index()] {
            assert!(checker.extend(&logical).is_none());
            continue;
        }
        let extended = checker.extend(&logical).unwrap();
        assert_eq!(checker.project(&extended), Some(logical.clone()));
        assert_eq!(checker.accepts(&extended), direct.satisfies(&padded));
        let mut checker_padded = extended.clone();
        checker_padded.resize(16, false);
        assert!(
            unbound.satisfies(&checker_padded),
            "unbound rows compute even for bad source witnesses"
        );
        let count = usize::from(logical[1]) + usize::from(logical[2]);
        if logical[3] == (logical[1] & logical[2]) {
            failures[count] = true;
            assert_eq!(extended[checker.accept().index()], count == 0);
        }
    }
    assert_eq!(
        failures, [true; 3],
        "include zero, one, and two failed assertions"
    );
}

#[test]
fn exhaustive_checker_witnesses_cannot_forge_accept_or_auxiliaries() {
    let source = source();
    let checker = source.identity_checker();
    let direct = source.to_block_r1cs(3, 0, 0).unwrap();
    let unbound = checker.unbound_circuit().to_block_r1cs(4, 0, 0).unwrap();
    assert_eq!(checker.unbound_circuit().value_count(), 15);
    let mut accepted = 0;
    for mask in 0..1usize << 15 {
        let witness: Vec<_> = (0..15).map(|i| mask >> i & 1 != 0).collect();
        let mut padded = witness.clone();
        padded.push(false);
        // `satisfies` checks equations; statement pins must be enforced separately.
        let bound_sparse = unbound.satisfies(&padded)
            && witness[unbound.const_pin.unwrap()]
            && witness[checker.accept().index()];
        assert_eq!(checker.accepts(&witness), bound_sparse);
        if bound_sparse {
            accepted += 1;
            let projected = checker.project(&witness).unwrap();
            assert_eq!(checker.extend(&projected), Some(witness));
            let mut original = projected;
            original.resize(8, false);
            assert!(direct.satisfies(&original));
        }
    }
    assert_eq!(accepted, 1);
}

#[test]
fn empty_computation_still_enforces_one_and_terminal_binding() {
    let source = CircuitBuilder::new().finish();
    let checker = source.identity_checker();
    assert_eq!(checker.unbound_circuit().value_count(), 3);
    let witness = checker.extend(&[true]).unwrap();
    assert_eq!(witness, [true; 3]);
    assert!(checker.accepts(&witness));
    assert!(!checker.accepts(&[false; 3]));
    for i in 0..witness.len() {
        let mut bad = witness.clone();
        bad[i] = false;
        assert!(!checker.accepts(&bad));
    }
    for malformed in [vec![], vec![true; 2]] {
        assert!(checker.extend(&malformed).is_none());
    }
    for malformed in [vec![], vec![true; 2], vec![true; 4]] {
        assert!(checker.project(&malformed).is_none());
        assert!(!checker.accepts(&malformed));
    }
    let r1cs = checker.unbound_circuit().to_block_r1cs(2, 0, 0).unwrap();
    assert!(r1cs.satisfies(&[true, true, true, false]));
    assert!(!r1cs.satisfies(&[true; 4]), "padding is pinned to zero");
}

#[test]
fn projection_and_manifest_account_for_every_checker_value_and_row() {
    let source = source();
    let before = source.structure_digest();
    let checker = source.identity_checker();
    let circuit = checker.unbound_circuit();
    let mut covered = vec![false; circuit.value_count()];
    for row in source.rows() {
        if let Some(value) = row.defined_value() {
            let mapped = checker.mapped_value(value).unwrap();
            assert!(!std::mem::replace(&mut covered[mapped.index()], true));
            assert_eq!(
                circuit.definition_row(mapped).unwrap().index(),
                mapped.index()
            );
        }
    }
    let foreign = CircuitBuilder::new().finish();
    assert!(checker.mapped_value(foreign.one()).is_none());
    for aux in checker.auxiliaries() {
        assert!(!std::mem::replace(&mut covered[aux.value.index()], true));
        assert_eq!(circuit.definition_row(aux.value), Some(aux.row));
        assert_eq!(aux.value.index(), aux.row.index());
        assert!(!aux.source_rows.is_empty());
        assert!(aux.source_rows.end <= source.row_count());
        if aux.purpose == LoweringPurpose::RowProduct {
            assert_eq!(aux.source_rows.len(), 1);
        }
    }
    assert!(covered.iter().all(|bit| *bit));
    assert_eq!(
        checker.auxiliaries().last().unwrap().value,
        checker.accept()
    );
    assert_eq!(
        checker.auxiliaries().last().unwrap().source_rows,
        0..source.row_count()
    );
    assert_eq!(source.structure_digest(), before);
    assert_ne!(circuit.structure_digest(), before);
    assert_eq!(
        circuit.structure_digest(),
        source
            .identity_checker()
            .unbound_circuit()
            .structure_digest()
    );
}
