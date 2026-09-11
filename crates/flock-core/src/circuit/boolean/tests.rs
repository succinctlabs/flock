use super::*;

#[path = "tests/placement.rs"]
mod placement;
#[path = "tests/relation.rs"]
mod relation;

#[test]
fn materialization_is_explicit_and_exactly_one_row() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let initial_values = builder.value_count();
    let initial_rows = builder.row_count();

    let sum = builder.xor([a.expr(), b.expr()]);
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows);

    // Copies and ordinary Rust container operations are shape-neutral.
    let copied = sum;
    let expressions = [sum, copied];
    assert_eq!(builder.value_count(), initial_values);
    assert_eq!(builder.row_count(), initial_rows);
    assert_eq!(expressions[0], expressions[1]);

    let sum_bit = builder.materialize(sum);
    assert_eq!(builder.value_count(), initial_values + 1);
    assert_eq!(builder.row_count(), initial_rows + 1);
    let _product = builder.and(sum_bit.expr(), a.expr());
    assert_eq!(builder.value_count(), initial_values + 2);
    assert_eq!(builder.row_count(), initial_rows + 2);

    let circuit = builder.finish();
    let defining_row = circuit
        .rows()
        .iter()
        .find(|row| row.defined_value() == Some(sum_bit.value_id()))
        .unwrap();
    assert_eq!(defining_row.kind(), RowKind::Materialize);
}

#[test]
fn structural_dag_survives_normalization_and_cancellation() {
    let mut builder = CircuitBuilder::new();
    let a = builder.input();
    let b = builder.input();
    let ab = builder.xor([a.expr(), b.expr()]);
    let nested = builder.xor([ab, a.expr()]);
    let nested_id = nested.id();
    let ab_id = ab.id();
    let circuit = builder.finish();

    assert_eq!(
        circuit.expression(nested_id),
        Some(&ExpressionNode::Xor(vec![ab_id, a.expr().id()]))
    );
    assert_eq!(circuit.support(nested_id), Some(vec![b.value_id()]));

    let expressions: Vec<_> = circuit.expressions().collect();
    assert_eq!(expressions.len(), circuit.expression_count());
    assert_eq!(
        expressions[nested_id.index()],
        circuit.expression(nested_id).unwrap()
    );

    let layout = PhysicalLayout::source_order(&circuit);
    assert_eq!(layout.value_positions(), &[0, 1, 2]);
    assert_eq!(layout.row_positions(), &[0, 1, 2]);
}

#[test]
fn changing_materialization_changes_only_the_requested_boundary() {
    fn build(materialize: bool) -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        let a = builder.input();
        let b = builder.input();
        let sum = builder.xor([a.expr(), b.expr()]);
        let lhs = if materialize {
            builder.materialize(sum).expr()
        } else {
            sum
        };
        builder.and(lhs, a.expr());
        builder.finish()
    }

    let virtual_sum = build(false);
    let materialized_sum = build(true);
    assert_eq!(
        materialized_sum.value_count(),
        virtual_sum.value_count() + 1
    );
    assert_eq!(materialized_sum.row_count(), virtual_sum.row_count() + 1);
    assert_eq!(
        materialized_sum
            .rows()
            .iter()
            .filter(|row| row.kind() == RowKind::Materialize)
            .count(),
        1
    );
    let virtual_product = &virtual_sum.rows()[3];
    assert_eq!(virtual_product.kind(), RowKind::And);
    assert_eq!(
        virtual_sum.support(virtual_product.lhs()),
        Some(vec![virtual_sum.inputs()[0], virtual_sum.inputs()[1]])
    );
    let copy = &materialized_sum.rows()[3];
    let materialized_product = &materialized_sum.rows()[4];
    assert_eq!(copy.kind(), RowKind::Materialize);
    assert_eq!(materialized_product.kind(), RowKind::And);
    assert_eq!(
        materialized_sum.support(copy.lhs()),
        Some(vec![
            materialized_sum.inputs()[0],
            materialized_sum.inputs()[1]
        ])
    );
    assert_eq!(
        materialized_sum.support(materialized_product.lhs()),
        Some(vec![copy.defined_value().unwrap()])
    );
    assert_ne!(
        virtual_sum.structure_digest(),
        materialized_sum.structure_digest()
    );
    assert_ne!(
        virtual_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest(),
        materialized_sum
            .to_block_r1cs(3, 0, 0)
            .unwrap()
            .statement_digest()
    );
}

#[test]
fn ripple_carry_example_keeps_linear_carries_virtual() {
    fn full_adder(
        builder: &mut CircuitBuilder,
        a: LinearExpr,
        b: LinearExpr,
        carry_in: LinearExpr,
    ) -> (LinearExpr, LinearExpr) {
        let a_xor_b = builder.xor([a, b]);
        let sum = builder.xor([a_xor_b, carry_in]);
        let a_and_b = builder.and(a, b);
        let carry_and_xor = builder.and(carry_in, a_xor_b);
        let carry_out = builder.xor([a_and_b.expr(), carry_and_xor.expr()]);
        (sum, carry_out)
    }

    let mut builder = CircuitBuilder::new();
    let a = [builder.input(), builder.input()];
    let b = [builder.input(), builder.input()];
    let carry_in = builder.input();

    let (sum_0, carry_0) = full_adder(&mut builder, a[0].expr(), b[0].expr(), carry_in.expr());
    let values_after_low_bit = builder.value_count();
    let (sum_1, carry_1) = full_adder(&mut builder, a[1].expr(), b[1].expr(), carry_0);
    // Passing the virtual carry into the next bit does not materialize it:
    // only the two nonlinear products allocate values in each full adder.
    assert_eq!(builder.value_count(), values_after_low_bit + 2);

    let sum_0 = builder.materialize(sum_0);
    let sum_1 = builder.materialize(sum_1);
    let carry_1 = builder.materialize(carry_1);
    let circuit = builder.finish();
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    assert!(r1cs.c0_is_identity());

    for input in 0u8..32 {
        let bits: [bool; 5] = std::array::from_fn(|i| (input >> i) & 1 == 1);
        let witness = circuit.evaluate_r1cs(&bits, 4).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output = u8::from(witness[sum_0.value_id().index()])
            | (u8::from(witness[sum_1.value_id().index()]) << 1)
            | (u8::from(witness[carry_1.value_id().index()]) << 2);
        let a_value = (input & 1) | (((input >> 1) & 1) << 1);
        let b_value = ((input >> 2) & 1) | (((input >> 3) & 1) << 1);
        let carry_value = (input >> 4) & 1;
        assert_eq!(output, a_value + b_value + carry_value);
    }
}

#[test]
fn word_helpers_are_virtual_until_materialize_word() {
    let mut builder = CircuitBuilder::new();
    let input = builder.input_word::<8>("input");
    let values_before = builder.value_count();
    let rows_before = builder.row_count();
    let rotated = builder.rotate_right(input, 2);
    let shifted = builder.shift_right(input, 3);
    let mixed = builder.xor2_words(rotated, shifted);
    assert_eq!(builder.value_count(), values_before);
    assert_eq!(builder.row_count(), rows_before);

    let output = builder.materialize_word(mixed);
    assert_eq!(builder.value_count(), values_before + 8);
    assert_eq!(builder.row_count(), rows_before + 8);
    builder.output_word("output", output);
    let circuit = builder.finish();
    assert_eq!(
        circuit.port("input").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    assert_eq!(
        circuit.port("output").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    let r1cs = circuit.to_block_r1cs(5, 0, 0).unwrap();

    for value in 0u8..=u8::MAX {
        let input_bits: [bool; 8] = std::array::from_fn(|i| value >> i & 1 == 1);
        let witness = circuit.evaluate_r1cs(&input_bits, 5).unwrap();
        assert!(r1cs.satisfies(&witness));
        let output_value = output.iter().enumerate().fold(0u8, |acc, (i, bit)| {
            acc | (u8::from(witness[bit.value_id().index()]) << i)
        });
        assert_eq!(output_value, value.rotate_right(2) ^ (value >> 3));
    }
}

#[test]
fn fixed_words_are_named_and_constrained() {
    let mut builder = CircuitBuilder::new();
    let fixed = builder.fixed_word::<8>("iv", 0xa5);
    let circuit = builder.finish();
    assert_eq!(
        circuit.port("iv").unwrap().direction(),
        PortDirection::Fixed
    );
    assert_eq!(
        circuit.port("iv").unwrap().encoding(),
        PortEncoding::LittleEndianWord { alignment_bits: 1 }
    );
    let r1cs = circuit.to_block_r1cs(4, 0, 0).unwrap();
    let witness = circuit.evaluate_r1cs(&[], 4).unwrap();
    assert!(r1cs.satisfies(&witness));
    for (i, bit) in fixed.iter().enumerate() {
        assert_eq!(witness[bit.value_id().index()], 0xa5 >> i & 1 == 1);
    }
    let mut mutated = witness;
    mutated[fixed[3].value_id().index()] ^= true;
    assert!(!r1cs.satisfies(&mutated));
}

#[test]
fn port_encoding_is_structural_but_not_arithmetic() {
    fn bits() -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        builder.input_bits::<8>("value");
        builder.finish()
    }

    fn word(alignment_bits: usize) -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        builder.input_word_aligned::<8>("value", alignment_bits);
        builder.finish()
    }

    let bits = bits();
    let unaligned_word = word(1);
    let aligned_word = word(8);
    assert_ne!(bits.structure_digest(), unaligned_word.structure_digest());
    assert_ne!(
        unaligned_word.structure_digest(),
        aligned_word.structure_digest()
    );
    assert_eq!(
        bits.to_block_r1cs(4, 0, 0).unwrap().statement_digest(),
        unaligned_word
            .to_block_r1cs(4, 0, 0)
            .unwrap()
            .statement_digest()
    );
}

#[test]
fn structure_and_relation_digests_are_deterministic_and_golden() {
    fn build() -> BooleanCircuit {
        let mut builder = CircuitBuilder::new();
        let input = builder.input_bits::<2>("input");
        let sum_expr = builder.xor2(input[0], input[1]);
        let sum = builder.materialize(sum_expr);
        builder.assert_zero_product(input[0], input[1]);
        builder.output("sum", [sum]);
        builder.finish()
    }

    let first = build();
    let second = build();
    assert_eq!(first.structure_digest(), second.structure_digest());
    let first_layout = PhysicalLayout::source_order(&first);
    let second_layout = PhysicalLayout::source_order(&second);
    assert_eq!(first_layout, second_layout);
    assert_eq!(
        first.structure_layout_digest(&first_layout).unwrap(),
        second.structure_layout_digest(&second_layout).unwrap()
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        second.to_block_r1cs(3, 0, 0).unwrap().statement_digest()
    );
    assert_eq!(
        first.structure_digest(),
        [
            132, 17, 99, 150, 177, 225, 101, 6, 129, 196, 80, 54, 185, 56, 22, 18, 174, 130, 22,
            164, 248, 233, 108, 114, 130, 93, 11, 188, 243, 65, 167, 185,
        ]
    );
    assert_eq!(
        first.to_block_r1cs(3, 0, 0).unwrap().statement_digest(),
        [
            106, 16, 156, 236, 111, 130, 214, 34, 173, 41, 61, 220, 52, 226, 61, 162, 41, 193, 169,
            235, 128, 224, 120, 125, 227, 239, 50, 114, 147, 168, 88, 148,
        ]
    );
}

#[test]
fn structure_digest_is_process_and_thread_deterministic() {
    const CHILD_ENV: &str = "FLOCK_BOOLEAN_DIGEST_TEST_CHILD";
    const MARKER: &str = "FLOCK_STRUCTURE_DIGEST=";

    fn digest() -> [u8; 32] {
        let mut builder = CircuitBuilder::new();
        let input = builder.input_word_aligned::<8>("input", 8);
        let mixed = builder.xor3(input[0], input[3], input[7]);
        let output = builder.materialize(mixed);
        builder.output("output", [output]);
        builder.finish().structure_digest()
    }

    if std::env::var_os(CHILD_ENV).is_some() {
        println!("{MARKER}{:?}", digest());
        return;
    }

    let run = |rayon_threads: &str| {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "circuit::boolean::tests::structure_digest_is_process_and_thread_deterministic",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .env("RAYON_NUM_THREADS", rayon_threads)
            .output()
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        stdout
            .split_once(MARKER)
            .map(|(_, digest)| digest.lines().next().unwrap().to_owned())
            .expect("child process did not print its structure digest")
    };

    assert_eq!(run("1"), run("4"));
    assert_eq!(run("2"), format!("{:?}", digest()));
}

#[test]
#[should_panic(expected = "expression belongs to another circuit builder")]
fn cross_builder_handles_fail_closed() {
    let mut first = CircuitBuilder::new();
    let bit = first.input();
    let mut second = CircuitBuilder::new();
    second.materialize(bit.expr());
}
