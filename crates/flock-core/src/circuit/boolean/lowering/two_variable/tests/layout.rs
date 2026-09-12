use super::*;
use crate::circuit::boolean::{ColumnRole, ColumnSchema, ColumnVisitor};

struct Schema;
impl ColumnSchema for Schema {
    type Cols<T> = ([T; 2], T);
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        (
            v.word_aligned("input", ColumnRole::Input, 4),
            v.bit("output", ColumnRole::Output),
        )
    }
}

#[test]
fn identity_lowering_keeps_alignment_and_checks_all_padding() {
    let compiled = CircuitBuilder::compile(Schema, |b, (input, output)| {
        b.constrain(input[0], input[1], b.zero());
        b.define_linear(*output, input[0]);
    });
    let source = compiled.circuit();
    let placement = source.layout().unwrap();
    let lowered = source.lower_identity_c().unwrap();
    assert_eq!(
        &lowered.layout().value_positions()[..source.value_count()],
        placement.value_positions()
    );
    let relation = lowered.to_block_r1cs(4, 0, 0).unwrap();
    assert!(relation.c0_is_identity());
    assert_eq!(relation.const_pin, Some(0));
    let logical = lowered.evaluate(&[true, false]).unwrap();
    let witness = physical(&lowered, &logical, 16);
    assert!(relation.satisfies(&witness));
    let plan = lowered.walk_plan().unwrap();
    assert_eq!(plan.forward(&[true, false], 4).unwrap().z, witness);
    for position in 0..16 {
        if !lowered.layout().value_positions().contains(&position) {
            let mut invalid = witness.clone();
            invalid[position] = true;
            assert!(!relation.satisfies(&invalid), "padding {position}");
        }
    }
}
