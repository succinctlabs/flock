use super::*;

struct Aligned {
    width: usize,
    alignment: usize,
}

impl ColumnSchema for Aligned {
    type Cols<T> = Vec<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Vec<V::Value> {
        v.bits("input", ColumnRole::Input, self.width, self.alignment)
    }
}

#[test]
fn configured_width_and_alignment_survive_resolution() {
    let compiled = CircuitBuilder::compile(
        Aligned {
            width: 7,
            alignment: 8,
        },
        |_, _| {},
    );
    assert_eq!(compiled.schema()[0].alignment_bits, 8);
    let circuit = compiled.circuit();
    assert_eq!(circuit.column("input").unwrap().alignment_bits, 8);
    let layout = circuit.layout().unwrap();
    let inputs = compiled.inputs(|cols| {
        for bit in cols {
            *bit = true;
        }
    });
    let sparse = circuit.to_block_r1cs_with_layout(4, 0, 0, &layout).unwrap();
    let witness = circuit
        .evaluate_r1cs_with_layout(&inputs, 4, &layout)
        .unwrap();
    assert!(sparse.satisfies(&witness));
    assert_eq!(&witness[8..15], &[true; 7]);
}

#[test]
fn zero_width_invalid_alignment_and_changed_alignment_are_rejected() {
    for (width, alignment) in [(0, 1), (3, 0), (3, 3)] {
        assert!(
            std::panic::catch_unwind(|| CircuitBuilder::compile(
                Aligned { width, alignment },
                |_, _| {}
            ))
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| {
            let mut builder = CircuitBuilder::new();
            builder.reserve_columns(&Aligned {
                width: 4,
                alignment: 4,
            });
            builder.finish_columns(Aligned {
                width: 4,
                alignment: 8,
            });
        })
        .is_err()
    );
}
