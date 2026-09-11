use super::*;
use crate::circuit::boolean::CircuitBuilder;

#[test]
fn execution_initializes_cancellation_boundaries_even_in_dirty_storage() {
    let mut b = CircuitBuilder::new();
    let [input] = b.input_bits("input");
    b.assert_zero(input);
    let source = b.finish();
    let lowered = source
        .lower_identity_c(&source.layout().finish().unwrap())
        .unwrap();
    let plan = lowered.walk_plan().unwrap();
    let expected = plan.forward(&[false], 3).unwrap();
    let mut dirty = ForwardTrace {
        z: vec![true; 8],
        a_z: vec![true; 8],
        b_z: vec![true; 8],
        c_z: vec![true; 8],
    };
    let mut temporaries = vec![true; plan.stats().max_live_temporaries];
    // Bypass prepare(): t initialization must be an execution rule, not incidental clearing.
    plan.execute_block(
        &[false],
        &mut dirty.z,
        &mut dirty.a_z,
        &mut dirty.b_z,
        &mut dirty.c_z,
        &mut temporaries,
    )
    .unwrap();
    for &position in lowered.layout().value_positions() {
        assert_eq!(dirty.z[position], expected.z[position]);
    }
    for &position in lowered.layout().row_positions() {
        assert_eq!(dirty.a_z[position], expected.a_z[position]);
        assert_eq!(dirty.b_z[position], expected.b_z[position]);
        assert_eq!(dirty.c_z[position], expected.c_z[position]);
    }
}
