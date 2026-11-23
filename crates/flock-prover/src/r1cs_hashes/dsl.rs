//! Shared authoring operations for the two hash circuits.

use std::{collections::HashMap, sync::Arc};

use flock_core::circuit::boolean::{
    BooleanCircuit, BooleanGate, BooleanGateError, CircuitBuilder, ColumnRole, ColumnVisitor,
    LinearExpr, SchemaOperation, Var, WalkPlan,
};
use flock_field::test_rng::Rng;

mod addition;
#[cfg(test)]
pub(super) mod tests;
pub use addition::{Add3, Add4, Add32, AddConst32};

/// A DSL circuit as a gate ([`BooleanGate::new`]) with every hand-written
/// generator attached: each operation that has one (a BLAKE3 compression
/// built with [`super::blake3_dsl::compress_op`]) is packed natively rather
/// than walked. `operations` are the compiled circuit's
/// ([`flock_core::circuit::boolean::CompiledColumns::operations`]).
pub fn boolean_gate(
    name: &str,
    circuit: &BooleanCircuit,
    operations: &[SchemaOperation],
    inputs: &[&str],
    outputs: &[&str],
    nu: usize,
) -> Result<BooleanGate, BooleanGateError> {
    BooleanGate::new(name, circuit, inputs, outputs, nu)?.with_kernel(
        circuit,
        operations,
        super::blake3_dsl::OPERATION,
        Arc::new(super::blake3::Blake3Kernel),
    )
}

/// A 32-bit word of expressions, little-endian.
pub type Word = [LinearExpr; 32];

/// The walk of a DSL twin, moved onto a hand-written layout of the same
/// circuit, so the hand-written relation's lincheck and fold marginals walk
/// ([`flock_core::r1cs::BlockR1cs::with_walk`]).
///
/// A hand-written relation and its DSL twin compute the same values, in
/// other positions. `sample` draws one random input and returns it twice:
/// as the DSL's input words (little-endian bits, in the walk's input order)
/// and as the hand-written block's packed `(z, A·z, B·z)` (bit `p` of word
/// `p / 64`). 64 samples, one per lane, give each position a fingerprint:
/// its value and its row's two operands. The DSL side gets all 64 from one
/// bit-sliced forward walk. Values alone are not enough, since some are
/// always zero and some are equal, but the rows that define them differ.
///
/// Every position the walk occupies goes to the hand-written position with
/// its fingerprint. That position must be unique, the fingerprint nonzero,
/// and the map must reach every hand-written position whose fingerprint is
/// nonzero. A chance collision fails the uniqueness check rather than
/// mislabeling.
///
/// Matching fingerprints is not matching matrices. Each hash checks the
/// result against its `build_matrices` in a test, which pins every entry.
pub(crate) fn relabel_onto(
    plan: &WalkPlan,
    useful_bits: usize,
    mut sample: impl FnMut(&mut Rng) -> (Vec<u32>, [Vec<u64>; 3]),
) -> Result<WalkPlan, String> {
    let mut rng = Rng::new(0x5EED_DA7A_1AB3_1000);
    let mut inputs = Vec::new();
    let mut hand = [(); 3].map(|_| vec![0u64; useful_bits]);
    for lane in 0..64 {
        let (words, packed) = sample(&mut rng);
        inputs.resize(words.len() * 32, 0u64);
        for (input, bit) in inputs.iter_mut().zip(
            words
                .iter()
                .flat_map(|&word| (0..32).map(move |bit| u64::from(word >> bit & 1))),
        ) {
            *input |= bit << lane;
        }
        for (lanes, packed) in hand.iter_mut().zip(&packed) {
            for (p, slot) in lanes.iter_mut().enumerate() {
                *slot |= (packed[p / 64] >> (p % 64) & 1) << lane;
            }
        }
    }
    let k_log = plan.useful_bits().next_power_of_two().trailing_zeros() as usize;
    let (mut z, mut a, mut b, mut temporaries) = Default::default();
    plan.forward_sliced_into(
        &inputs,
        u64::MAX,
        k_log,
        &mut z,
        &mut a,
        &mut b,
        &mut temporaries,
    )
    .map_err(|e| e.to_string())?;
    let dsl = [z, a, b];
    let fingerprint = |zab: &[Vec<u64>; 3], p: usize| zab.each_ref().map(|lanes| lanes[p]);

    // Fingerprint -> its hand-written position, or usize::MAX when two share it.
    let mut by_print = HashMap::with_capacity(useful_bits);
    for position in 0..useful_bits {
        let print = fingerprint(&hand, position);
        if print != [0; 3] {
            by_print
                .entry(print)
                .and_modify(|p| *p = usize::MAX)
                .or_insert(position);
        }
    }
    let mut map = vec![usize::MAX; plan.useful_bits()];
    for &position in plan.occupied_positions() {
        let print = fingerprint(&dsl, position);
        map[position] = match by_print.get(&print) {
            _ if print == [0; 3] => {
                return Err(format!("DSL position {position} has a zero fingerprint"));
            }
            Some(&usize::MAX) => {
                return Err(format!("DSL position {position} matches several positions"));
            }
            Some(&target) => target,
            None => return Err(format!("DSL position {position} has no hand-written twin")),
        };
    }
    if by_print.len() != plan.occupied_positions().len() {
        return Err(format!(
            "{} hand-written positions are live, the walk occupies {}",
            by_print.len(),
            plan.occupied_positions().len()
        ));
    }
    plan.relabeled(&map, useful_bits).map_err(|e| e.to_string())
}

pub(super) fn physical_witness(
    logical: &[bool],
    layout: &flock_core::circuit::boolean::PhysicalLayout,
    capacity: usize,
) -> Vec<bool> {
    assert_eq!(logical.len(), layout.value_positions().len());
    let mut physical = vec![false; capacity];
    for (&bit, &position) in logical.iter().zip(layout.value_positions()) {
        physical[position] = bit;
    }
    physical
}

pub(super) fn words<V: ColumnVisitor, const N: usize>(
    v: &mut V,
    name: &str,
    role: ColumnRole,
    alignment: usize,
) -> [[V::Value; 32]; N] {
    let mut bits = v.bits(name, role, N * 32, alignment).into_iter();
    std::array::from_fn(|_| std::array::from_fn(|_| bits.next().unwrap()))
}

pub(super) fn materialize(b: &mut CircuitBuilder, cols: &[Var; 32], word: Word) -> Word {
    for (&col, expr) in cols.iter().zip(word) {
        b.define_linear(col, expr);
    }
    cols.map(Into::into)
}

pub(super) fn constant(b: &CircuitBuilder, word: u32) -> Word {
    std::array::from_fn(|bit| {
        if word >> bit & 1 != 0 {
            b.one().into()
        } else {
            b.zero()
        }
    })
}

pub(super) fn populate_words<const N: usize>(cols: [[&mut bool; 32]; N], values: &[u32; N]) {
    for (cols, value) in cols.into_iter().zip(values) {
        for (bit, col) in cols.into_iter().enumerate() {
            *col = value >> bit & 1 != 0;
        }
    }
}
