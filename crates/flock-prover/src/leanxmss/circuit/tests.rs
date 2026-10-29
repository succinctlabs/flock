use std::time::Instant;

use super::*;
use crate::leanxmss::reference::{XmssHash, sign_keyless, verify};

/// `n` signers, each signing its own message with a key-free signature.
fn signed(n: usize, seed: u64) -> (Vec<Statement>, Vec<Signature>) {
    (0..n)
        .map(|s| {
            let pp: Value = std::array::from_fn(|j| (seed as usize + 31 * s + j) as u8);
            let msg: [u8; 32] = std::array::from_fn(|j| (7 * s + 3 * j + 1) as u8);
            let idx = (0x9E37_79B9u64.wrapping_mul(seed + s as u64 + 1) >> 7) as u32;
            let (pk, sig) = sign_keyless(XmssHash::Blake3, &pp, idx, &msg, seed + s as u64, true);
            assert_eq!(verify(XmssHash::Blake3, &pk, &msg, &sig, None), Ok(()));
            (Statement { pk, msg }, sig)
        })
        .unzip()
}

#[test]
fn compressions_per_signature() {
    assert_eq!((STEPS, LEAF_BLOCKS), (68, 15));
    assert_eq!(COMPRESSIONS_PER_SIGNATURE, 201);
}

/// 1000 signatures lay out in a 2^32 Boolean address space.
#[test]
fn layout_plan_for_1000_signatures() {
    let kinds: Vec<(usize, usize)> = gates(3)
        .iter()
        .zip(GADGETS_PER_SIGNATURE)
        .map(|(g, c)| (1000 * c, g.k_log()))
        .collect();
    let nu = plan_nu(&kinds);
    let space: u128 = kinds
        .iter()
        .map(|&(c, k)| c.div_ceil(1 << nu) as u128 * (1u128 << (nu + k)))
        .sum();
    eprintln!(
        "kinds {kinds:?}: nu = {nu}, space 2^{:.2}",
        (space as f64).log2()
    );
    assert!(space <= 1 << 32);
}

/// Every slot of a two-signature witness packs the same, batched or not:
/// the batched packer keeps the walk's bits on every gadget kind.
#[test]
fn every_slot_packs_as_the_walk() {
    use flock_core::union::SlotWitnessDest;
    let (statements, sigs) = signed(2, 5);
    let circuit = XmssCircuit::new(2);
    let witness = circuit.witness(&statements, &sigs);
    for (id, gate) in &circuit.gadgets {
        let rows = witness.rows::<BooleanGate>(*id);
        let len = 1usize << (circuit.nu() + gate.k_log() - 7);
        let pack = |batched: bool| {
            let poison = F128::new(0xDEAD, 0xBEEF);
            let (mut z, mut a, mut b) = (vec![poison; len], vec![poison; len], vec![poison; len]);
            let dst = SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
                dead_padding_unread: false,
            };
            let stripe = if batched {
                gate.pack_into(rows, dst)
            } else {
                gate.pack_into_unbatched(rows, dst)
            };
            (z, a, b, stripe)
        };
        assert!(
            pack(true) == pack(false),
            "{}: {} rows",
            gate.name(),
            rows.len()
        );
    }
}

/// Two signatures prove and verify; changing any signer's message, root or
/// `P` in the statement makes verification fail.
#[test]
fn two_signatures_prove_and_verify() {
    let (statements, sigs) = signed(2, 11);
    let t = Instant::now();
    let circuit = XmssCircuit::new(2);
    eprintln!(
        "shape: nu = {}, {:.0} ms",
        circuit.nu(),
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    let witness = circuit.witness(&statements, &sigs);
    eprintln!("online: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(witness.public, circuit.public(&statements));
    let compressions: usize = circuit
        .gadgets
        .iter()
        .filter(|(_, g)| g.name() == "leanxmss.compress")
        .map(|&(id, _)| witness.rows::<BooleanGate>(id).len())
        .sum();
    assert_eq!(compressions, 2 * COMPRESSIONS_PER_SIGNATURE);
    // The commit is full height, so only full slots drop their degree-one
    // words; at this size some do, so the proof exercises the linear check.
    let union = circuit.union();
    let full_drops = (0..union.registry().num_boolean())
        .filter(|&t| !union.zeroed_word_cols(t).is_empty())
        .count();
    eprintln!("slots dropping words: {full_drops}");
    assert!(full_drops > 0 && union.checks_linear_rows());
    let t = Instant::now();
    let proof = circuit.prove(&witness);
    eprintln!("prove: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    circuit
        .verify(&statements, &proof)
        .expect("honest proof verifies");

    let mut tampered = statements.clone();
    tampered[1].msg[0] ^= 1;
    assert!(circuit.verify(&tampered, &proof).is_err(), "message");
    let mut tampered = statements.clone();
    tampered[0].pk.root[3] ^= 0x40;
    assert!(circuit.verify(&tampered, &proof).is_err(), "root");
    let mut tampered = statements.clone();
    tampered[1].pk.pp[15] ^= 0x80;
    assert!(circuit.verify(&tampered, &proof).is_err(), "P");
    let swapped = vec![statements[1], statements[0]];
    assert!(circuit.verify(&swapped, &proof).is_err(), "signer order");
}

/// The online phase refuses a signature that does not verify: the bottom
/// root changes, so the top layer's digest is not a codeword.
#[test]
fn invalid_signature_is_refused() {
    let (statements, mut sigs) = signed(1, 5);
    sigs[0].layers[0].y[0][0] ^= 1;
    let circuit = XmssCircuit::new(1);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        circuit.witness(&statements, &sigs)
    }));
    assert!(refused.is_err());
}

/// The online phase with the resolved inputs edited by `edit` before the
/// gates run: a way to feed the step rows a schedule the honest prover
/// would not.
fn edited_witness(
    circuit: &XmssCircuit,
    statements: &[Statement],
    sigs: &[Signature],
    edit: impl FnOnce(&[Source], &mut [F128]),
) -> CircuitWitness {
    let plans: Vec<[LayerPlan; 2]> = statements
        .iter()
        .zip(sigs)
        .map(|(st, sig)| signature_plan(st, sig))
        .collect();
    let mut values: Vec<F128> = circuit
        .inputs
        .iter()
        .map(|&src| XmssCircuit::resolve(src, statements, Some((sigs, &plans))))
        .collect();
    edit(&circuit.inputs, &mut values);
    circuit
        .shape
        .run_filled_deferred(&circuit.plan, &values, &[])
}

/// Position of a step row's word among the circuit's inputs.
fn step_input(inputs: &[Source], s: usize, layer: usize, row: usize, w: usize) -> usize {
    inputs
        .iter()
        .position(
            |src| matches!(*src, Source::Step(a, b, c, d) if (a, b, c, d) == (s, layer, row, w)),
        )
        .expect("a step input")
}

/// The bus's products differ.
fn unbalanced(r: Result<(), FlockVerifyError>) -> bool {
    use flock_core::{circuit::WiringError, product_gkr::ProductGkrError};
    matches!(
        r,
        Err(FlockVerifyError::Wiring(WiringError::BusGkr(
            ProductGkrError::ProductMismatch
        )))
    )
}

/// The bus moves chain values by content: a layer's steps verify in any
/// order, but not with a wrong value in one step, and not with steps traded
/// between layers (each layer must spend exactly its own 68).
#[test]
fn the_bus_checks_the_chains() {
    let (statements, sigs) = signed(2, 23);
    let circuit = XmssCircuit::new(2);
    let proves = |witness: &CircuitWitness| {
        let proof = circuit.prove(witness);
        circuit.verify(&statements, &proof)
    };

    // Reversed step order within signature 1's bottom layer.
    let witness = edited_witness(&circuit, &statements, &sigs, |inputs, values| {
        for row in 0..STEPS / 2 {
            for w in 0..3 {
                let a = step_input(inputs, 1, 0, row, w);
                let b = step_input(inputs, 1, 0, STEPS - 1 - row, w);
                values.swap(a, b);
            }
        }
    });
    proves(&witness).expect("steps in any order verify");

    // One step hashes a different value.
    let witness = edited_witness(&circuit, &statements, &sigs, |inputs, values| {
        values[step_input(inputs, 0, 1, 17, 1)] += F128::ONE;
    });
    assert!(unbalanced(proves(&witness)), "a wrong chain value");

    // Signature 0's two layers trade one step row each: every step is
    // still computed correctly, but each layer's tuples now carry the other
    // layer's tag on one row.
    let witness = edited_witness(&circuit, &statements, &sigs, |inputs, values| {
        for w in 0..3 {
            let a = step_input(inputs, 0, 0, 3, w);
            let b = step_input(inputs, 0, 1, 40, w);
            values.swap(a, b);
        }
    });
    assert!(unbalanced(proves(&witness)), "steps traded between layers");
}

/// Every gate drops its degree-one words: the pin's word 0, the inputs,
/// and outputs defined by copies or XORs. Chain, address and address-step
/// compute nothing else, so the zerocheck reads none of their words.
#[test]
fn gates_drop_their_degree_one_words() {
    use flock_core::circuit::builder::GateType;
    let dropped: Vec<(String, Vec<usize>, usize)> = gates(3)
        .iter()
        .map(|g| {
            let table = g.table();
            (
                g.name().to_owned(),
                table.zeroed_word_cols,
                g.useful_bits().div_ceil(128),
            )
        })
        .collect();
    let all = |n: usize| (0..n).collect::<Vec<_>>();
    let compress: Vec<usize> = (0..8).chain(89..92).collect();
    let expected: [(&str, Vec<usize>, usize); 7] = [
        ("leanxmss.compress", compress.clone(), 92),
        ("leanxmss.compress", compress, 92),
        ("leanxmss.chain", all(7), 7),
        ("leanxmss.swap", vec![0, 1, 2, 3, 5, 6], 8),
        ("leanxmss.codeword", vec![0, 1], 14),
        ("leanxmss.address", all(4), 4),
        ("leanxmss.address-step", all(5), 5),
    ];
    for ((name, words, n), (e_name, e_words, e_n)) in dropped.iter().zip(&expected) {
        assert_eq!((name.as_str(), words, n), (*e_name, e_words, e_n));
    }
}
