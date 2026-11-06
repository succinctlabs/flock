use std::time::Instant;

use flock_core::{
    circuit::{WiringError, builder::GateType},
    product_gkr::ProductGkrError,
    union::SlotWitnessDest,
};

use super::*;
use crate::leanxmss::reference::{
    XmssHash, chain_address, grind, layer_root, sign_keyless, verify,
};

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
/// the batched packer keeps the walk's bits on every gadget kind. With
/// padding writes elided, both write the same slot into zeroed buffers and
/// the same stripe on its live batches.
#[test]
fn every_slot_packs_as_the_walk() {
    use flock_core::union::SlotWitnessDest;
    let (statements, sigs) = signed(2, 5);
    let circuit = XmssCircuit::new(2);
    let witness = circuit.witness(&statements, &sigs);
    for (id, gate) in &circuit.gadgets {
        let rows = witness.rows::<BooleanGate>(*id);
        let len = 1usize << (circuit.nu() + gate.k_log() - 7);
        let pack = |batched: bool, elide: bool| {
            let fill = if elide {
                F128::ZERO
            } else {
                F128::new(0xDEAD, 0xBEEF)
            };
            let (mut z, mut a, mut b) = (vec![fill; len], vec![fill; len], vec![fill; len]);
            let dst = SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: elide,
                dead_padding_unread: false,
            };
            let stripe = if batched {
                gate.pack_into(rows, dst)
            } else {
                gate.pack_into_unbatched(rows, dst)
            };
            (z, a, b, stripe)
        };
        let want = pack(false, false);
        let what = format!("{}: {} rows", gate.name(), rows.len());
        assert!(pack(true, false) == want, "{what}");
        let live = rows.len().div_ceil(8) << gate.k_log();
        for batched in [false, true] {
            let got = pack(batched, true);
            assert!(
                (&got.0, &got.1, &got.2) == (&want.0, &want.1, &want.2)
                    && got.3[..live] == want.3[..live],
                "{what}, batched {batched}, padding writes elided"
            );
        }
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
    // At nu = 5 a word column is 4096 bits, less than the RS zerocheck's
    // round-1 window, so no slot drops its degree-one words
    // (`UnionInstance::zeroed_word_cols`); `four_signatures_drop_words`
    // covers them.
    assert_eq!(circuit.nu(), 5);
    let union = circuit.union();
    assert!((0..union.registry().num_boolean()).all(|t| union.zeroed_word_cols(t).is_empty()));
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
    // A statement list of the wrong length is a rejection, not a panic.
    for wrong in [
        &statements[..1],
        &[statements[0], statements[1], statements[0]][..],
    ] {
        assert!(
            matches!(
                circuit.verify(wrong, &proof),
                Err(FlockVerifyError::CircuitMismatch)
            ),
            "{} statements",
            wrong.len()
        );
    }
}

/// Four signatures plan `nu = 6`, the smallest slot height at which words
/// drop: the commit is full height, so the full slots drop their degree-one
/// words and the lincheck checks the linear ones. The proof verifies, and
/// fails for a changed message. On x86 this is the RS zerocheck's test of
/// dropped words inside a multi-slot layout.
#[test]
fn four_signatures_drop_words() {
    let (statements, sigs) = signed(4, 13);
    let circuit = XmssCircuit::new(4);
    assert_eq!(circuit.nu(), 6);
    let union = circuit.union();
    let full_drops = (0..union.registry().num_boolean())
        .filter(|&t| !union.zeroed_word_cols(t).is_empty())
        .count();
    assert!(
        full_drops > 0 && union.checks_linear_rows(),
        "{full_drops} slots drop"
    );
    let witness = circuit.witness(&statements, &sigs);
    let proof = circuit.prove(&witness);
    circuit
        .verify(&statements, &proof)
        .expect("honest proof verifies");
    let mut tampered = statements.clone();
    tampered[2].msg[5] ^= 1;
    assert!(circuit.verify(&tampered, &proof).is_err(), "message");
}

/// The recursion leaf at four signatures: the chained transcript, no
/// dropped words, every cell-slot in the wiring. It proves and verifies,
/// fails for a changed message, and its proofs and the default ones are not
/// interchangeable.
#[test]
fn recursion_leaf_proves_and_verifies() {
    let (statements, sigs) = signed(4, 17);
    let leaf = XmssCircuit::with_options(4, None, XmssOptions::RECURSION_LEAF);
    let circuit = &leaf.shape.circuit;
    assert!(circuit.public_columns().next().is_some() && circuit.live_slots_packed());
    let public = leaf.public(&statements);
    let digest = leaf.statement_digest(&statements);
    assert_eq!(
        public[public.len() - 2..],
        digest_words(&digest),
        "the statement digest is the last public pair"
    );
    assert!(public.len() < 64, "no signer word is public");
    let witness = leaf.witness(&statements, &sigs);
    let proof = leaf.prove(&witness);
    leaf.verify(&statements, &proof)
        .expect("honest proof verifies");
    let mut tampered = statements.clone();
    tampered[3].msg[9] ^= 1;
    assert!(leaf.verify(&tampered, &proof).is_err(), "message");

    let default = XmssCircuit::new(4);
    assert!(
        default.verify(&statements, &proof).is_err(),
        "leaf proof, default verifier"
    );
    let proof = default.prove(&default.witness(&statements, &sigs));
    assert!(
        leaf.verify(&statements, &proof).is_err(),
        "default proof, leaf verifier"
    );
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

/// The bus's counting argument (`docs/tuple-bus-design.md` §2.3) ties three
/// things together, and this pins each against the code that enforces it,
/// so changing one side alone fails here rather than silently:
/// - the codeword gadget accepts exactly the digit sum `3·58 − STEPS`, the
///   sum the step count assumes (`STEPS = V·CHAIN_END − TARGET_SUM`);
/// - every layer has exactly `STEPS` step rows, which the bus's layer tag
///   divides by;
/// - `A(0)`, the chain row's address input, has a zero step field, so the
///   row's XOR of the digit gives `A(d_c)` with step field `d_c`, the same
///   position a step's `+1 mod 4` counts.
#[test]
fn steps_balance_the_codeword_sum() {
    use crate::leanxmss::gadgets::{address_gate, codeword_gate};
    let sum = V * CHAIN_END - STEPS;
    let check = codeword_gate(3).unwrap();
    let accepts = |digits: &[u8; V]| {
        let d = digits
            .iter()
            .enumerate()
            .fold(0u128, |d, (i, &x)| d | (x as u128) << (2 * i));
        let digest = F128::new(d as u64, (d >> 64) as u64);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out = Vec::new();
            check.eval_walk(&[digest], &mut out);
        }))
        .is_ok()
    };
    // Digits summing to `total`: as many 3s as fit, then the remainder.
    let digits_summing_to = |total: usize| -> [u8; V] {
        let mut left = total;
        std::array::from_fn(|_| {
            let x = left.min(CHAIN_END);
            left -= x;
            x as u8
        })
    };
    for total in [sum - 1, sum, sum + 1] {
        assert_eq!(
            accepts(&digits_summing_to(total)),
            total == sum,
            "the codeword gadget on digit sum {total}"
        );
    }

    // Step rows per layer: the shape's count and the witness's rows.
    assert_eq!(GADGETS_PER_SIGNATURE[0], 2 * STEPS);
    let (statements, sigs) = signed(1, 3);
    let plans = signature_plan(&statements[0], &sigs[0]);
    assert!(plans.iter().all(|p| p.steps.len() == STEPS));

    // A(0): the address gadget's output for the chain constant keeps the
    // constant's (zero) step field, and XOR by the digit gives the reference
    // chain address at that position.
    let address = address_gate(3).unwrap();
    let e = sigs[0].idx;
    for layer in 0..2 {
        let e = if layer == 0 { e } else { e >> HEIGHT };
        let mut a0 = Vec::new();
        address.eval_walk(
            &[
                F128::new(e as u64, 0xFEED),
                word(&layer_address_constants(layer)[CHAIN]),
            ],
            &mut a0,
        );
        let a0 = a0[0];
        assert_eq!(
            a0.lo >> STEP_FIELD & 3,
            0,
            "layer {layer}: A(0)'s step field"
        );
        for d in 0..=CHAIN_END {
            let mut a = a0;
            a.lo ^= (d as u64) << STEP_FIELD;
            assert_eq!(
                a,
                word(&chain_address(layer, e, d)),
                "layer {layer}, A({d})"
            );
        }
    }
}

/// [`XmssCircuit::prove`] from rows the test supplies (`rows[i]` for
/// `circuit.gadgets[i]`) under the `public` words given, with each slot's
/// packed `z` and lincheck stripe passed through `tamper(i, z, stripe)`
/// after its packer runs: how a test commits a witness the online phase
/// would not produce. Only declared rows reach the commitment this way: the
/// union commits dummy rows as zeros whatever the buffer holds there
/// ([`UnionInstance::compact_witness`]), so a committed dummy row needs a
/// cheater's own prover (`full_height_dummy_row_input_word_is_rejected` in
/// `flock-core/tests/union_lincheck.rs` plays one at the PIOP level).
fn prove_rows(
    circuit: &XmssCircuit,
    public: &[F128],
    rows: &[Vec<Box<[F128]>>],
    tamper: &(dyn Fn(usize, &mut [F128], &mut [u8]) + Sync),
) -> XmssProof {
    let union = circuit.union();
    let pcs = XmssCircuit::pcs_params(&union);
    let mut by_slot: Vec<(usize, UnionSlotProverInput<'_>)> = circuit
        .gadgets
        .iter()
        .enumerate()
        .map(|(i, (id, gate))| {
            let rows = &rows[i][..];
            let compress = (gate.name() == compress::NAME).then_some(&circuit.compress);
            let drive = move |dst: SlotWitnessDest<'_>| {
                let SlotWitnessDest {
                    z,
                    a,
                    b,
                    elide_padding_writes,
                    dead_padding_unread,
                } = dst;
                let dst = SlotWitnessDest {
                    z: &mut *z,
                    a: &mut *a,
                    b: &mut *b,
                    elide_padding_writes,
                    dead_padding_unread,
                };
                let mut stripe = match compress {
                    Some(packer) => packer.pack_into(rows, dst),
                    None => gate.pack_into(rows, dst),
                };
                tamper(i, z, &mut stripe);
                stripe
            };
            (
                circuit.shape.registry_slot(*id),
                UnionSlotProverInput::in_place(drive, gate.r1cs().csc_lincheck_circuit()),
            )
        })
        .collect();
    by_slot.sort_by_key(|&(slot, _)| slot);
    let mut challenger = circuit.challenger();
    let (proof, commitment, _) = prove_circuit(
        &union,
        &circuit.shape.circuit,
        public,
        &pcs,
        by_slot.into_iter().map(|(_, input)| input).collect(),
        Vec::new(),
        &mut challenger,
    );
    XmssProof { proof, commitment }
}

/// Every slot's rows of `witness`, in `circuit.gadgets` order.
fn rows_of(circuit: &XmssCircuit, witness: &CircuitWitness) -> Vec<Vec<Box<[F128]>>> {
    circuit
        .gadgets
        .iter()
        .map(|(id, _)| witness.rows::<BooleanGate>(*id).to_vec())
        .collect()
}

/// Flip bit `bit` of row `row` in a packed slot: the word in `z` and the
/// stripe's byte, which the lincheck reads.
fn flip(z: &mut [F128], stripe: &mut [u8], nu: usize, k: usize, row: usize, bit: usize) {
    let w = &mut z[((bit / 128) << nu) + row];
    if bit % 128 < 64 {
        w.lo ^= 1 << (bit % 64);
    } else {
        w.hi ^= 1 << (bit % 64);
    }
    stripe[(row / 8) * k + bit] ^= 1 << (row % 8);
}

/// Signer `s`'s signature edited by `edit`, then finished as a signer would
/// finish it: both layers' `ρ` reground so each digest is a codeword again.
/// Returns the statement it now signs, whose root is the one the edited
/// signature reaches.
fn resigned(
    statements: &[Statement],
    sigs: &[Signature],
    s: usize,
    edit: impl FnOnce(&mut Signature),
) -> (Vec<Statement>, Vec<Signature>) {
    let h = XmssHash::Blake3;
    let (mut statements, mut sigs) = (statements.to_vec(), sigs.to_vec());
    let (st, sig) = (&mut statements[s], &mut sigs[s]);
    edit(sig);
    let pp = st.pk.pp;
    sig.layers[0].rho = grind(h, &pp, 0, sig.idx, &st.msg, &sig.layers[0].rho, true).0;
    let bottom = layer_root(h, &pp, 0, sig.idx, &st.msg, &sig.layers[0], &mut None).unwrap();
    let top = sig.idx >> HEIGHT;
    sig.layers[1].rho = grind(h, &pp, 1, top, &bottom, &sig.layers[1].rho, true).0;
    st.pk.root = layer_root(h, &pp, 1, top, &bottom, &sig.layers[1], &mut None).unwrap();
    (statements, sigs)
}

/// Signer 1's signature edited — the leaf index at bit 0 or 17, or a
/// bottom-layer Merkle sibling — and finished honestly ([`resigned`]), so
/// every gate is satisfiable and the rows reach a different root. Proved
/// under the original public words, the only broken constraint is the
/// public root's wire, and the wiring rejects. The online phase itself
/// refuses each edit made in place: the edited index's digests are not
/// codewords, and the edited path misses the root.
#[test]
fn edited_index_or_sibling_is_rejected() {
    let (statements, sigs) = signed(2, 29);
    let circuit = XmssCircuit::new(2);
    let public = circuit.public(&statements);
    let edits: [(&str, Box<dyn Fn(&mut Signature)>); 3] = [
        ("idx bit 0", Box::new(|sig| sig.idx ^= 1)),
        ("idx bit 17", Box::new(|sig| sig.idx ^= 1 << 17)),
        ("sibling", Box::new(|sig| sig.layers[0].path[5][3] ^= 0x10)),
    ];
    for (what, edit) in &edits {
        let (forged, forged_sigs) = resigned(&statements, &sigs, 1, edit);
        assert_ne!(forged[1].pk.root, statements[1].pk.root, "{what}");
        let witness = circuit.witness(&forged, &forged_sigs);
        let proof = prove_rows(
            &circuit,
            &public,
            &rows_of(&circuit, &witness),
            &|_, _, _| {},
        );
        let verdict = circuit.verify(&statements, &proof);
        assert!(
            matches!(
                verdict,
                Err(FlockVerifyError::Wiring(WiringError::Gkr(
                    ProductGkrError::ProductMismatch
                )))
            ),
            "{what}: {verdict:?}"
        );

        // The same edit without re-signing: the online phase refuses it.
        let mut edited = sigs.clone();
        edit(&mut edited[1]);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            circuit.witness(&statements, &edited)
        }));
        assert!(refused.is_err(), "{what}: the online phase took the edit");
    }
}

/// Steps traded between two signers' same layer: every hash is still
/// computed right, but each step row's tuple carries its own signer's
/// layer tag `S` (and `P`), so the moved rows answer nobody's chain.
#[test]
fn steps_swapped_between_signers_are_rejected() {
    let (statements, sigs) = signed(2, 31);
    let circuit = XmssCircuit::new(2);
    let witness = edited_witness(&circuit, &statements, &sigs, |inputs, values| {
        for (row0, row1) in [(4, 4), (20, 33)] {
            for w in 0..3 {
                let a = step_input(inputs, 0, 1, row0, w);
                let b = step_input(inputs, 1, 1, row1, w);
                values.swap(a, b);
            }
        }
    });
    let proof = circuit.prove(&witness);
    assert!(unbalanced(circuit.verify(&statements, &proof)));
}

/// A chain's first step row receiving its start at the wrong position: the
/// address's step field one above or below the digit. The chain row sends
/// `A(d_c)`, so the bus cannot balance.
#[test]
fn chain_start_off_by_one_is_rejected() {
    let (statements, sigs) = signed(2, 37);
    let circuit = XmssCircuit::new(2);
    for delta in [1u64, 3] {
        let witness = edited_witness(&circuit, &statements, &sigs, |inputs, values| {
            // Row 0 of a layer is the first chain's first step: its A word
            // holds A(d_c).
            let a = &mut values[step_input(inputs, 1, 0, 0, 0)];
            let d = a.lo >> STEP_FIELD & 3;
            a.lo ^= (d ^ ((d + delta) & 3)) << STEP_FIELD;
        });
        let proof = circuit.prove(&witness);
        assert!(
            unbalanced(circuit.verify(&statements, &proof)),
            "step field {}",
            if delta == 1 { "+1" } else { "-1" }
        );
    }
}

/// A full slot drops its degree-one words; the lincheck's `η` term checks
/// them. End to end (the default zerocheck route: AG on aarch64, RS
/// elsewhere): one bit of a dropped XOR output word, `next` in a compression
/// slot no bus port and no wire reads, flipped in the committed `z` and the
/// lincheck's stripe alike. The zerocheck does not read the word, the
/// wiring and the bus do not either, and the opening sees a consistent
/// witness; the lincheck rejects.
#[test]
fn dropped_word_tamper_is_rejected_end_to_end() {
    let (statements, sigs) = signed(4, 41);
    let circuit = XmssCircuit::new(4);
    assert_eq!(circuit.nu(), 6);
    let union = circuit.union();
    let ported: Vec<usize> = circuit
        .shape
        .circuit
        .bus()
        .unwrap()
        .ports
        .iter()
        .map(|p| p.ty)
        .collect();
    let (target, gate) = circuit
        .gadgets
        .iter()
        .enumerate()
        .find_map(|(i, (id, gate))| {
            let slot = circuit.shape.registry_slot(*id);
            (gate.name() == compress::NAME
                && !ported.contains(&slot)
                && !union.zeroed_word_cols(slot).is_empty())
            .then_some((i, gate))
        })
        .expect("a full compression slot off the bus");
    let next_col = gate.table().io_schema[NEXT].word_col;
    let slot = circuit.shape.registry_slot(circuit.gadgets[target].0);
    assert!(union.zeroed_word_cols(slot).contains(&next_col));
    let (nu, k) = (circuit.nu(), 1usize << gate.k_log());

    let witness = circuit.witness(&statements, &sigs);
    let rows = rows_of(&circuit, &witness);
    let public = circuit.public(&statements);
    let proof = prove_rows(&circuit, &public, &rows, &|_, _, _| {});
    circuit
        .verify(&statements, &proof)
        .expect("the untampered rows verify");
    let tamper = |i: usize, z: &mut [F128], stripe: &mut [u8]| {
        if i == target {
            flip(z, stripe, nu, k, 5, 128 * next_col + 7);
        }
    };
    let proof = prove_rows(&circuit, &public, &rows, &tamper);
    let verdict = circuit.verify(&statements, &proof);
    assert!(
        matches!(verdict, Err(FlockVerifyError::Lincheck(_))),
        "{verdict:?}"
    );
}
