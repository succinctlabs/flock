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
    assert_eq!(ROWS, 83);
    assert_eq!(COMPRESSIONS_PER_SIGNATURE, 201);
}

/// 1000 signatures lay out in a 2^32 Boolean address space.
#[test]
fn layout_plan_for_1000_signatures() {
    let kinds: Vec<(usize, usize)> = gadget_constructors()
        .iter()
        .zip(GADGETS_PER_SIGNATURE)
        .map(|(f, c)| (1000 * c, f(3).k_log()))
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
        .filter(|(_, g)| g.name() == "leanxmss.machine")
        .map(|&(id, _)| witness.rows::<BooleanGate>(id).len())
        .sum();
    assert_eq!(compressions, 2 * COMPRESSIONS_PER_SIGNATURE);
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
