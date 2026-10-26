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
    assert_eq!(STEP_SLOTS, [22, 34, 58]);
    assert_eq!(COMPRESSIONS_PER_SIGNATURE, 293);
}

/// 1000 signatures lay out with a 2^33 Boolean address space: BLAKE3 over
/// several slots, where one slot big enough for all of it would need 2^34.
#[test]
fn layout_plan_for_1000_signatures() {
    let gadget_k = [10, 10, 10, 10, 11, 11, 9, 10, 13, 12, 10];
    let mut kinds = vec![(1000 * COMPRESSIONS_PER_SIGNATURE, 14)];
    kinds.extend(
        GADGETS_PER_SIGNATURE
            .iter()
            .zip(gadget_k)
            .map(|(&c, k)| (1000 * c, k)),
    );
    let nu = plan_nu(&kinds);
    let space: u128 = kinds
        .iter()
        .map(|&(c, k)| c.div_ceil(1 << nu) as u128 * (1u128 << (nu + k)))
        .sum();
    eprintln!("nu = {nu}, space 2^{:.2}", (space as f64).log2());
    assert!(space <= 1 << 33);
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
        .blake3
        .iter()
        .map(|&id| witness.rows::<Blake3Gate>(id).len())
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

/// The online phase refuses a signature that does not verify: the codeword
/// gadget rejects the tampered layer's digest.
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

/// One layer's chain section (codeword check, selectors, routing, the three
/// steps) at the routing's limits: 22 chains at digit 0, filling every
/// step-0 slot; 34 at digit 1, filling every step-1 slot; 16 at 0 and 10 at 1;
/// and an ordinary codeword. Each chain end must match the native walk.
#[test]
fn chain_routing_at_its_limits() {
    use crate::leanxmss::{
        reference::{TARGET_SUM, chain_address, th},
        routing::{block_controls, route, sort_by_digit},
    };
    let patterns: [&[(u8, usize)]; 4] = [
        &[(0, 22), (1, 1), (3, 35)],
        &[(1, 34), (3, 24)],
        &[(0, 16), (1, 10), (3, 32)],
        &[(0, 9), (1, 12), (2, 17), (3, 20)],
    ];
    let mut state = 0x5eed_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for pattern in patterns {
        let mut digits: Vec<u8> = pattern
            .iter()
            .flat_map(|&(d, n)| std::iter::repeat_n(d, n))
            .collect();
        assert_eq!(digits.len(), V);
        assert_eq!(
            digits.iter().map(|&d| d as usize).sum::<usize>(),
            TARGET_SUM
        );
        for i in (1..V).rev() {
            digits.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let digest: Value = digits
            .iter()
            .enumerate()
            .fold(0u128, |d, (i, &x)| d | (x as u128) << (2 * i))
            .to_le_bytes();
        let pp: Value = std::array::from_fn(|_| next() as u8);
        let e = next() as u32;
        let y: Vec<Value> = (0..V)
            .map(|_| std::array::from_fn(|_| next() as u8))
            .collect();

        let nu = 7;
        let mut b = Builder {
            sb: ShapeBuilder::new(nu),
            inputs: Vec::new(),
            publics: Vec::new(),
        };
        let (mut slots, _) = declare_slots(&mut b, nu, &[1; CHAIN_END + 9]);
        let k = declare_constants(&mut b);
        let pp_w = b.constant(word(&pp));
        let digest_w = b.constant(word(&digest));
        let a_chain = std::array::from_fn(|step| b.constant(word(&chain_address(0, e, step))));
        let codeword_slot = slots.codeword.next();
        b.sb.gate(codeword_slot, &[digest_w]);
        let mut sel = Vec::new();
        let mut r = digest_w;
        for _ in 0..SPREADS {
            let slot = slots.spread.next();
            let out = b.sb.gate(slot, &[r]);
            sel.extend_from_slice(&out[..SPREAD_DIGITS]);
            r = out[SPREAD_DIGITS];
        }
        let chains: Vec<Wire> = y.iter().map(|v| b.constant(word(v))).collect();
        let network = route(&sort_by_digit(&digits));
        let ctrl: Vec<Vec<Wire>> = (0..GROUPS)
            .map(|g| {
                block_controls(&network, g)
                    .iter()
                    .map(|&c| b.constant(F128::new(c as u64, 0)))
                    .collect()
            })
            .collect();
        let ends = chain_ends(
            &mut b, &mut slots, &k, pp_w, digest_w, &a_chain, &sel, &chains, &ctrl,
        );
        for &end in &ends {
            b.sb.publish(end);
        }
        let values: Vec<F128> = b
            .inputs
            .iter()
            .map(|src| match src {
                Source::Const(v) => *v,
                _ => unreachable!(),
            })
            .collect();
        let shape = b.sb.finish().unwrap();
        let witness = shape.run_filled_deferred(&shape.fill_plan(), &values, &[]);
        let got = &witness.public[witness.public.len() - V..];
        for i in 0..V {
            let mut c = y[i];
            for step in digits[i] as usize..CHAIN_END {
                c = th(
                    XmssHash::Blake3,
                    &pp,
                    &chain_address(0, e, step),
                    &[&c, &le128(i)],
                );
            }
            assert_eq!(
                got[i],
                word(&c),
                "pattern {pattern:?}: chain {i} (digit {})",
                digits[i]
            );
        }
    }
}
