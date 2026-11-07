use std::sync::Arc;

use flock_core::{
    circuit::builder::{CircuitShape, SlotId},
    lincheck::LincheckCircuit,
    pcs::Commitment,
    r1cs::{BlockR1cs, SparseBinaryMatrix},
};
use flock_transcript::transcript_record::{Stream, StreamWord, TranscriptOp, TranscriptOp as Op};
#[cfg(test)]
use {
    crate::r1cs_hashes::blake3::{build_block_r1cs, generate_witness_batch_major_partial},
    crate::tower::{
        Blake3Gate, DOMAIN, HashKind, PowMaskGate, PowMaskTable, gates_glue::PowMaskInput,
        pcs_batch_for,
    },
    crate::{
        challenger::FsChallenger,
        prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit},
        r1cs_hashes::fs_chain::CHAIN_SQUEEZE,
        union::UnionInstance,
        verifier::verify_ligerito_union_circuit,
    },
    flock_core::{circuit::builder::CircuitWitness, pcs::ligerito::LigeritoProfile},
    flock_hash::blake3_compress,
    flock_transcript::challenger::pow_squeeze_counter,
    std::array::from_fn,
    std::panic::AssertUnwindSafe,
    std::panic::catch_unwind,
};

use crate::{
    r1cs_hashes::fs_chain::FsChainTrace,
    tower::{
        CHUNK_END, CHUNK_START, CollapsedSlots, F128, MixedProof, PARENT, PcsParams, ShapeBuilder,
        Wire, cw, pack_params,
    },
};

/// BLAKE3 serializes its output words little-endian, while "leading bits"
/// means most-significant-bit first within each serialized byte. Return the
/// circuit-word mask whose set bits are exactly that prefix.
pub(super) fn pow_leading_zero_mask(bits: u32) -> F128 {
    assert!(bits <= 128, "the fused predicate is one F128 word");
    let mut mask = 0u128;
    for k in 0..bits as usize {
        let serialized_bit = 8 * (k / 8) + (7 - k % 8);
        mask |= 1u128 << serialized_bit;
    }
    F128::new(mask as u64, (mask >> 64) as u64)
}

/// Arithmetize every grinding operation in a recorded verifier transcript.
///
/// The fused BLAKE3 row has already bound the nonce to the transcript,
/// advanced the state and produced the protected challenge. This helper adds
/// only the selected-zero relations
///
/// ```text
/// prefix_bits(predicate_word, lambda) = 0^lambda
/// nonce[64..128] = 0.
/// ```
///
/// The selected-zero equations are rows of the shared PoW-mask table, whose
/// mask and check inputs are statement constants. A zero-bit operation instead
/// enforces the canonical nonce 0.
pub(super) fn emit_pow_checks(
    sb: &mut ShapeBuilder,
    _b3: SlotId,
    pow: SlotId,
    _iv: [Wire; 2],
    pows: &[([Wire; 2], u32)],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    let check_word = (!pows.is_empty()).then(|| cw(sb, vals, consts, F128::new(0, 1u64 << 63)));
    for &([predicate, nonce], bits) in pows {
        // One fused PowMask row per check: the prefix cells mask the
        // predicate and the mask word's wire-bound high half pins the
        // nonce to 64 bits — the transcript stream allocates a whole F128
        // word to the 8-byte nonce, and this is what keeps the remaining
        // eight bytes padding rather than an extra grinding knob for a
        // malicious recursive prover.
        assert!(
            bits <= 64,
            "the PowMask row's prefix cells cover the low mask half"
        );
        if bits == 0 {
            // Canonical zero nonce: the nonce rides BOTH input words — the
            // prefix cells pin its low half under the all-ones low mask,
            // the structural high-half cells pin the rest.  All 128 bits,
            // so a disabled site cannot become a grinding knob either.
            let ones = cw(sb, vals, consts, F128::new(u64::MAX, 0));
            let _ = sb.gate(
                pow,
                &[nonce, nonce, ones, check_word.expect("nonempty PoW list")],
            );
        } else {
            let mask_w = cw(sb, vals, consts, pow_leading_zero_mask(bits));
            let _ = sb.gate(
                pow,
                &[
                    predicate,
                    nonce,
                    mask_w,
                    check_word.expect("nonempty PoW list"),
                ],
            );
        }
    }
}

/// Locate fused PoW predicate and nonce wires on an arbitrary recorded tape
/// and constrain every native fused verification call in-circuit.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_recorded_pow_checks(
    sb: &mut ShapeBuilder,
    b3: SlotId,
    spread: SlotId,
    iv: [Wire; 2],
    ops: &[TranscriptOp],
    trace: &FsChainTrace,
    stream: &Stream,
    outs: &[Vec<Wire>],
    ww: &[Option<Wire>],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
) {
    let (mut fin, mut pay) = (0usize, 0usize);
    let mut pows = Vec::new();
    for op in ops {
        if let Op::Pow { bits } = op {
            pows.push((fin, pay, *bits));
        }
        if op.finalizes() {
            fin += 1;
        }
        if op.carries_payload() {
            pay += 1;
        }
    }
    let checks: Vec<([Wire; 2], u32)> = pows
        .into_iter()
        .map(|(fin, pay, bits)| {
            let sq = &trace.squeezes[fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, StreamWord::Bytes { payload, .. } if *payload == pay))
                .expect("pow nonce stream word");
            ([outs[sq[0]][1], ww[wi].expect("pow nonce wired")], bits)
        })
        .collect();
    emit_pow_checks(sb, b3, spread, iv, &checks, vals, consts);
}

#[test]
pub(super) fn fused_pow_masks_match_raw_compression() {
    let cv: [u32; 8] = from_fn(|i| 0x1020_3040u32.wrapping_mul(i as u32 + 1));
    for pending_words in 0..4 {
        let pending_len = 16 * pending_words;
        for nonce in 0..64u64 {
            for bits in [1u32, 2, 7, 8, 9, 13, 16, 17, 31, 64, 127, 128] {
                let mut block = [0u8; 64];
                for (i, b) in block[..pending_len].iter_mut().enumerate() {
                    *b = (17 * i + 9) as u8;
                }
                block[pending_len..pending_len + 8].copy_from_slice(&nonce.to_le_bytes());
                let message: [u32; 16] =
                    from_fn(|i| u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap()));
                let out = blake3_compress(
                    &cv,
                    &message,
                    pow_squeeze_counter(bits, pending_len + 16),
                    64,
                    CHAIN_SQUEEZE,
                );
                let mut predicate = [0u8; 16];
                for (i, word) in out[4..8].iter().enumerate() {
                    predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
                let predicate_word = u128::from_le_bytes(predicate);
                let mask = pow_leading_zero_mask(bits);
                let mask = (mask.lo as u128) | ((mask.hi as u128) << 64);
                let circuit_accepts = predicate_word & mask == 0;
                let native_accepts =
                    (0..bits as usize).all(|k| predicate[k / 8] & (1 << (7 - k % 8)) == 0);
                assert_eq!(
                    circuit_accepts, native_accepts,
                    "pending words {pending_words}, nonce {nonce}, lambda {bits}"
                );
            }
        }
    }

    // Pin the other half of the gadget on the fused PowMask row: a
    // nonzero-bit PoW permits only a 64-bit nonce, and a zero-bit site
    // permits only the canonical nonce zero.  The prefix checks live in the
    // R1CS; the nonce-width check is the mask input word's WIRE BINDING
    // (the word must equal the statement's mask constant, whose high half
    // is zero), so the pin models both.
    let ty = PowMaskTable;
    let r1cs = ty.build_block_r1cs(0);
    let accepted = |pred: u128, nonce: u128, mask: u128| {
        let [z, _, _] = ty.build_witness(PowMaskInput { pred, nonce, mask });
        let word2 = (256..384).fold(0u128, |acc, i| acc | ((z[i] as u128) << (i - 256)));
        r1cs.satisfies(&z) && word2 == mask
    };
    // The nonce width, under a clearing predicate.
    assert!(accepted(0, 42, 0b11));
    assert!(!accepted(0, (1u128 << 100) | 42, 0b11));
    // The prefix itself.
    assert!(!accepted(0b10, 42, 0b11));
    // The canonical zero-bit shape: the nonce as both words, all-ones low mask.
    assert!(accepted(0, 0, u64::MAX as u128));
    assert!(!accepted(1, 1, u64::MAX as u128));
    assert!(!accepted(1u128 << 100, 1u128 << 100, u64::MAX as u128));
}

#[test]
pub(super) fn recursive_pow_relation_accepts_valid_and_rejects_invalid_nonce() {
    let bits = 6u32;
    let mut state_digest = [0u8; 32];
    for (i, b) in state_digest.iter_mut().enumerate() {
        *b = (29 * i + 3) as u8;
    }
    let cv: [u32; 8] =
        from_fn(|i| u32::from_le_bytes(state_digest[4 * i..4 * i + 4].try_into().unwrap()));
    let fused = |nonce: u64| {
        let mut block = [0u8; 64];
        block[..8].copy_from_slice(&nonce.to_le_bytes());
        let message: [u32; 16] =
            from_fn(|i| u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap()));
        blake3_compress(
            &cv,
            &message,
            pow_squeeze_counter(bits, 16),
            64,
            CHAIN_SQUEEZE,
        )
    };
    let accepts = |nonce: u64| {
        let out = fused(nonce);
        let mut predicate = [0u8; 16];
        for (i, word) in out[4..8].iter().enumerate() {
            predicate[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        predicate[0] & 0b1111_1100 == 0
    };
    let good = (0..u64::MAX)
        .find(|&n| accepts(n))
        .expect("a six-bit nonce exists");
    let bad = (good + 1..u64::MAX)
        .find(|&n| !accepts(n))
        .expect("a neighboring invalid nonce exists");

    let build = |nonce: u64, circuit_bits: u32| {
        // BLAKE3 has k_log=15; nu=7 places this focused union at the
        // smallest embedded security-config size m=22.
        let nu = 7usize;
        let mut sb = ShapeBuilder::new(nu);
        let b3 = sb.slot(Blake3Gate { nu });
        let spread = sb.slot(PowMaskGate { nu });
        let mut vals = Vec::new();
        let digest_v = [
            F128::new(
                u64::from_le_bytes(state_digest[..8].try_into().unwrap()),
                u64::from_le_bytes(state_digest[8..16].try_into().unwrap()),
            ),
            F128::new(
                u64::from_le_bytes(state_digest[16..24].try_into().unwrap()),
                u64::from_le_bytes(state_digest[24..].try_into().unwrap()),
            ),
        ];
        vals.extend_from_slice(&[digest_v[0], digest_v[1], F128::new(nonce, 0)]);
        let digest_w = [sb.input(), sb.input()];
        let nonce_w = sb.input();
        let mut consts = Vec::new();
        let zero = cw(&mut sb, &mut vals, &mut consts, F128::ZERO);
        let params = cw(
            &mut sb,
            &mut vals,
            &mut consts,
            pack_params(pow_squeeze_counter(circuit_bits, 16), 64, CHAIN_SQUEEZE),
        );
        let h = sb.gate(
            b3,
            &[digest_w[0], digest_w[1], nonce_w, zero, zero, zero, params],
        );
        emit_pow_checks(
            &mut sb,
            b3,
            spread,
            digest_w,
            &[([h[1], nonce_w], circuit_bits)],
            &mut vals,
            &mut consts,
        );
        let shape = sb.finish().expect("the focused PoW circuit builds");
        let built = shape.run(&vals, &[]);
        (nu, b3, spread, shape, built)
    };

    let (nu, good_b3, good_spread, good_shape, good_built) = build(good, bits);
    let (_, bad_b3, bad_spread, bad_shape, bad_built) = build(bad, bits);
    assert_eq!(good_shape.circuit.digest(), bad_shape.circuit.digest());
    let (_, _, _, downgraded_shape, downgraded_built) = build(0, 0);
    assert_ne!(
        good_shape.circuit.digest(),
        downgraded_shape.circuit.digest(),
        "changing the PoW difficulty changes digest-bound counter/mask constants"
    );
    assert!(!good_shape.circuit.check_public(&downgraded_built.public));
    assert!(
        good_built
            .rows::<PowMaskGate>(good_spread)
            .iter()
            .all(|r| r.pred & r.mask == 0 && r.nonce >> 64 == 0),
        "the valid witness satisfies every fused PoW row"
    );
    assert!(
        bad_built
            .rows::<PowMaskGate>(bad_spread)
            .iter()
            .any(|r| r.pred & r.mask != 0 || r.nonce >> 64 != 0),
        "the invalid nonce reaches a failing in-circuit prefix row"
    );

    let prove = |shape: &CircuitShape, built: &CircuitWitness, b3_slot, spread_slot| {
        let mut union = UnionInstance::new(&shape.registry, shape.counts.clone());
        union.set_dense_floor(22);
        let pcs = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: pcs_batch_for(&union, LigeritoProfile::Fast),
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(pcs_batch_for(&union, LigeritoProfile::Fast)),
            merkle_hash: HashKind::Blake3,
        };
        let b3_r1cs = build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let spread_ty = PowMaskTable;
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let mut slots = vec![
            (
                shape.registry_slot(b3_slot),
                UnionSlotProverInput::new(
                    generate_witness_batch_major_partial(built.rows::<Blake3Gate>(b3_slot), nu),
                    b3_lc,
                ),
            ),
            (
                shape.registry_slot(spread_slot),
                UnionSlotProverInput::new(
                    spread_ty
                        .generate_witness_batch_major(built.rows::<PowMaskGate>(spread_slot), nu),
                    spread_lc,
                ),
            ),
        ];
        slots.sort_by_key(|(i, _)| *i);
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs,
            slots.into_iter().map(|(_, s)| s).collect(),
            Vec::new(),
            &mut ch,
        );
        let mut lcs = vec![
            (shape.registry_slot(b3_slot), b3_lc),
            (shape.registry_slot(spread_slot), spread_lc),
        ];
        lcs.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn LincheckCircuit> = lcs
            .into_iter()
            .map(|(_, lc)| lc as &dyn LincheckCircuit)
            .collect();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs,
            &mut ch,
        )
    };

    prove(&good_shape, &good_built, good_b3, good_spread)
        .expect("a valid grinding witness proves and verifies");
    let bad_result = catch_unwind(AssertUnwindSafe(|| {
        prove(&bad_shape, &bad_built, bad_b3, bad_spread)
    }));
    assert!(
        match bad_result {
            Ok(result) => result.is_err(),
            Err(_) => true,
        },
        "an invalid grinding witness must not yield an accepted recursive proof"
    );
}

/// Emit one Merkle opening as rows of the shipped BLAKE3 table plus glue,
/// wired together. Returns the two words of the root.
///
/// This is what replaces a composite row. The dataflow, per level:
///
/// ```text
///   index word ─▶ BitSpread ─bit_l─▶ Swap ─left‖right─▶ BLAKE3 ─out_lo─▶ next Swap.prev
/// ```
///
/// and before it, the chunk chain: `leaf_blocks` BLAKE3 rows whose out_lo
/// threads row to row, seeded by the IV, `CHUNK_START` on the first and
/// `CHUNK_END` on the last. Every arrow is a copy constraint on whole words.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_opening(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    leaf_w: &[Wire],
    index_w: Wire,
    depth: usize,
    cap_depth: usize,
    position_prefix: usize,
    mut consts: Option<&mut Vec<(F128, Wire)>>,
    pubs: &mut Vec<F128>,
) -> ([Wire; 2], Wire) {
    // A leaf need NOT be a whole number of 64-byte blocks: a mixed circuit
    // union commits `num_lanes` ACTIVE lanes (`dense_words.div_ceil(2^log_dim)`
    // — an arbitrary integer, since the top lanes are definitionally zero and
    // never encoded), so a row can be e.g. 61 words = 976 bytes. BLAKE3 hashes
    // that as 16 blocks whose last carries b = 16 bytes with the rest of the
    // message zero — and the compression's `b` is already a free input here,
    // so the partial block costs one zero-padding wire, not a wire-format
    // change. `blocks` counts up to a chunk's 16; larger leaves would need
    // real chunk merging, which nothing here produces.
    assert!(!leaf_w.is_empty(), "a leaf has data");
    let blocks = leaf_w.len().div_ceil(4);
    assert!(blocks <= 16, "a leaf is one BLAKE3 chunk (<= 1024 bytes)");
    let mut shared = |sb: &mut ShapeBuilder, pubs: &mut Vec<F128>, v: F128| -> Wire {
        match consts.as_deref_mut() {
            Some(c) => cw(sb, pubs, c, v),
            None => {
                pubs.push(v);
                sb.public_input()
            }
        }
    };
    let zero_w = shared(sb, pubs, F128::ZERO);
    let pad_w = if leaf_w.len().is_multiple_of(4) {
        None
    } else {
        Some(zero_w)
    };

    // The index word's bits, one per level.
    // Its zero mask is empty: this row only relocates bits.  Grinding rows
    // below reuse the same table with nonzero masks to enforce predicates.
    let position_bits = depth - cap_depth;
    let position_mask = if position_bits == 128 {
        u128::MAX
    } else {
        (1u128 << position_bits) - 1
    };
    assert_eq!(
        position_prefix & position_mask as usize,
        0,
        "the fixed stratum and sampled low bits must be disjoint"
    );
    let mask_w = shared(
        sb,
        pubs,
        F128::new(position_mask as u64, (position_mask >> 64) as u64),
    );
    let prefix_w = shared(sb, pubs, F128::new(position_prefix as u64, 0));
    let spread = sb.gate(s.spread, &[index_w, zero_w, zero_w, mask_w, prefix_w]);
    let bits = &spread[..spread.len() - 1];
    let position_w = *spread.last().expect("spread emits the derived position");

    // Chunk chain: the leaf hashed as a BLAKE3 chunk.
    let mut cv = iv;
    for i in 0..blocks {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == blocks {
            flags |= CHUNK_END;
        }
        // The final block carries only the bytes that remain.
        let words = (leaf_w.len() - 4 * i).min(4);
        let params = shared(sb, pubs, pack_params(0, 16 * words as u32, flags));
        let mw = |j: usize| -> Wire {
            if j < words {
                leaf_w[4 * i + j]
            } else {
                pad_w.expect("a short block needs the zero pad")
            }
        };
        let out = sb.gate(s.b3, &[cv[0], cv[1], mw(0), mw(1), mw(2), mw(3), params]);
        cv = [out[0], out[1]];
    }

    // Node levels: swap, then a PARENT compression over the swapped pair.
    // The sibling is the swap's hint, supplied at `run` time in this call
    // order — setup has no values. Under capping the fold stops `cap_depth`
    // levels below the root: the returned digest is the depth-`cap_depth`
    // ancestor, which the CHECKER compares against the absorbed cap layer
    // (the boundary select — the circuit never touches the cap words).
    // `cap_depth = 0` is the uncapped statement, terminal = root.
    for l in 0..(depth - cap_depth) {
        let sw = sb.gate_hinted(s.swap, &[bits[l], cv[0], cv[1]]);
        let params = shared(sb, pubs, pack_params(0, 64, PARENT));
        let out = sb.gate(s.b3, &[iv[0], iv[1], sw[0], sw[1], sw[2], sw[3], params]);
        cv = [out[0], out[1]];
    }
    (cv, position_w)
}

/// The leaf outer's artifacts (built inside [`build_first_level_node_k`] and
/// [`build_node_outer_app`]) so the
/// recursion swap can consume the proof as ITS inner: the circuit shape
/// (owning registry + counts — `UnionInstance::new(&shape.registry,
/// shape.counts.clone())` reconstructs the instance), the public segment,
/// the BLAKE3/BLAKE3 circuit proof, and the boolean tables whose lincheck
/// circuits a verifier needs (in registry order via the `*_slot` indices).
pub struct LeafOuter {
    pub(super) shape: Arc<CircuitShape>,
    pub(super) public: Vec<F128>,
    pub(super) proof: MixedProof,
    pub(super) commitment: Commitment,
    pub(super) pcs: PcsParams,
    /// The BLAKE3 block, shared process-wide (`chain_blake_r1cs`): every
    /// node at one envelope reads ONE block and ONE compressed lincheck
    /// circuit instead of building and keeping its own ~170 MiB copy.
    pub(super) b3_r1cs: Arc<BlockR1cs>,
    pub(super) swap_r1cs: BlockR1cs,
    pub(super) spread_r1cs: BlockR1cs,
    pub(super) pow_r1cs: BlockR1cs,
    pub(super) family_r1cs: BlockR1cs,
    pub(super) b3_slots: Vec<usize>,
    pub(super) swap_slot: usize,
    pub(super) spread_slot: usize,
    pub(super) pow_slot: usize,
    pub(super) family_slot: usize,
    /// The tower's affine slot layouts (the envelope's), carried so a
    /// parent declares the same list as this child.
    pub(super) affine_env: crate::tower::affine_walk::AffineEnvLayouts,
}

pub(super) fn leaf_boolean_lcs(lo: &LeafOuter) -> Vec<&dyn LincheckCircuit> {
    let mut ordered: Vec<(usize, &dyn LincheckCircuit)> = vec![
        (lo.swap_slot, lo.swap_r1cs.csc_lincheck_circuit()),
        (lo.spread_slot, lo.spread_r1cs.csc_lincheck_circuit()),
        (lo.pow_slot, lo.pow_r1cs.csc_lincheck_circuit()),
        (lo.family_slot, lo.family_r1cs.csc_lincheck_circuit()),
    ];
    ordered.extend(lo.b3_slots.iter().map(|&slot| {
        (
            slot,
            lo.b3_r1cs.csc_lincheck_circuit() as &dyn LincheckCircuit,
        )
    }));
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, circuit)| circuit).collect()
}

pub(super) fn leaf_boolean_mats(lo: &LeafOuter) -> Vec<(&SparseBinaryMatrix, &SparseBinaryMatrix)> {
    let mut ordered = vec![
        (lo.swap_slot, (&lo.swap_r1cs.a_0, &lo.swap_r1cs.b_0)),
        (lo.spread_slot, (&lo.spread_r1cs.a_0, &lo.spread_r1cs.b_0)),
        (lo.pow_slot, (&lo.pow_r1cs.a_0, &lo.pow_r1cs.b_0)),
        (lo.family_slot, (&lo.family_r1cs.a_0, &lo.family_r1cs.b_0)),
    ];
    ordered.extend(
        lo.b3_slots
            .iter()
            .map(|&slot| (slot, (&lo.b3_r1cs.a_0, &lo.b3_r1cs.b_0))),
    );
    ordered.sort_by_key(|(slot, _)| *slot);
    ordered.into_iter().map(|(_, matrices)| matrices).collect()
}
