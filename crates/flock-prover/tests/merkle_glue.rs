//! Correctness of the two glue tables (`r1cs_hashes::merkle_glue`) that let a
//! Merkle opening be expressed as wiring over the shipped BLAKE3 table.
//!
//! The bar is the composite: `SwapTable` must reproduce, level by level, the
//! pair `MerkleTreeLayout`'s own fold hashes — otherwise the collapsed circuit
//! would compute a different root and every downstream test would be
//! measuring the wrong thing.

use std::array::from_fn;

use flock_core::{
    field::F128,
    lincheck::LincheckCircuit,
    pcs::{
        PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    r1cs::BlockR1cs,
    schedule::{IoDirection, IoWord},
    scratch::{give_f128, take_f128},
    test_rng::Rng,
    union::SlotWitnessDest,
};
use flock_hash::BLAKE3_IV;
use flock_hash::blake3_compress;
use flock_prover::{
    challenger::FsChallenger,
    prover::{UnionSlotProverInput, prove_fast_ligerito_union},
    r1cs_hashes::{
        blake3::build_matrices,
        merkle_glue::{
            BitSpreadInput, BitSpreadTable, FamilyTransposeTileInput, FamilyTransposeTileTable,
            PowMaskInput, PowMaskTable, SwapInput, SwapTable,
        },
        merkle_r1cs::{
            BLAKE3_FLAG_CHUNK_END, BLAKE3_FLAG_CHUNK_START, BLAKE3_FLAG_PARENT, ChunkPathInput,
            MerkleTreeLayout, NODE_BLOCK_LEN, NODE_COUNTER, SLOT_WORDS, blake3_spec,
        },
    },
    schedule::{Registry, TableType},
    union::UnionInstance,
    verifier::verify_ligerito_union,
};
/// One compression's output chaining value.
fn chaining_value(h: &[u32; 8], m: &[u32; 16], flags: u32) -> [u32; SLOT_WORDS] {
    let out = blake3_compress(h, m, NODE_COUNTER, NODE_BLOCK_LEN, flags);
    out[..SLOT_WORDS].try_into().unwrap()
}

/// Site-specific draws kept verbatim from this file's former local `Rng`.
trait RngExt {
    fn digest(&mut self) -> [u32; SLOT_WORDS];
    fn word(&mut self) -> u128;
}
impl RngExt for Rng {
    fn digest(&mut self) -> [u32; SLOT_WORDS] {
        from_fn(|_| self.next_u32())
    }
    fn word(&mut self) -> u128 {
        (0..4).fold(0u128, |a, _| (a << 32) | self.next_u32() as u128)
    }
}

fn nnz(r: &BlockR1cs) -> usize {
    r.a_0.rows.iter().map(|r| r.len()).sum::<usize>()
        + r.b_0.rows.iter().map(|r| r.len()).sum::<usize>()
}

/// The relation accepts honest swaps at both polarities, and every column it
/// exports is load-bearing.
#[test]
fn swap_relation_is_honest_and_tight() {
    let r1cs = SwapTable::build_block_r1cs(0);
    let mut rng = Rng(0x_5A_4B_51_A9);

    for trial in 0..8 {
        let bit_word = rng.word();
        let input = SwapInput {
            bit_word,
            prev: rng.digest(),
            sib: rng.digest(),
        };
        let [z, a, b] = SwapTable::build_witness(&input);
        assert!(r1cs.satisfies(&z), "honest swap rejected (trial {trial})");
        assert_eq!(a, r1cs.apply_a(&z), "emitted a (trial {trial})");
        assert_eq!(b, r1cs.apply_b(&z), "emitted b (trial {trial})");

        // The committed left/right ARE the pair the fold hashes.
        let (left, right) = SwapTable::outputs(&input);
        for j in 0..256 {
            assert_eq!(
                z[SwapTable::LEFT + j],
                (left[j / 32] >> (j % 32)) & 1 == 1,
                "left bit {j}"
            );
            assert_eq!(
                z[SwapTable::RIGHT + j],
                (right[j / 32] >> (j % 32)) & 1 == 1,
                "right bit {j}"
            );
        }

        // Every exported column is constrained.
        for (what, col) in [
            ("bit", SwapTable::BIT),
            ("prev", SwapTable::PREV + 11),
            ("sibling", SwapTable::SIB + 200),
            ("t", SwapTable::T + 5),
            ("left", SwapTable::LEFT + 3),
            ("right", SwapTable::RIGHT + 250),
            ("const", SwapTable::CONST),
        ] {
            let mut bad = z.clone();
            bad[col] ^= true;
            assert!(
                !r1cs.satisfies(&bad),
                "flipping {what} (column {col}) was accepted"
            );
        }
    }

    // The bit-word's high bits are FREE — that is what lets a Fiat-Shamir
    // challenge word be wired straight in, exactly as the composite's index
    // word does.
    let base = SwapInput {
        bit_word: 0,
        prev: rng.digest(),
        sib: rng.digest(),
    };
    let dressed = SwapInput {
        bit_word: !1u128,
        ..base.clone()
    };
    let [z0, ..] = SwapTable::build_witness(&base);
    let [z1, ..] = SwapTable::build_witness(&dressed);
    assert!(r1cs.satisfies(&z1), "a nonzero high half was rejected");
    assert_eq!(
        &z0[SwapTable::PREV..],
        &z1[SwapTable::PREV..],
        "the high bits reached the fold"
    );
}

/// **The bar**: level by level, the swap's outputs are the pair the composite
/// hashes. If these ever disagree the collapsed circuit computes a different
/// root than `reference_root_chunk`, and nothing downstream would notice.
#[test]
fn swap_matches_the_composite_fold() {
    for (depth, leaf_bytes) in [(2usize, 128usize), (8, 128), (14, 1024)] {
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let mut rng = Rng(0x_C0_11_4B_5E ^ depth as u64);
        for _ in 0..4 {
            let hi = rng.word() >> depth;
            let index = (hi << depth) | (rng.word() & ((1u128 << depth) - 1));
            let input = ChunkPathInput {
                leaf_data: (0..leaf_bytes).map(|_| rng.next_u32() as u8).collect(),
                index,
                siblings: (0..depth).map(|_| rng.digest()).collect(),
            };
            // Walk the fold INDEPENDENTLY: chunk chain, then one compression
            // per level over whatever `SwapTable` says the pair is. Nothing
            // here re-derives the swap rule, so agreeing with `root_chunk` is
            // a real check rather than a restatement.
            let blocks = leaf_bytes / 64;
            let mut prev = BLAKE3_IV;
            for i in 0..blocks {
                let m: [u32; 16] = from_fn(|w| {
                    let o = i * 64 + 4 * w;
                    u32::from_le_bytes(input.leaf_data[o..o + 4].try_into().unwrap())
                });
                let mut f = 0;
                if i == 0 {
                    f |= BLAKE3_FLAG_CHUNK_START;
                }
                if i + 1 == blocks {
                    f |= BLAKE3_FLAG_CHUNK_END;
                }
                prev = chaining_value(&prev, &m, f);
            }
            for (l, &sib) in input.siblings.iter().enumerate() {
                let (left, right) = SwapTable::outputs(&SwapInput {
                    bit_word: index >> l,
                    prev,
                    sib,
                });
                let mut m = [0u32; 16];
                m[..SLOT_WORDS].copy_from_slice(&left);
                m[SLOT_WORDS..].copy_from_slice(&right);
                prev = chaining_value(&BLAKE3_IV, &m, BLAKE3_FLAG_PARENT);
            }
            assert_eq!(
                prev,
                layout.root_chunk(&input),
                "depth {depth}: walking the swap reproduces the composite's root"
            );
        }
    }
}

/// The spread relocates bit `l` to position 0 of output `l`, leaves the rest
/// zero, and the outputs are constrained.
#[test]
fn bit_spread_relocates_and_is_tight() {
    for depth in [1usize, 8, 14] {
        let ty = BitSpreadTable::new(depth);
        let r1cs = ty.build_block_r1cs(0);
        let mut rng = Rng(0x_B1_75_9E ^ depth as u64);

        for _ in 0..4 {
            let idx = rng.word();
            let [z, a, b] = ty.build_witness(idx);
            assert!(r1cs.satisfies(&z), "depth {depth}: honest spread rejected");
            assert_eq!(a, r1cs.apply_a(&z), "depth {depth} emitted a");
            assert_eq!(b, r1cs.apply_b(&z), "depth {depth} emitted b");

            for l in 0..depth {
                assert_eq!(z[ty.out(l)], (idx >> l) & 1 == 1, "depth {depth} out {l}");
                assert!(
                    z[ty.out(l) + 1..ty.out(l) + 128].iter().all(|&v| !v),
                    "depth {depth} out {l} has nonzero high bits"
                );
            }
            // Flipping an output, or a high bit that must stay zero, breaks it.
            for (what, col) in [("out bit", ty.out(depth - 1)), ("out pad", ty.out(0) + 7)] {
                let mut bad = z.clone();
                bad[col] ^= true;
                assert!(!r1cs.satisfies(&bad), "depth {depth}: {what} was accepted");
            }
        }
    }
}

/// The optional mask is a constraint, not an unchecked witness annotation:
/// selected zero bits pass, while any selected one bit makes the row fail.
#[test]
fn bit_spread_zero_mask_is_enforced() {
    let ty = BitSpreadTable::new(8);
    let r1cs = ty.build_block_r1cs(0);
    let word = 0xA55A_0F0Fu128;

    let allowed = BitSpreadInput {
        word,
        zero_mask: !word,
        position_mask: 0,
        position_prefix: 0,
    };
    let [z, a, b] = ty.build_masked_witness(allowed);
    assert!(r1cs.satisfies(&z), "a disjoint zero mask must pass");
    assert_eq!(a, r1cs.apply_a(&z));
    assert_eq!(b, r1cs.apply_b(&z));

    // Manual on purpose: the std method needs 1.97+, above the Blackwell
    // CI runner's toolchain (see flock_core::bits::lowest_one).
    #[allow(clippy::manual_isolate_lowest_one)]
    let forbidden_bit = word & word.wrapping_neg();
    let rejected = BitSpreadInput {
        word,
        zero_mask: forbidden_bit,
        position_mask: 0,
        position_prefix: 0,
    };
    let [z, a, b] = ty.build_masked_witness(rejected);
    assert!(!r1cs.satisfies(&z), "a masked one bit must be rejected");
    assert_eq!(a, r1cs.apply_a(&z));
    assert_eq!(b, r1cs.apply_b(&z));

    // Under C=I an attacker can otherwise repair the failing row by writing
    // the product into its private check cell. The check word must therefore
    // be a circuit input, which production wiring binds to zero.
    let mut repaired = z.clone();
    repaired[ty.check_pos() + forbidden_bit.trailing_zeros() as usize] = true;
    assert!(
        r1cs.satisfies(&repaired),
        "the regression probe must model the C=I repair attack"
    );
    assert!(
        ty.io_schema()
            .contains(&IoWord::input(ty.check_pos() / 128)),
        "the load-bearing check word must be a circuit input"
    );
}

#[test]
fn bit_spread_derives_stratified_position() {
    let ty = BitSpreadTable::new(14);
    let word = 0xDEAD_BEEF_1234_5678u128;
    let position_mask = (1u128 << 9) - 1;
    let position_prefix = 5u128 << 9;
    let input = BitSpreadInput {
        word,
        zero_mask: 0,
        position_mask,
        position_prefix,
    };
    let [z, a, b] = ty.build_masked_witness(input);
    let r1cs = ty.build_block_r1cs(0);
    assert!(r1cs.satisfies(&z));
    assert_eq!(a, r1cs.apply_a(&z));
    assert_eq!(b, r1cs.apply_b(&z));
    let got = (0..128).fold(0u128, |acc, j| {
        acc | ((z[ty.position_pos() + j] as u128) << j)
    });
    assert_eq!(got, (word & position_mask) ^ position_prefix);

    let mut bad = z;
    bad[ty.position_pos()] ^= true;
    assert!(!r1cs.satisfies(&bad));
}

/// The fused PoW row has the same C=I shape as the optional BitSpread mask:
/// its prefix products are sound only when word 3 is an input wired to
/// the fixed 0..0,1 word by the recursive verifier circuit.
#[test]
fn pow_mask_inputs_the_load_bearing_check_word() {
    let ty = PowMaskTable;
    let r1cs = ty.build_block_r1cs(0);
    let input = PowMaskInput {
        pred: 1,
        nonce: 0,
        mask: 1,
    };
    let [z, ..] = ty.build_witness(input);
    assert!(
        !r1cs.satisfies(&z),
        "an honest bad-prefix witness is rejected"
    );

    let mut repaired = z;
    repaired[384] = true;
    assert!(
        r1cs.satisfies(&repaired),
        "the regression probe must model the formerly accepted repair attack"
    );
    assert_eq!(ty.io_schema().last().map(|io| io.word_col), Some(3));
    assert_eq!(
        ty.io_schema().last().map(|io| io.dir),
        Some(IoDirection::In)
    );

    let check_word = (384..512).fold(0u128, |acc, i| acc | ((repaired[i] as u128) << (i - 384)));
    assert_ne!(
        check_word,
        1u128 << 127,
        "the malicious repair differs from the constant word production wiring requires"
    );
}

/// The reason the glue exists: both tables are negligible against the BLAKE3
/// block whose duplication they remove.
#[test]
fn glue_is_negligible_against_blake3() {
    let b3 = {
        let (a, b) = build_matrices();
        a.rows.iter().map(|r| r.len()).sum::<usize>()
            + b.rows.iter().map(|r| r.len()).sum::<usize>()
    };
    let swap = nnz(&SwapTable::build_block_r1cs(0));
    let spread = nnz(&BitSpreadTable::new(14).build_block_r1cs(0));
    println!(
        "\nglue nnz: swap {swap} | bit-spread {spread} | blake3 {b3} \
         ({:.4}% of blake3)\n",
        100.0 * (swap + spread) as f64 / b3 as f64
    );
    assert!(
        swap + spread < b3 / 1000,
        "glue ({swap} + {spread}) is not negligible against blake3 ({b3})"
    );
}

/// Stale pool contents, as a recycled witness buffer would hold them.
const STALE: F128 = F128 {
    lo: 0xDEAD_BEEF_DEAD_BEEF,
    hi: 0xDEAD_BEEF_DEAD_BEEF,
};

/// Row counts around the 8-row group boundary at capacity `2^5`: empty, a
/// partial first group, exact multiples of 8, partial later groups, and full
/// capacity.
const DIRTY_DESTINATION_COUNTS: [usize; 10] = [0, 1, 4, 7, 8, 9, 15, 16, 31, 32];
const DIRTY_DESTINATION_NU: usize = 5;

/// Run an in-place glue driver over a destination full of stale words and
/// compare it against the copying driver's `canonical` output.
///
/// Without elision the whole block must match. With elision the driver may
/// leave all-dummy groups as it found them, but every row of a group that
/// holds a live row must match — the trailing partial group's dead lanes
/// included, because the RS zerocheck's round 1 reads whole 512-bit
/// sub-windows up to the last live row.
fn assert_in_place_matches_canonical_on_dirty_destination(
    count: usize,
    nu: usize,
    canonical: (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>),
    generate: impl Fn(SlotWitnessDest<'_>) -> Vec<u8>,
) {
    let (canonical_z, canonical_a, canonical_b, canonical_stripe) = canonical;
    let capacity = 1usize << nu;
    let columns = canonical_z.len() / capacity;
    let group_bytes = canonical_stripe.len() / (capacity / 8);
    let live_rows = count.next_multiple_of(8);
    for elide_padding_writes in [false, true] {
        for dead_padding_unread in [false, true] {
            let mode = format!(
                "count {count}, elide_padding_writes {elide_padding_writes}, \
                 dead_padding_unread {dead_padding_unread}"
            );
            let mut z = vec![STALE; canonical_z.len()];
            let mut a = z.clone();
            let mut b = z.clone();

            let stripe = generate(SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes,
                dead_padding_unread,
            });

            let written_rows = if elide_padding_writes {
                live_rows
            } else {
                capacity
            };
            for (name, actual, expected) in [
                ("z", &z, &canonical_z),
                ("a", &a, &canonical_a),
                ("b", &b, &canonical_b),
            ] {
                for column in 0..columns {
                    let start = column * capacity;
                    assert_eq!(
                        actual[start..start + written_rows],
                        expected[start..start + written_rows],
                        "{name} chunk-column {column} ({mode})"
                    );
                    // Elision still skips the all-dummy groups.
                    assert!(
                        actual[start + written_rows..start + capacity]
                            .iter()
                            .all(|word| *word == STALE),
                        "{name} chunk-column {column} wrote an all-dummy group ({mode})"
                    );
                }
            }
            let stripe_bytes = written_rows / 8 * group_bytes;
            assert_eq!(
                stripe[..stripe_bytes],
                canonical_stripe[..stripe_bytes],
                "stripe ({mode})"
            );
        }
    }
}

/// Every glue table's in-place driver leaves the rows a prover reads exactly
/// as the copying driver does, whatever the destination held before.
#[test]
fn in_place_drivers_zero_trailing_group_in_dirty_destination() {
    let nu = DIRTY_DESTINATION_NU;
    let mut rng = Rng(0x_D1_27_7A_11);
    let spread = BitSpreadTable::new(24);
    let pow_mask = PowMaskTable;
    for count in DIRTY_DESTINATION_COUNTS {
        let swap_rows: Vec<SwapInput> = (0..count)
            .map(|_| SwapInput {
                bit_word: rng.word(),
                prev: rng.digest(),
                sib: rng.digest(),
            })
            .collect();
        assert_in_place_matches_canonical_on_dirty_destination(
            count,
            nu,
            SwapTable::generate_witness_batch_major(&swap_rows, nu),
            |dst| SwapTable::generate_witness_batch_major_into(&swap_rows, dst),
        );

        let spread_rows: Vec<BitSpreadInput> = (0..count)
            .map(|_| BitSpreadInput {
                word: rng.word(),
                zero_mask: 0,
                position_mask: 0xff,
                position_prefix: 0x100,
            })
            .collect();
        assert_in_place_matches_canonical_on_dirty_destination(
            count,
            nu,
            spread.generate_witness_batch_major(&spread_rows, nu),
            |dst| spread.generate_witness_batch_major_into(&spread_rows, dst),
        );

        let pow_mask_rows: Vec<PowMaskInput> = (0..count)
            .map(|_| PowMaskInput {
                pred: rng.word() << 16,
                nonce: rng.word() >> 64,
                mask: 0xffff,
            })
            .collect();
        assert_in_place_matches_canonical_on_dirty_destination(
            count,
            nu,
            pow_mask.generate_witness_batch_major(&pow_mask_rows, nu),
            |dst| pow_mask.generate_witness_batch_major_into(&pow_mask_rows, dst),
        );

        let tile_rows: Vec<FamilyTransposeTileInput> = (0..count)
            .map(|_| FamilyTransposeTileInput {
                rows: from_fn(|_| {
                    let word = rng.word();
                    F128::new(word as u64, (word >> 64) as u64)
                }),
                selector: rng.next_u32() as u8,
            })
            .collect();
        assert_in_place_matches_canonical_on_dirty_destination(
            count,
            nu,
            FamilyTransposeTileTable::generate_witness_batch_major(&tile_rows, nu),
            |dst| FamilyTransposeTileTable::generate_witness_batch_major_into(&tile_rows, dst),
        );
    }
}

/// End to end, through the real scratch pool: a union proof whose glue slots
/// are generated in place over recycled, stale witness buffers is
/// byte-identical to the proof from prebuilt (copied) witnesses, and verifies.
///
/// Nothing is injected into the destinations — the pool is drained and then
/// handed stale buffers, exactly what a long-running prover's pool holds
/// after unrelated work, and the prover picks its own buffer mode. Covers a
/// partial trailing group, counts that are multiples of 8, empty slots, and
/// full capacity (the one shape that does not elide padding writes).
#[test]
fn in_place_glue_proof_ignores_stale_pool_buffers() {
    const DOMAIN: &[u8] = b"flock-glue-stale-pool-v0";
    let nu = 9usize;
    let capacity = 1usize << nu;
    let spread = BitSpreadTable::new(24);
    let spread_r1cs = spread.build_block_r1cs(nu);
    let swap_r1cs = SwapTable::build_block_r1cs(nu);
    // Slot order is capacity area descending: bit spread, then swap.
    let registry = Registry::new(
        vec![
            TableType::from_block_r1cs(&swap_r1cs),
            TableType::from_block_r1cs(&spread_r1cs),
        ],
        nu,
    );
    assert_eq!(registry.types()[0].k_log, spread.k_log());
    let spread_circuit = spread_r1cs.csc_lincheck_circuit();
    let swap_circuit = swap_r1cs.csc_lincheck_circuit();
    let circuits: [&dyn LincheckCircuit; 2] = [spread_circuit, swap_circuit];
    let mut rng = Rng(0x_57_A1_E0_01);

    for counts in [[4usize, 78], [1, 9], [8, 80], [0, 0], [capacity, capacity]] {
        let spread_rows: Vec<BitSpreadInput> = (0..counts[0])
            .map(|_| BitSpreadInput {
                word: rng.word(),
                zero_mask: 0,
                position_mask: 0xff,
                position_prefix: 0x100,
            })
            .collect();
        let swap_rows: Vec<SwapInput> = (0..counts[1])
            .map(|_| SwapInput {
                bit_word: rng.word(),
                prev: rng.digest(),
                sib: rng.digest(),
            })
            .collect();
        let union = UnionInstance::new(&registry, counts.to_vec());
        let log_batch_size = embedded_initial_k_or_default(union.dense_m(), LigeritoProfile::Fast);
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size,
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(log_batch_size),
            merkle_hash: Default::default(),
        };
        let prove = |slots: Vec<UnionSlotProverInput<'_>>| {
            let mut challenger = FsChallenger::new(DOMAIN);
            let (proof, commitment, _claim) =
                prove_fast_ligerito_union(&union, &pcs_params, slots, &mut challenger);
            (proof, commitment)
        };

        let (prebuilt_proof, prebuilt_commitment) = prove(vec![
            UnionSlotProverInput::new(
                spread.generate_witness_batch_major(&spread_rows, nu),
                spread_circuit,
            ),
            UnionSlotProverInput::new(
                SwapTable::generate_witness_batch_major(&swap_rows, nu),
                swap_circuit,
            ),
        ]);

        // Empty the pool of every buffer that could serve the witness
        // request, then stock it with stale ones.
        let len = union.packed_len();
        for _ in 0..64 {
            drop(take_f128(len));
        }
        for _ in 0..6 {
            give_f128(vec![STALE; len]);
        }
        // The destinations really arrive stale — otherwise this test would
        // pass without exercising anything.
        let arrived_stale = |dst: &SlotWitnessDest<'_>| {
            assert!(
                [&*dst.z, &*dst.a, &*dst.b]
                    .iter()
                    .all(|block| block.iter().all(|word| *word == STALE)),
                "the pool did not hand out the stale buffers at counts {counts:?}"
            );
        };
        let (in_place_proof, in_place_commitment) = prove(vec![
            UnionSlotProverInput::in_place(
                |dst| {
                    arrived_stale(&dst);
                    spread.generate_witness_batch_major_into(&spread_rows, dst)
                },
                spread_circuit,
            ),
            UnionSlotProverInput::in_place(
                |dst| {
                    arrived_stale(&dst);
                    SwapTable::generate_witness_batch_major_into(&swap_rows, dst)
                },
                swap_circuit,
            ),
        ]);

        let mut challenger = FsChallenger::new(DOMAIN);
        verify_ligerito_union(
            &union,
            &circuits,
            &in_place_commitment,
            &in_place_proof,
            &pcs_params,
            &mut challenger,
        )
        .unwrap_or_else(|e| panic!("in-place proof rejected at counts {counts:?}: {e:?}"));
        assert_eq!(
            prebuilt_commitment.cap, in_place_commitment.cap,
            "commitment root differs at counts {counts:?}"
        );
        assert!(
            prebuilt_proof == in_place_proof,
            "proof differs at counts {counts:?}"
        );
    }
}
