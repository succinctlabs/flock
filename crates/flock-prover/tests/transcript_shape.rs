//! The Fiat–Shamir transcript's SHAPE, recorded off real proofs.
//!
//! The recursive verifier replays this transcript inside a fixed-topology
//! circuit, so the circuit is generated from the schedule
//! [`RecordingChallenger`] observes while the *actual* verifier runs. Two
//! obligations follow, and this file is where they are discharged:
//!
//! 1. **The shape must not depend on data.** Same config, different counts and
//!    different witnesses must give the identical op sequence. If it ever does
//!    not, there is a second `sample_distinct_queries` hiding somewhere and no
//!    fixed-topology circuit exists until it is found. The failure names the
//!    op index rather than just asserting.
//!
//! 2. **Prover and verifier must agree.** They share one transcript by
//!    construction, so their recorded shapes must be equal — a free
//!    differential over the whole FS order.
//!
//! The pinned digest is the third guard: any protocol change that moves the FS
//! shape fails here loudly and gets a deliberate re-pin, the same discipline
//! the proof-byte fixtures use.
//!
//! Recording is done against **honest, accepted** proofs on purpose. The
//! verifier early-returns on rejection, so a rejected proof would yield a
//! silently truncated schedule — a circuit constraining a prefix of the
//! transcript would look perfectly healthy. Every case here asserts the verify
//! accepted before touching the shape.

use std::{env::var_os, sync::Arc};

use flock_core::{
    element_r1cs::{ElementTableBuilder, ElementTableType},
    field::F128,
    pcs::ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    schedule::TableClass,
    test_rng::Rng,
    transcript_record::{RecordingChallenger, TranscriptOp, TranscriptShape},
};
use flock_prover::{
    challenger::FsChallenger,
    pcs::PcsParams,
    prover::{self, UnionElementSlotInput},
    schedule::{Registry, TableType},
    union::UnionInstance,
    verifier,
};
use prover::prove_fast_ligerito_union_mixed_class;
use verifier::verify_ligerito_union_mixed_class;
const DOMAIN: &[u8] = b"flock-union-element-v0";

/// Same element gate block `union_element.rs` uses: two free wires, a product,
/// a linear pin, zero padding above.
fn gate_block(kappa: usize, w0: F128, w1: F128) -> Arc<ElementTableType> {
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0)
        .free_wire(1)
        .mult(2, 0, 1)
        .linear(3, &[(0, w0), (1, w1)]);
    Arc::new(b.build().expect("gate block is valid"))
}

fn gate_witness(
    ty: &ElementTableType,
    nu: usize,
    n: usize,
    w0: F128,
    w1: F128,
    rng: &mut Rng,
) -> Vec<F128> {
    let at = |c: usize, j: usize| (c << nu) + j;
    let mut z = vec![F128::ZERO; ty.width() << nu];
    for j in 0..n {
        let (a, b) = (rng.f128(), rng.f128());
        z[at(0, j)] = a;
        z[at(1, j)] = b;
        z[at(2, j)] = a * b;
        z[at(3, j)] = w0 * a + w1 * b;
    }
    assert!(ty.satisfies(&z, nu, n), "generated witness must satisfy");
    z
}

fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: embedded_initial_k_or_default(union.dense_m(), LigeritoProfile::Fast),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(embedded_initial_k_or_default(
            union.dense_m(),
            LigeritoProfile::Fast,
        )),
        merkle_hash: Default::default(),
    }
}

/// Prove + verify an element-only union proof, recording BOTH sides.
/// Panics if the verify rejects, so a truncated shape can never escape.
fn record_element_only(
    nu: usize,
    kappas: &[usize],
    counts: &[usize],
    seed: u64,
) -> (TranscriptShape, TranscriptShape) {
    let mut rng = Rng::new(seed);
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));

    let tys: Vec<Arc<ElementTableType>> = kappas.iter().map(|&k| gate_block(k, w0, w1)).collect();
    let registry = Registry::new(
        tys.iter().map(|t| TableType::element(t.clone())).collect(),
        nu,
    );
    let slot_tys: Vec<Arc<ElementTableType>> = registry
        .element_types()
        .iter()
        .map(|t| match &t.class {
            TableClass::LargeField(e) => e.clone(),
            _ => unreachable!("element-only registry"),
        })
        .collect();
    let union = UnionInstance::new(&registry, counts.to_vec());
    let pcs_params = union_pcs_params(&union);

    let witnesses: Vec<Vec<F128>> = slot_tys
        .iter()
        .zip(counts)
        .map(|(t, &n)| gate_witness(t, nu, n, w0, w1, &mut rng))
        .collect();
    let element_slots: Vec<UnionElementSlotInput<'_>> = witnesses
        .iter()
        .map(|w| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(w)))
        .collect();

    let mut ch_p = RecordingChallenger::new(FsChallenger::new(DOMAIN));
    let (proof, commitment, _claims_p) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        element_slots,
        &mut ch_p,
    );

    let mut ch_v = RecordingChallenger::new(FsChallenger::new(DOMAIN));
    verify_ligerito_union_mixed_class(
        &union,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| {
        panic!("verify rejected (nu={nu}, kappas={kappas:?}, counts={counts:?}): {e:?} — the recorded shape would be TRUNCATED")
    });

    (ch_p.shape(), ch_v.shape())
}

/// The FS shape is a function of the CONFIG only — not of counts, not of
/// witness values. This is the property the whole fixed-topology circuit rests
/// on, so it is checked across the utilization ladder rather than assumed.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_only_transcript_shape_is_data_independent() {
    let nu = 12;
    let kappas = [3usize];

    // Same config; full, non-power-of-two, single-row and empty utilization,
    // each on a different witness seed.
    let cases: [(usize, u64); 4] = [
        (1 << 12, 0xA11CE_0001),
        (2731, 0xA11CE_0002),
        (1, 0xA11CE_0003),
        (0, 0xA11CE_0004),
    ];

    let mut reference: Option<TranscriptShape> = None;
    for (count, seed) in cases {
        let (shape_p, shape_v) = record_element_only(nu, &kappas, &[count], seed);

        // Prover and verifier share one transcript, so the prover's shape
        // must be a PREFIX of the verifier's: every op the prover performs,
        // the verifier replays in the same order. On the merged transport
        // the verifier legitimately continues past the prover's last op —
        // the succinct Ligerito verifier's closing spot-check draws its
        // base-level queries and β AFTER the proof is complete
        // (`sample_queries` + `beta_last`), randomness the prover never
        // needs. Those trailing ops must be SQUEEZES (or PoW) only: a
        // verifier-only ABSORB really would be a broken FS order. (The
        // removed jagged transport happened to be draw-symmetric end to
        // end, which is why this test could assert strict equality before.)
        match shape_p.first_difference(&shape_v) {
            None => {}
            Some(i) => {
                assert_eq!(
                    i,
                    shape_p.ops().len(),
                    "prover and verifier transcript shapes diverge MID-STREAM \
                     at op {i} (count={count}); FS order is broken \
                     (prover {:?} vs verifier {:?})",
                    shape_p.ops().get(i),
                    shape_v.ops().get(i),
                );
                for (j, op) in shape_v.ops()[i..].iter().enumerate() {
                    assert!(
                        matches!(
                            op,
                            TranscriptOp::SqueezeScalar
                                | TranscriptOp::SqueezeSlice(_)
                                | TranscriptOp::Pow { .. }
                        ),
                        "verifier-only op {} past the prover's end is not a \
                         squeeze: {op:?} (count={count}); FS order is broken",
                        i + j,
                    );
                }
            }
        }

        match &reference {
            None => reference = Some(shape_v),
            Some(r) => {
                if let Some(i) = r.first_difference(&shape_v) {
                    panic!(
                        "FS shape depends on DATA, not just config: count={count} diverges from \
                         the reference at op {i}\n  reference: {:?}\n  this run:  {:?}\n\
                         A fixed-topology circuit cannot be built until this is resolved.",
                        r.ops().get(i),
                        shape_v.ops().get(i),
                    );
                }
            }
        }
    }
}

/// The recorded shape, pinned. Any protocol change that moves the Fiat–Shamir
/// order or the message sizes lands here first.
///
/// Regenerate deliberately with `TRANSCRIPT_SHAPE_PRINT=1 ... --nocapture`,
/// and record why in the re-pin history below.
///
/// Re-pin history: initial pin (2026-07-31), element-only nu=12 kappa=3.
/// Re-pinned 2026-08-02: element-only mixed-class moved to the MERGED
/// transport (the jagged transport was removed); the shape now ends with
/// the succinct verifier's trailing spot-check draws.
/// Re-pinned 2026-08-02 (later): Merkle capping — commit absorbs are the
/// cap layers (ObserveBytes 32 -> 32·2^c), octopus removed. The cap sizes
/// are config-static, which the data-independence test enforces.
/// Runs by default since 2026-08-27: CI never passes `--ignored`, which is
/// how the 700cace sweep missed this pin (and it takes well under a second).
#[test]
fn element_only_transcript_shape_is_pinned() {
    // Re-pinned 2026-08-02: multipoint-twisted assist (proof_io v8) — the
    // per-statement assist became 128K dual values + one product sumcheck +
    // one untwisted anchor; transcript + wire moved by design.
    // Re-pinned 2026-08-02: two-product multipoint grouping (proof_io v9) —
    // element-only claims are all packed-direct, so the values absorb shrinks
    // from 128·K to ONE word per merged-row group and the sumcheck is the
    // single untwisted product; multipoint label v1.
    // Re-pinned 2026-08-04: merged-open v1 — value-only packed-direct
    // intake (points are transcript-derived; label v0 -> v1).
    // Re-pinned 2026-08-05: stratified queries (all TOMLs stratified = true)
    // — the absorbed cap moves to the top set bit of each level's query
    // count and openings carry per-summand path lengths
    // (docs/stratified-queries.tex). The squeeze widths are UNCHANGED (one
    // F128 per query); only the cap payload sizes move the shape.
    // Re-pinned 2026-08-27 after at least two transcript-moving changes
    // since the 08-05 pin, neither of which re-pinned this file (caught by
    // the Phase 0 bloat census): the 08-11 assist-transcript fork (4787509)
    // and the per-level consistency-batch grinding fix (700cace), which
    // moved each level's Pow bits. Measured: the digest at 700cace~1 was a
    // third value, neither the old nor the new pin. Two deterministic print
    // runs agreed.
    // Re-pinned 2026-08-27 for the profile consolidation (proof-IO v22,
    // bloat ledger §C): `Fast` now carries the former Fast128 schedule —
    // aggressive +2/level ladder and 16-bit query PoW at every level — so
    // the per-level query counts, caps and Pow bits all move. Two
    // deterministic print runs agreed.
    const EXPECTED: &str = "46b9b760ea72bcc0e549196bb31401e88993767d71636614de61270fd4cfdee3";

    let (_, shape) = record_element_only(12, &[3], &[1 << 12], 0xB0DD_1E01);

    // The inventory the FS chain table is sized from. Printed either way: it
    // is the number that matters and it should be visible when it moves.
    println!(
        "element-only nu=12 kappa=3: {} ops | {} absorbed bytes | {} squeezed bytes | \
         {} finalizations | {} squeezes addressed by role",
        shape.len(),
        shape.absorbed_bytes(),
        shape.squeezed_bytes(),
        shape.finalizations(),
        shape.squeeze_roles().len(),
    );

    // The FS chain's actual row inventory, derived from the schedule rather
    // than estimated. `finalize_parents` is the term a flat "one compression
    // per squeeze" model misses: a finalize collapses the chunk stack, so it
    // gets more expensive as the transcript grows.
    let inv = shape.blake3_inventory(DOMAIN.len());
    println!(
        "  BLAKE3 rows: absorb {} | chunk parents {} | finalize blocks {} | \
         finalize parents {} | xof {} = {} total",
        inv.absorb_blocks,
        inv.chunk_parents,
        inv.finalize_blocks,
        inv.finalize_parents,
        inv.xof_blocks,
        inv.total(),
    );
    println!(
        "    (a flat one-per-squeeze model would say {}, missing {} stack merges)",
        inv.total() - inv.finalize_parents,
        inv.finalize_parents,
    );

    if var_os("TRANSCRIPT_SHAPE_PRINT").is_some() {
        println!("const EXPECTED: &str = \"{}\";", shape.digest_hex());
        return;
    }
    assert_eq!(
        shape.digest_hex(),
        EXPECTED,
        "the Fiat-Shamir transcript SHAPE moved. If that was intended, \
         regenerate with TRANSCRIPT_SHAPE_PRINT=1 and record the reason; \
         if not, stop and find out what changed the FS order."
    );
}
