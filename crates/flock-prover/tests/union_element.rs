//! Element tables in the union: element-only and mixed boolean+element proofs
//! end to end, plus the tamper matrix, the dummy-row closure, and the
//! differential against the standalone element proof.
//!
//! Geometry notes (why these `(ν, κ)` pairs): the committed dense stack must
//! reach `dense_m = 22`, the smallest embedded Ligerito security config
//! (`union::MIN_DENSE_M`), so the union needs `M ≥ 22`.
//!
//! - **Element-only**: one element slot IS the address space, so
//!   `M = ν + κ + 7`. `ν = 12, κ = 3` lands exactly on 22 (2^15 committed
//!   words) with 4096 rows of 8 word-columns.
//! - **Mixed**: BLAKE3 (κ_bool = 14) at `ν = 7` gives a boolean region of
//!   `2^21`, an element region of `2^17` based at `2^21`, so `M = 22`. The
//!   boolean claims then carry ONE frozen-zero high coordinate and the element
//!   claims five frozen prefix coordinates (`element_base >> M_elem = 0b10000`).
//!
//! Run with `cargo test --release -p flock-prover --test union_element --
//! --ignored`. A DEBUG run needs `--test-threads=1`: the parallel debug harness
//! overflows a worker stack in the Ligerito recursion, the repo's known
//! pre-existing hazard (`union_mixed` behaves the same way).
//!
//! Two cases below feed the prover a witness whose DROPPED words are non-zero,
//! which violates the union's honest-witness contract. In debug that trips
//! `compact_witness`'s assertion (asserted directly, in
//! `satisfying_dummy_row_is_rejected_under_the_union`); in release the
//! assertion is compiled out and the resulting proof is REJECTED. Both halves
//! are the closure of the standalone milestone's dummy-row gap, so each is
//! checked in the profile where it is observable.

use bincode::deserialize;
use bincode::serialize;
use blake3::Compression;
use blake3::build_block_r1cs;
use blake3::generate_witness_batch_major_partial;
use el::prove;
use el::verify;
use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::merkle::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::ligerito::embedded_initial_k_or_default;
use flock_core::proof::R1csProofMixedClassMerged;
use flock_core::proof::UnionClassClaims;
use flock_core::r1cs::BlockR1cs;
use flock_core::schedule::TableClass;
use flock_prover::challenger::Challenger as _;
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::Commitment;
use flock_prover::pcs::PcsParams;
use flock_prover::prover::{self, UnionElementSlotInput, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::blake3::USEFUL_BITS;
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier;
use prover::prove_fast_ligerito_union;
use prover::prove_fast_ligerito_union_mixed_class;
use sha2_hash::Sha256;
use std::array::from_fn;
use std::env::var_os;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::panic::set_hook;
use std::panic::take_hook;
use std::sync::Arc;
use verifier::FlockVerifyError;
use verifier::verify_ligerito_union_mixed_class;

use flock_core::element_r1cs::{self as el, ElementStatement};
use flock_core::test_rng::Rng;
use sha2 as sha2_hash;
use sha2_hash::Digest as _;
const DOMAIN: &[u8] = b"flock-union-element-v0";

// ---------------------------------------------------------------------------
// The test element block: every row encoding the class supports.
// ---------------------------------------------------------------------------

/// Columns `0,1` free wires (the operands), `2 = z0·z1`, `3 = w0·z0 + w1·z1`
/// (a linear pin); columns `4..2^κ` are self-pinned zero padding. So `k = 4`
/// real word-columns of the `2^κ` the slot spans.
fn gate_block(kappa: usize, w0: F128, w1: F128) -> Arc<ElementTableType> {
    assert!(kappa >= 2, "the block needs at least 4 columns");
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0)
        .free_wire(1)
        .mult(2, 0, 1)
        .linear(3, &[(0, w0), (1, w1)]);
    Arc::new(b.build().expect("gate block is valid"))
}

/// A satisfying witness for [`gate_block`]: `n` real rows of random operands,
/// rows `[n, 2^nu)` and the padding columns all zero. BatchMajor, rows low.
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

/// PCS params over the committed dense stack — same shape as `union_mixed`'s.
fn union_pcs_params_with_profile(union: &UnionInstance<'_>, profile: LigeritoProfile) -> PcsParams {
    let log_batch_size = embedded_initial_k_or_default(union.dense_m(), profile);
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size,
        profile,
        num_lanes: union.commit_lanes(log_batch_size),
        merkle_hash: Default::default(),
    }
}

fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    union_pcs_params_with_profile(union, LigeritoProfile::Fast)
}

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Element-only union proofs
// ---------------------------------------------------------------------------

/// **Element-only union proof**, at several `(ν, κ, counts)` shapes including
/// non-power-of-two and partial counts, one and two element slots.
///
/// Also the count-proportionality assertion: the committed dense area is
/// exactly `Σ_t n_t · used_cols_t` (before the power-of-two rounding and the
/// config floor).
#[test]
// Default-run (0.03 s): the only CI coverage of the element-in-union
// end-to-end path — the heavier shapes below stay `-- --ignored`.
fn element_only_union_roundtrip() {
    let mut rng = Rng::new(0x0E1E_0001);
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));

    // Shapes: (nu, kappas, counts). M = ceil(log2(Σ 2^{nu+κ_t+7})) must be ≥ 22.
    let shapes: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![
        (12, vec![3], vec![1 << 12]),       // M = 22, full utilization
        (12, vec![3], vec![2731]),          // non-power-of-two count
        (12, vec![3], vec![1]),             // one real row
        (12, vec![3], vec![0]),             // empty table
        (11, vec![3, 3], vec![2048, 1365]), // two equal slots, M = 22
        (11, vec![4, 2], vec![1000, 2048]), // two slots of different widths
        (8, vec![7], vec![200]),            // kappa = 7: the packed-word width
    ];
    for (nu, kappas, counts) in shapes {
        let tys: Vec<Arc<ElementTableType>> =
            kappas.iter().map(|&k| gate_block(k, w0, w1)).collect();
        let registry = Registry::new(
            tys.iter().map(|t| TableType::element(t.clone())).collect(),
            nu,
        );
        assert_eq!(registry.num_boolean(), 0);
        assert_eq!(registry.num_element(), kappas.len());
        assert_eq!(registry.m_bool(), 0, "no boolean region");
        assert_eq!(
            registry.element_base(),
            0,
            "the region IS the prefix subcube"
        );
        assert!(
            registry.m_total() >= 22,
            "shape (nu={nu}, kappas={kappas:?}) commits below the Ligerito floor"
        );
        // Element types in SLOT order (area-descending), which is what the
        // counts and the prover inputs are indexed by.
        let slot_tys: Vec<Arc<ElementTableType>> = registry
            .element_types()
            .iter()
            .map(|t| match &t.class {
                TableClass::LargeField(e) => e.clone(),
                _ => unreachable!(),
            })
            .collect();
        let union = UnionInstance::new(&registry, counts.clone());
        let mut pcs_params = union_pcs_params(&union);
        if counts == [2731] {
            // One shape runs on BLAKE3 Merkle: the union entries route their
            // Ligerito configs through the PcsParams merkle-hash stamp, and
            // nothing else in the tree exercises a non-default hash
            // end-to-end (found the hard way — the raw `*_config_for` calls
            // silently dropped the hash and L0 verified against the wrong
            // leaf function).
            pcs_params.merkle_hash = HashKind::Blake3;
        }

        // Count-proportional committed area: Σ_t n_t · used_cols_t.
        let expected_dense: usize = slot_tys.iter().zip(&counts).map(|(t, &n)| t.k() * n).sum();
        assert_eq!(
            union.dense_words(),
            expected_dense,
            "dense area must be count-proportional (nu={nu}, counts={counts:?})"
        );

        let witnesses: Vec<Vec<F128>> = slot_tys
            .iter()
            .zip(&counts)
            .map(|(t, &n)| gate_witness(t, nu, n, w0, w1, &mut rng))
            .collect();
        let element_slots: Vec<UnionElementSlotInput<'_>> = witnesses
            .iter()
            .map(|w| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(w)))
            .collect();

        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, claims_p) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            Vec::new(),
            element_slots,
            &mut ch_p,
        );
        assert!(proof.boolean.is_none(), "no boolean class");
        let el = proof.element.as_ref().expect("element sub-proof");
        assert_eq!(
            el.zerocheck.rounds.len(),
            union.m_elem() - 7,
            "element zerocheck spans the region"
        );
        assert_eq!(
            el.lincheck.rounds.len(),
            union.m_elem() - 7 - nu,
            "the lincheck's row half is collapsed"
        );

        let mut ch_v = FsChallenger::new(DOMAIN);
        let claims_v = verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("verify rejected (nu={nu}, counts={counts:?}): {e:?}"));
        assert_eq!(claims_p, claims_v);
        assert!(claims_v.boolean.is_none());
        let c = claims_v.element.expect("element claims");
        // Both claims share the reused row coordinates.
        assert_eq!(&c.c_point[..nu], &c.lc_point[..nu]);
        assert_eq!(c.c_point.len(), registry.m_total() - 7);
    }
}

/// The production element path under strict `Fast`: native proving, native
/// verification, and the merged opening all see the element PIOP's PoW
/// witnesses.  The recursive-node test separately consumes this same proof
/// shape inside its Boolean R1CS verifier.
#[test]
#[ignore] // Run with `cargo test --release -p flock-prover --test union_element strict_fast_profile_grinds_element_piops -- --ignored`.
fn strict_fast_profile_grinds_element_piops() {
    let (nu, kappa, n) = (12usize, 3usize, 2731usize);
    let (w0, w1) = (F128::new(0x128, 0), F128::new(0, 0xE1E));
    let ty = gate_block(kappa, w0, w1);
    let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params_with_profile(&union, LigeritoProfile::Fast);
    let mut rng = Rng::new(0x128_E1E_0002);
    let z = gate_witness(&ty, nu, n, w0, w1, &mut rng);

    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claims_p) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&z)
        })],
        &mut ch_p,
    );
    let el = proof.element.as_ref().expect("element proof");
    let grinding = pcs_params.element_grinding();
    let e_vars = union.m_elem() - 7;
    assert_eq!(
        el.zerocheck.grinding_nonces.len(),
        grinding.zerocheck_nonce_count(e_vars)
    );
    assert_eq!(
        el.lincheck.grinding_nonces.len(),
        grinding.lincheck_nonce_count(e_vars - nu)
    );
    // The merged PCS transport contributes its own independent grinding
    // witnesses: one PoW gates the vector squeeze for both packed-direct
    // claim-batching coefficients in this element-only proof, followed by one
    // dense sumcheck witness per round and the multipoint
    // batching/round/anchor witnesses.  This keeps the test from only
    // exercising only the element PIOP half of the strict profile.
    let open = &proof.pcs_open;
    let opening_grinding = pcs_params.opening_grinding();
    assert!(
        open.ring_switches.is_empty(),
        "element-only has no RS claims"
    );
    assert_eq!(
        open.batching_nonces.len(),
        1,
        "one PoW gates the c/lincheck coefficient-vector squeeze"
    );
    assert_eq!(
        open.merged_round_nonces.len(),
        open.merged_rounds.len(),
        "one PoW before every dense quadratic-round challenge"
    );
    assert_eq!(
        open.frobenius.round_grinding_nonces.len(),
        open.frobenius.rounds.len(),
        "one PoW before every multipoint quadratic-round challenge"
    );
    assert_eq!(
        open.frobenius.anchor.grinding_nonces.len(),
        open.frobenius.anchor.rounds.len(),
        "one PoW before every Frobenius-anchor quadratic-round challenge"
    );
    assert!(opening_grinding.claim_batch_bits > 0);
    assert!(opening_grinding.merged_round_bits > 0);
    assert!(opening_grinding.multipoint.gamma_bits > 0);

    let verify = |p: &R1csProofMixedClassMerged| {
        let mut ch = FsChallenger::new(DOMAIN);
        verify_ligerito_union_mixed_class(&union, &[], &commitment, p, &pcs_params, &mut ch)
    };
    assert_eq!(verify(&proof).expect("honest proof"), claims_p);

    let mut missing = proof.clone();
    missing
        .element
        .as_mut()
        .expect("element proof")
        .lincheck
        .grinding_nonces
        .pop();
    assert!(verify(&missing).is_err(), "missing element PoW must reject");

    let mut missing_batch = proof.clone();
    missing_batch.pcs_open.batching_nonces.pop();
    assert!(
        verify(&missing_batch).is_err(),
        "missing PCS batching PoW must reject"
    );

    let mut missing_merged = proof.clone();
    missing_merged.pcs_open.merged_round_nonces.pop();
    assert!(
        verify(&missing_merged).is_err(),
        "missing dense-sumcheck PoW must reject"
    );

    let mut missing_multipoint = proof.clone();
    missing_multipoint
        .pcs_open
        .frobenius
        .round_grinding_nonces
        .pop();
    assert!(
        verify(&missing_multipoint).is_err(),
        "missing multipoint-sumcheck PoW must reject"
    );

    let mut missing_anchor = proof.clone();
    missing_anchor
        .pcs_open
        .frobenius
        .anchor
        .grinding_nonces
        .pop();
    assert!(
        verify(&missing_anchor).is_err(),
        "missing Frobenius-anchor PoW must reject"
    );
}

/// The element claims are discharged by the **opening**, not just by the PIOP.
///
/// On an element-only proof the merged opening carries NO ring-switched claims
/// at all — the two packed-direct element claims are its entire intake — so
/// tampering with the opening's own messages (the merged sumcheck, its claimed
/// `q_eval`, and the Frobenius assist) probes exactly the packed-direct path:
/// the merged target and the twisted weight `W` are built from the element
/// claim values and points, and nothing else.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_claims_are_bound_by_the_opening() {
    let (nu, kappa) = (12usize, 3usize);
    let (w0, w1) = (F128::new(17, 0), F128::new(0, 23));
    let ty = gate_block(kappa, w0, w1);
    let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
    let n = 2731usize;
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);

    let mut rng = Rng::new(0x0E1E_09E4);
    let z = gate_witness(&ty, nu, n, w0, w1, &mut rng);
    let zc = z.clone();
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&zc)
        })],
        &mut ch_p,
    );
    let verify = |p: &R1csProofMixedClassMerged| {
        let mut ch = FsChallenger::new(DOMAIN);
        verify_ligerito_union_mixed_class(&union, &[], &commitment, p, &pcs_params, &mut ch)
    };
    assert!(verify(&proof).is_ok(), "honest proof");
    assert!(
        proof.pcs_open.ring_switches.is_empty(),
        "an element-only opening has no ring-switched claims — \
         the packed-direct intake is all of it"
    );

    // `q_eval` is the merged sumcheck's claimed q̂(ρ), checked against the
    // inner eq-basis opening; the weight side of `Σ q·W = target` is built
    // ENTIRELY from the two packed-direct claims.
    let mut bad = proof.clone();
    bad.pcs_open.q_eval += F128::ONE;
    assert!(verify(&bad).is_err(), "tampered q_eval");

    // Every merged sumcheck round message.
    for i in 0..proof.pcs_open.merged_rounds.len() {
        for which in 0..2 {
            let mut bad = proof.clone();
            if which == 0 {
                bad.pcs_open.merged_rounds[i].0 += F128::ONE;
            } else {
                bad.pcs_open.merged_rounds[i].1 += F128::ONE;
            }
            assert!(
                verify(&bad).is_err(),
                "tampered merged round {i} msg {which}"
            );
        }
    }

    // The Frobenius assist — V and every round message — proves Ŵ(ρ), i.e.
    // the element claims' weight evaluation. An element-only proof has no
    // ring-switched claims, so its dual values are the scalar groups'
    // (`group_values`); `values` is empty.
    assert!(proof.pcs_open.frobenius.values.is_empty());
    let mut bad = proof.clone();
    bad.pcs_open.frobenius.group_values[0] += F128::ONE;
    assert!(verify(&bad).is_err(), "tampered frobenius V");
    for i in 0..proof.pcs_open.frobenius.rounds.len() {
        for which in 0..2 {
            let mut bad = proof.clone();
            if which == 0 {
                bad.pcs_open.frobenius.rounds[i].0 += F128::ONE;
            } else {
                bad.pcs_open.frobenius.rounds[i].1 += F128::ONE;
            }
            assert!(
                verify(&bad).is_err(),
                "tampered frobenius round {i} msg {which}"
            );
        }
    }

    // The inner eq-basis opening of q̂(ρ).
    let mut bad = proof.clone();
    bad.pcs_open.inner.ligerito.initial_cap = vec![[0u8; 32]];
    assert!(verify(&bad).is_err(), "tampered inner open root");
}

/// **Differential** (acceptance criterion 5): on the same element instance, an
/// element-only union proof accepts exactly when the STANDALONE element proof
/// does — over honest witnesses and over each way of breaking one IN THE
/// DECLARED SUPPORT. The two are not byte-identical (different statement
/// bindings, different domains), but on in-support tampers they must never
/// disagree on accept/reject.
///
/// Tampers in DROPPED words (a dummy row, a padding column) are the one
/// legitimate divergence on the merged transport: the union's committed
/// stack never contains those words and the merged open never reads the
/// padded buffer, so the union proves the sanitized statement — which is
/// satisfied — while the standalone proof (dense over the full padded
/// buffer) still sees the violation. Those cases assert the split verdict
/// explicitly; `dummy_row_is_structurally_invisible_under_the_union` pins
/// the byte-identity side of the same story.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_only_agrees_with_the_standalone_proof() {
    let (nu, kappa) = (12usize, 3usize);
    let (w0, w1) = (F128::new(11, 0), F128::new(0, 5));
    let ty = gate_block(kappa, w0, w1);
    let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
    assert_eq!(registry.m_total(), nu + kappa + 7);

    let mut rng = Rng::new(0x0E1E_D1FF);
    // Cases: honest at several counts, then five ways of breaking the witness.
    // `tamper` mutates the witness in place; `dirty_support` marks the tampers
    // that write a word the union's height-`n_t` transport DROPS (a dummy row,
    // a padding column). Those violate the union prover's honest-witness
    // contract, which `UnionInstance::compact_witness` debug-asserts — so in a
    // debug build the union prover panics rather than producing a proof, and
    // only the release arm yields a verdict. (The panic itself is asserted in
    // `dummy_row_is_structurally_invisible_under_the_union`.)
    type Tamper = fn(&mut Vec<F128>, usize, usize);
    let cases: Vec<(&str, usize, Option<Tamper>, bool)> = vec![
        ("honest full", 1 << nu, None, false),
        ("honest partial", 2731, None, false),
        ("honest empty", 0, None, false),
        (
            "broken product",
            2731,
            Some(|z: &mut Vec<F128>, nu: usize, _n: usize| z[(2 << nu) + 7] += F128::ONE),
            false,
        ),
        (
            "broken linear pin",
            2731,
            Some(|z: &mut Vec<F128>, nu: usize, _n: usize| z[(3 << nu) + 100] += F128::ONE),
            false,
        ),
        (
            "dirty dummy row",
            2731,
            Some(|z: &mut Vec<F128>, nu: usize, n: usize| z[(2 << nu) + n + 5] = F128::ONE),
            true,
        ),
        (
            "non-zero padding column",
            2731,
            Some(|z: &mut Vec<F128>, nu: usize, _n: usize| z[4 << nu] = F128::ONE),
            true,
        ),
    ];

    for (name, n, tamper, dirty_support) in cases {
        if dirty_support && cfg!(debug_assertions) {
            continue;
        }
        let mut z = gate_witness(&ty, nu, n, w0, w1, &mut rng);
        if let Some(t) = tamper {
            t(&mut z, nu, n);
        }

        // --- Union (element-only) verdict.
        let union = UnionInstance::new(&registry, vec![n]);
        let pcs_params = union_pcs_params(&union);
        let zc = z.clone();
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            Vec::new(),
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&zc)
            })],
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(DOMAIN);
        let union_ok = verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .is_ok();

        // --- Standalone verdict on the same instance.
        let stmt = ElementStatement {
            ty: &ty,
            n_log: nu,
            n,
        };
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (sp, _) = prove(&stmt, &z, &mut ch_p);
        let mut ch_v = FsChallenger::new(DOMAIN);
        let standalone_ok = verify(&stmt, &sp, &mut ch_v).is_ok();

        if dirty_support {
            // Dropped words: structurally invisible to the union (see the
            // doc comment), a relation violation to the standalone prover.
            assert!(
                union_ok,
                "the union must accept '{name}' — the dirty word is not part \
                 of its statement"
            );
            assert!(!standalone_ok, "the standalone proof must reject '{name}'");
        } else {
            assert_eq!(
                union_ok, standalone_ok,
                "union and standalone disagree on '{name}' \
                 (union {union_ok}, standalone {standalone_ok})"
            );
            // Every in-support tamper here is a RELATION violation, so both
            // provers' verdicts are also the expected ones.
            assert_eq!(union_ok, tamper.is_none(), "unexpected verdict on '{name}'");
        }
    }
}

/// **Dummy-row closure** (acceptance criterion 4). The standalone milestone's
/// known gap (`satisfying_dummy_row_is_not_detected`) is that a *satisfying*
/// non-zero row past the declared count verifies: the PIOP proves the relation
/// on every row, not that dummy rows are zero.
///
/// Under the union that gap closes STRUCTURALLY — the dummy row is not part
/// of the statement at all — and this pins both halves of the mechanism:
///
/// 1. the height-`n_t` transport does not commit those words at all
///    (`jagged_heights` stops at `n_t`), so the compaction's
///    dropped-words-are-zero invariant catches them in debug builds; and
/// 2. on the MERGED transport a release prover's proof on such a witness is
///    BYTE-IDENTICAL to the proof of the same witness with the row zeroed —
///    and both verify. The dirty word is invisible end to end: the element
///    zerocheck's `RowSupport` skips dead rows (≤ 50% region utilization,
///    this shape included), the lincheck's row collapse truncates at the
///    declared count, the committed stack drops the word, and the merged
///    open never reads the padded buffer for a packed-direct claim (the
///    same invariant `merged_padding_unread_poison_pool` pins for pooled
///    boolean padding). Byte-identity is a STRONGER statement than the
///    jagged transport's old fail-closed rejection: nothing the prover
///    computes ever depended on the dropped word.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn dummy_row_is_structurally_invisible_under_the_union() {
    let (nu, kappa) = (12usize, 3usize);
    let (w0, w1) = (F128::new(13, 0), F128::new(0, 9));
    let ty = gate_block(kappa, w0, w1);
    let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
    let n = 2731usize;
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);

    let mut rng = Rng::new(0x0E1E_DEAD);
    let mut z = gate_witness(&ty, nu, n, w0, w1, &mut rng);
    // Row `n + 5` is a dummy row. Fill it with a fully SATISFYING assignment —
    // the standalone PIOP accepts exactly this (see its scope-boundary test).
    let (a, b) = (rng.f128(), rng.f128());
    let at = |c: usize, j: usize| (c << nu) + j;
    z[at(0, n + 5)] = a;
    z[at(1, n + 5)] = b;
    z[at(2, n + 5)] = a * b;
    z[at(3, n + 5)] = w0 * a + w1 * b;
    assert!(
        ty.satisfies(&z, nu, 1 << nu),
        "the tampered witness satisfies the RELATION on every row"
    );

    // (1) The words are not committed: the heights stop at n_t, so the
    // compaction map drops them — and the honest-witness invariant it asserts
    // in debug builds is exactly "every dropped word is zero", which this
    // witness violates.
    let heights = union.jagged_heights();
    assert!(
        heights.iter().all(|&h| h as usize == n || h == 0),
        "used columns commit exactly n_t rows"
    );
    assert_eq!(union.dense_words(), ty.k() * n);

    // (1b) DEBUG builds: the compaction's honest-witness assertion fires — the
    // prover refuses to build a stack whose dropped words are not zero.
    if cfg!(debug_assertions) {
        let hook = take_hook();
        set_hook(Box::new(|_| {}));
        let caught = catch_unwind(AssertUnwindSafe(|| {
            union.compact_witness(&z);
        }));
        set_hook(hook);
        assert!(
            caught.is_err(),
            "compact_witness must reject a non-zero dropped word"
        );
    }

    // (2) RELEASE builds: the debug assertion is compiled out, so the prover
    // runs — and the proof it produces on the dirty witness must be
    // BYTE-IDENTICAL to the proof of the clean one: no computation on the
    // prover's path may read the dropped word. Any divergence here means a
    // consumer of the padded buffer leaked past the declared support.
    if !cfg!(debug_assertions) {
        let prove = |z: &[F128]| {
            let zc = z.to_vec();
            let mut ch_p = FsChallenger::new(DOMAIN);
            prove_fast_ligerito_union_mixed_class(
                &union,
                &pcs_params,
                Vec::new(),
                vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                    dst.copy_from_slice(&zc)
                })],
                &mut ch_p,
            )
        };
        let (proof_dirty, cm_dirty, claims_dirty) = prove(&z);
        let mut z_clean = z.clone();
        for c in 0..4 {
            z_clean[at(c, n + 5)] = F128::ZERO;
        }
        let (proof_clean, cm_clean, claims_clean) = prove(&z_clean);
        assert_eq!(cm_dirty.cap, cm_clean.cap, "same committed stack");
        assert_eq!(claims_dirty, claims_clean, "same claims");
        assert_eq!(
            serialize(&proof_dirty).unwrap(),
            serialize(&proof_clean).unwrap(),
            "a satisfying non-zero dummy row must be INVISIBLE: the dirty and \
             clean witnesses must produce byte-identical proofs"
        );
    }

    // The clean witness verifies (in debug builds this is also the only
    // prove this test runs — the dirty one panics in compact_witness).
    for c in 0..4 {
        z[at(c, n + 5)] = F128::ZERO;
    }
    let zc = z.clone();
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&zc)
        })],
        &mut ch_p,
    );
    let mut ch_v = FsChallenger::new(DOMAIN);
    assert!(
        verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .is_ok(),
        "the clean witness must verify"
    );
}

// ---------------------------------------------------------------------------
// Mixed boolean + element
// ---------------------------------------------------------------------------

/// The mixed setup: BLAKE3 (κ = 14) + the element gate block, at ν = 7 → a
/// `2^21` boolean region, a `2^17` element region based at `2^21`, `M = 22`.
struct Mixed {
    registry: Registry,
    blake3_r1cs: BlockR1cs,
    ty: Arc<ElementTableType>,
    w: (F128, F128),
}

fn mixed_setup(nu: usize, kappa: usize) -> Mixed {
    let w = (F128::new(3, 0), F128::new(0, 7));
    let ty = gate_block(kappa, w.0, w.1);
    let blake3_r1cs = build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            // Element type FIRST in the input list, so the class-major sort has
            // something to do.
            TableType::element(ty.clone()),
            TableType::from_block_r1cs(&blake3_r1cs),
        ],
        nu,
    );
    assert_eq!(registry.num_boolean(), 1, "BLAKE3 sorts before the element");
    assert!(registry.types()[1].is_element());
    Mixed {
        registry,
        blake3_r1cs,
        ty,
        w,
    }
}

/// **THE milestone test**: one proof attesting a mixed batch of boolean rows
/// (BLAKE3 compressions) and element rows (F128 gates), plus the tamper matrix.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mixed_boolean_element_roundtrip_and_tamper() {
    let (nu, kappa) = (7usize, 3usize);
    let m = mixed_setup(nu, kappa);
    assert_eq!(m.registry.m_total(), 22);
    assert_eq!(m.registry.m_bool(), 21);
    assert_eq!(m.registry.m_elem(), 17);
    assert_eq!(m.registry.element_base(), 1 << 21);

    let (n_bool, n_elem) = (100usize, 90usize); // both non-powers of two
    let union = UnionInstance::new(&m.registry, vec![n_bool, n_elem]);
    let pcs_params = union_pcs_params(&union);
    // Region geometry the two PIOPs run over.
    assert_eq!(union.boolean_packed_len(), 1 << 14);
    assert_eq!(
        union.element_word_range(),
        (1 << 14)..((1 << 14) + (1 << 10))
    );
    // Count-proportional dense area across BOTH classes. The boolean side is
    // USEFUL_BITS packed into words (92 as of the Option F adders — a
    // hardcoded 93 rotted here when the encoder dieted).
    let bool_words = USEFUL_BITS.div_ceil(128);
    assert_eq!(
        union.dense_words(),
        bool_words * n_bool + m.ty.k() * n_elem,
        "dense area is count-proportional over both classes"
    );

    let mut rng = Rng::new(0x0E1E_B1AD);
    let inputs = random_blake3_inputs(&mut rng, n_bool);
    let z_elem = gate_witness(&m.ty, nu, n_elem, m.w.0, m.w.1, &mut rng);
    let circuit = m.blake3_r1cs.csc_lincheck_circuit();

    let prove = |union: &UnionInstance<'_>, params: &PcsParams| {
        let z_elem = z_elem.clone();
        let mut ch = FsChallenger::new(DOMAIN);
        prove_fast_ligerito_union_mixed_class(
            union,
            params,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(&inputs, nu),
                circuit,
            )],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z_elem)
            })],
            &mut ch,
        )
    };
    let (proof, commitment, claims_p) = prove(&union, &pcs_params);
    assert!(proof.boolean.is_some() && proof.element.is_some());

    let verify = |union: &UnionInstance<'_>,
                  params: &PcsParams,
                  commitment: &Commitment,
                  proof: &R1csProofMixedClassMerged| {
        let mut ch = FsChallenger::new(DOMAIN);
        verify_ligerito_union_mixed_class(union, &[circuit], commitment, proof, params, &mut ch)
    };
    let claims_v = verify(&union, &pcs_params, &commitment, &proof).expect("honest mixed proof");
    assert_eq!(claims_p, claims_v);
    let bc = claims_v.boolean.as_ref().expect("boolean claims");
    let ec = claims_v.element.as_ref().expect("element claims");
    // The boolean claims carry M − M_bool = 1 frozen-ZERO high coordinate.
    assert_eq!(*bc.ab.point.x_outer.last().unwrap(), F128::ZERO);
    assert_eq!(*bc.c.point.x_outer.last().unwrap(), F128::ZERO);
    // The element claims carry the region's five frozen prefix coordinates
    // (element_base >> M_elem = 0b10000, LSB-first).
    assert_eq!(
        &ec.c_point[ec.c_point.len() - 5..],
        &[F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO, F128::ONE]
    );

    // ---- Tamper matrix ---------------------------------------------------
    //
    // (a) A tampered ELEMENT word.
    {
        let mut bad_elem = z_elem.clone();
        bad_elem[(2 << nu) + 3] += F128::ONE; // break one product
        let inputs = &inputs;
        let mut ch = FsChallenger::new(DOMAIN);
        let (p, cm, _) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(inputs, nu),
                circuit,
            )],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&bad_elem)
            })],
            &mut ch,
        );
        assert!(
            verify(&union, &pcs_params, &cm, &p).is_err(),
            "a tampered element word must be rejected"
        );
    }

    // (b) A tampered BOOLEAN word: flip one input bit so the trace is
    //     inconsistent with the (unchanged) declared statement is not possible
    //     — instead flip a witness word directly by proving on a corrupted
    //     boolean witness through the prebuilt path.
    {
        let (mut z, a, b, stripe) = generate_witness_batch_major_partial(&inputs, nu);
        z[5] += F128::ONE;
        let z_elem = z_elem.clone();
        let mut ch = FsChallenger::new(DOMAIN);
        let (p, cm, _) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new((z, a, b, stripe), circuit)],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z_elem)
            })],
            &mut ch,
        );
        assert!(
            verify(&union, &pcs_params, &cm, &p).is_err(),
            "a tampered boolean word must be rejected"
        );
    }

    // (c) Wrong declared counts, each class in turn — they bind in the
    //     statement, in the heights, and in the lincheck's pin target.
    for bad_counts in [
        vec![n_bool + 1, n_elem],
        vec![n_bool - 1, n_elem],
        vec![n_bool, n_elem + 1],
        vec![n_bool, n_elem - 1],
    ] {
        let bad = UnionInstance::new(&m.registry, bad_counts.clone());
        let bad_params = union_pcs_params(&bad);
        assert!(
            verify(&bad, &bad_params, &commitment, &proof).is_err(),
            "counts {bad_counts:?} must be rejected"
        );
    }

    // (d) Swapped registry digest: same geometry, different element base block
    //     (one linear coefficient changed). Verifier-side quantities are all
    //     identical, so only the digest binding catches it.
    {
        let other_ty = gate_block(kappa, m.w.0, m.w.1 + F128::ONE);
        let other = Registry::new(
            vec![
                TableType::from_block_r1cs(&m.blake3_r1cs),
                TableType::element(other_ty),
            ],
            nu,
        );
        assert_ne!(m.registry.digest(), other.digest());
        let bad = UnionInstance::new(&other, vec![n_bool, n_elem]);
        assert_eq!(bad.dense_m(), union.dense_m(), "same geometry");
        assert!(
            verify(&bad, &pcs_params, &commitment, &proof).is_err(),
            "a swapped registry digest must be rejected"
        );
    }

    // (e) Tampered claim values, in every sub-proof.
    {
        type Mut = fn(&mut R1csProofMixedClassMerged);
        let cases: [(&str, Mut); 5] = [
            ("element ec", |p| {
                p.element.as_mut().unwrap().zerocheck.ec += F128::ONE
            }),
            ("element ea", |p| {
                p.element.as_mut().unwrap().zerocheck.ea += F128::ONE
            }),
            ("element z_eval", |p| {
                p.element.as_mut().unwrap().lincheck.z_eval += F128::ONE
            }),
            ("element zc round", |p| {
                p.element.as_mut().unwrap().zerocheck.rounds[0].0 += F128::ONE
            }),
            ("element lc round", |p| {
                p.element.as_mut().unwrap().lincheck.rounds[0].1 += F128::ONE
            }),
        ];
        for (name, mutate) in cases {
            let mut bad = proof.clone();
            mutate(&mut bad);
            assert!(
                verify(&union, &pcs_params, &commitment, &bad).is_err(),
                "tampered {name} must be rejected"
            );
        }
    }

    // (f) A missing / extra class sub-proof.
    {
        let mut bad = proof.clone();
        bad.element = None;
        assert!(matches!(
            verify(&union, &pcs_params, &commitment, &bad),
            Err(FlockVerifyError::ClassMismatch)
        ));
        let mut bad = proof.clone();
        bad.boolean = None;
        assert!(matches!(
            verify(&union, &pcs_params, &commitment, &bad),
            Err(FlockVerifyError::ClassMismatch)
        ));
    }

    // (g) Truncated and bit-flipped proof bytes.
    {
        let bytes = serialize(&proof).expect("serialize");
        let decoded: R1csProofMixedClassMerged = deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded, proof);
        assert!(verify(&union, &pcs_params, &commitment, &decoded).is_ok());
        for frac in [1usize, 2, 4, 8] {
            let cut = bytes.len() - bytes.len() / frac;
            match deserialize::<R1csProofMixedClassMerged>(&bytes[..cut]) {
                Err(_) => {}
                Ok(p) => assert!(
                    verify(&union, &pcs_params, &commitment, &p).is_err(),
                    "truncation to {cut} bytes verified"
                ),
            }
        }
        let n_flips = 16usize;
        for i in 0..n_flips {
            let pos = i * (bytes.len() / n_flips);
            let mut b = bytes.clone();
            b[pos] ^= 1 << (i % 8);
            match deserialize::<R1csProofMixedClassMerged>(&b) {
                Err(_) => {}
                Ok(p) => assert!(
                    verify(&union, &pcs_params, &commitment, &p).is_err(),
                    "bit flip at byte {pos} verified"
                ),
            }
        }
    }

    // (h) A different transcript domain.
    {
        let mut ch = FsChallenger::new(b"a-different-domain");
        assert!(
            verify_ligerito_union_mixed_class(
                &union,
                &[circuit],
                &commitment,
                &proof,
                &pcs_params,
                &mut ch,
            )
            .is_err(),
            "the proof must be bound to its transcript"
        );
    }
}

/// A mixed proof where ONE class has count zero, in both directions. The
/// zero-count class still occupies its address space (all zero) and its PIOP
/// still runs — over a domain whose declared support is empty.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mixed_with_one_class_at_zero_count() {
    let (nu, kappa) = (7usize, 3usize);
    let m = mixed_setup(nu, kappa);
    let circuit = m.blake3_r1cs.csc_lincheck_circuit();

    for (n_bool, n_elem) in [(0usize, 90usize), (100usize, 0usize)] {
        let union = UnionInstance::new(&m.registry, vec![n_bool, n_elem]);
        let pcs_params = union_pcs_params(&union);
        let bool_words = USEFUL_BITS.div_ceil(128);
        assert_eq!(
            union.dense_words(),
            bool_words * n_bool + m.ty.k() * n_elem,
            "a zero count contributes no committed words"
        );

        let mut rng = Rng::new(0x0E1E_0000 + n_bool as u64);
        let inputs = random_blake3_inputs(&mut rng, n_bool);
        let z_elem = gate_witness(&m.ty, nu, n_elem, m.w.0, m.w.1, &mut rng);

        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(&inputs, nu),
                circuit,
            )],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z_elem)
            })],
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(DOMAIN);
        verify_ligerito_union_mixed_class(
            &union,
            &[circuit],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("verify rejected (n_bool={n_bool}, n_elem={n_elem}): {e:?}"));
    }
}

/// A boolean-only registry proved through the MIXED-CLASS merged entry must
/// be transcript-identical to the same registry proved through the plain
/// merged entry: the mixed-class pipeline adds nothing when there is no
/// element class. Only the proof struct differs (an `Option` wrapper), so the
/// sub-proof bytes and the claims must match exactly.
///
/// This is also the TWO-BODY drift detector: the plain merged entry is a
/// hand-written standalone body while the mixed-class one goes through
/// `prove_union_with_binding` — until they are unified, this equality is the
/// proof that the two bodies produce the same transcript.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn boolean_only_mixed_class_matches_the_plain_entry() {
    // ν = 8 so a BLAKE3-only registry lands on M = 22, the smallest embedded
    // Ligerito config (at ν = 7 it would be M = 21, which has none).
    let nu = 8usize;
    let blake3_r1cs = build_block_r1cs(nu);
    let registry = Registry::new(vec![TableType::from_block_r1cs(&blake3_r1cs)], nu);
    let n = 100usize;
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);
    let circuit = blake3_r1cs.csc_lincheck_circuit();
    let mut rng = Rng::new(0x0E1E_B001);
    let inputs = random_blake3_inputs(&mut rng, n);

    let mut ch = FsChallenger::new(DOMAIN);
    let (plain, cm_plain, claim_plain) = prove_fast_ligerito_union(
        &union,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(&inputs, nu),
            circuit,
        )],
        &mut ch,
    );
    let tail_plain = ch.sample_f128();

    let mut ch = FsChallenger::new(DOMAIN);
    let (mixed, cm_mixed, claim_mixed) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(&inputs, nu),
            circuit,
        )],
        Vec::new(),
        &mut ch,
    );
    let tail_mixed = ch.sample_f128();

    assert_eq!(cm_plain.cap, cm_mixed.cap);
    assert!(mixed.element.is_none());
    let b = mixed.boolean.as_ref().expect("boolean sub-proof");
    assert_eq!(
        serialize(&plain.zerocheck).unwrap(),
        serialize(&b.zerocheck).unwrap()
    );
    assert_eq!(
        serialize(&plain.lincheck).unwrap(),
        serialize(&b.lincheck).unwrap()
    );
    assert_eq!(
        serialize(&plain.pcs_open).unwrap(),
        serialize(&mixed.pcs_open).unwrap()
    );
    assert_eq!(Some(claim_plain), claim_mixed.boolean);
    assert_eq!(
        tail_plain, tail_mixed,
        "post-proof transcript state must match"
    );

    // …and both verify.
    let mut ch = FsChallenger::new(DOMAIN);
    verify_ligerito_union_mixed_class(&union, &[circuit], &cm_mixed, &mixed, &pcs_params, &mut ch)
        .expect("boolean-only mixed-class proof verifies");
}

// ---------------------------------------------------------------------------
// Byte-identity fixtures for the mixed-class wire payload.
//
// SHA-256 digests of the serialized proof bundle on deterministic seeded
// witnesses — the same convention as `union_m6_fixtures`, and the same
// determinism argument: witness generation is pure, the challenger is
// Fiat-Shamir, and every parallel reduction is XOR/add in GF(2^128)
// (associative + commutative), so the digests are stable across runs and
// thread counts.
//
// These pin the mixed-class transcript AS SHIPPED so a later optimization has
// to prove it changed no value. Regenerate after an INTENTIONAL protocol
// change with `ELEMENT_FIXTURES_PRINT=1 ... --nocapture`.
// ---------------------------------------------------------------------------

// Re-pinned 2026-08-02: multipoint-twisted assist (proof_io v8) — the
// per-statement assist became 128K dual values + one product sumcheck +
// one untwisted anchor; transcript + wire moved by design.
// Re-pinned 2026-08-02: two-product multipoint grouping (proof_io v9) —
// the element claims here are packed-direct, so they collapse into
// merged-column scalar groups carrying ONE dual value each (128 -> 1 per
// distinct row point); multipoint label v1.
fn check(label: &str, expected: &str, got: String) {
    if var_os("ELEMENT_FIXTURES_PRINT").is_some() {
        println!("(\"{label}\", \"{got}\"),");
        return;
    }
    assert_eq!(
        got, expected,
        "mixed-class byte-identity broken for fixture `{label}`"
    );
}

/// The MERGED mirror of [`bundle_digest`].
fn bundle_digest_merged(
    proof: &R1csProofMixedClassMerged,
    commitment: &Commitment,
    claims: &UnionClassClaims,
) -> String {
    let mut h = Sha256::new();
    h.update(serialize(proof).expect("proof serializes"));
    h.update(commitment.cap.as_flattened());
    let mut absorb = |v: F128| {
        h.update(v.lo.to_le_bytes());
        h.update(v.hi.to_le_bytes());
    };
    if let Some(b) = &claims.boolean {
        absorb(b.ab.value);
        absorb(b.c.value);
    }
    if let Some(e) = &claims.element {
        absorb(e.c_value);
        absorb(e.lc_value);
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

/// The MERGED mirror of [`mixed_class_proof_bytes_pinned`]: the same seven
/// statements (same seeds, same witnesses) through the merged transport.
/// Minted ahead of the jagged transport's removal — the merged mixed-class
/// path previously had no absolute byte anchor at all, and the ELEMENT_ONLY
/// half pins the element-only merged transcript at its birth (it exists
/// only since `open_batch_merged` learned to skip an empty ring-switch
/// batch).
///
/// Re-pinned 2026-08-05 (mix-* only): the BLAKE3 "Option E" lin-id drop
/// narrowed the b3 table 121 -> 93 word-cols; element-only fixtures held
/// byte-identical (the drop is blake3-local).
/// Re-pinned on `recursion_circuit` 2026-08-02 at the jagged-removal merge:
/// this branch has replacement sampling (`4e46b0a`), which `multitable`
/// predates — the multitable-minted digests can never match here.
/// Re-pinned 2026-08-02: Merkle capping (proof_io v7): cap layers absorbed instead of roots (ObserveBytes 32 -> 32*2^c per commit absorb); octopus multi-proofs replaced by flat per-query capped paths.
/// Re-pinned 2026-08-02: two-product multipoint grouping (proof_io v9) —
/// element/packed-direct claims enter as merged-column scalar groups, one
/// dual value each; multipoint label v1, values absorb 128K -> 128R+P.
// Re-pinned 2026-08-04: merged-open v1 — value-only packed-direct intake
// (points are transcript-derived; label v0 -> v1).
// Re-pinned 2026-08-05: stratified queries (m22_fast stratified = true) —
// per-summand sampling, cap at the top set bit, per-summand path lengths
// (docs/stratified-queries.tex); same (1-gamma)^q bound, smaller proofs.
// Re-pinned 2026-08-12, two 2026-08-11 protocol changes at once: path
// truncation (`ec51b71` — per-query paths stop at the cap depth) and the
// assist transcript fork (`4787509` — merged opens prove the assist
// beside the inner open on a forked chain, unconditional). All seven
// fixtures moved; roundtrip suites green, digests stable across two
// print runs.
// Re-pinned 2026-08-13 for the deliberate proof-IO v18 Ligerito changes:
// two-point OOD binding, the Flock paper's Appendix C.3 algebraic grinding,
// and strict-128 Johnson query schedules. All seven fixtures moved;
// roundtrip suites are green and digests were stable across two print runs.
// Re-pinned 2026-08-13 for the deliberate proof-IO v19 Ligerito change:
// F256 mutual correlated agreement and the base-field split handoff. All
// seven fixtures moved; roundtrip suites are green and deterministic
// generation produced stable digests.
// Re-pinned 2026-08-13 for proof-IO v20. Strict Fast and Slim proofs now
// include every non-Ligerito grinding nonce, changing the transcript and
// serialized proof bytes. Two deterministic print runs agreed.
// Re-pinned 2026-08-13 after circuit digests began binding fixed-public
// declarations and the retained registry. The statement transcript changes
// by design; two deterministic print runs agreed for all seven fixtures.
// Re-pinned 2026-08-27 after two transcript-moving changes since the 08-13
// pin, neither of which re-pinned this file (caught by the Phase 0 bloat
// census): the 08-14 blake3 Option-F adders (176c869) and the per-level
// consistency-batch grinding fix (700cace: each Ligerito level's alpha now
// grinds under its OWN bits). Measured at 700cace~1: the two element-only
// pins that moved did so under 700cace alone; mix-100-90 and mix-100-0 had
// already reached their new values on 08-14 and 700cace left them;
// mix-128-128 and mix-0-90 moved under both. The empty-instance fixture
// elem-merged-nu12-0 held throughout. union_m6_fixtures was re-pinned with
// 700cace but this file was missed. Two deterministic print runs agreed.
// Re-pinned 2026-08-27 for the profile consolidation (proof-IO v22, bloat
// ledger §C): `Fast` now carries the former Fast128 schedule (aggressive
// +2/level ladder, 16-bit query PoW every level), so every fixture moved,
// including the empty-instance one (the ladder itself changed, not just
// the grinding). Two deterministic print runs agreed.
// Runs by default since 2026-08-27: CI never passes `--ignored`, which is
// how the 700cace sweep missed this pin. Same policy as `union_m6_fixtures`.
#[test]
fn mixed_class_merged_proof_bytes_pinned() {
    const ELEMENT_ONLY: [(&str, usize, &str); 3] = [
        (
            "elem-merged-nu12-full",
            1 << 12,
            "43e3c8b0de3a1fbd682ee1a07d78c4bdfce0de49a25fa237648eaef113f90532",
        ),
        (
            "elem-merged-nu12-2731",
            2731,
            "fc293680f649aecf034122cb391b1ff35e49240b494d261dde3f4ed35d1bd685",
        ),
        (
            "elem-merged-nu12-0",
            0,
            "ebb1dd96ef1a5a381f2b9f47f3d194d0505b39370f581c648a064ced08065a1b",
        ),
    ];
    const MIXED: [(&str, [usize; 2], &str); 4] = [
        (
            "mix-merged-nu7-128-128",
            [128, 128],
            "1d0f6680e46c7254447cf2b2e37a6b7f06213d6a1fea17fcce9cd8f857a24ee7",
        ),
        (
            "mix-merged-nu7-100-90",
            [100, 90],
            "84456fa03bd629fd217a21fda86b8d3527ea5819a6971988a6aa68bd119efca8",
        ),
        (
            "mix-merged-nu7-0-90",
            [0, 90],
            "1e5a3c409edf63ded8fb863d74c090598725267e048a84646ebc2fa69cb99398",
        ),
        (
            "mix-merged-nu7-100-0",
            [100, 0],
            "bee4a92a2b0fec998aeae218a29608c1a0512d2e9dc7e161aff44950ed564b65",
        ),
    ];

    let (w0, w1) = (F128::new(0x51F0, 0), F128::new(0, 0x2C7E));

    // ---- element-only ----
    let (nu, kappa) = (12usize, 3usize);
    let ty = gate_block(kappa, w0, w1);
    let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
    for (label, n, expected) in ELEMENT_ONLY {
        let union = UnionInstance::new(&registry, vec![n]);
        let pcs_params = union_pcs_params(&union);
        let mut rng = Rng::new(0xE1E_0000 ^ n as u64);
        let z = gate_witness(&ty, nu, n, w0, w1, &mut rng);
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claims) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            Vec::new(),
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z)
            })],
            &mut ch,
        );
        // A pin is only meaningful for bytes the verifier accepts.
        let mut ch_v = FsChallenger::new(DOMAIN);
        let claims_v = verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("{label}: pinned proof must verify: {e:?}"));
        assert_eq!(claims_v, claims, "{label}: verifier claims must match");
        check(
            label,
            expected,
            bundle_digest_merged(&proof, &commitment, &claims),
        );
    }

    // ---- mixed ----
    let (nu, kappa) = (7usize, 3usize);
    let m = mixed_setup(nu, kappa);
    let circuit = m.blake3_r1cs.csc_lincheck_circuit();
    for (label, counts, expected) in MIXED {
        let [n_bool, n_elem] = counts;
        let union = UnionInstance::new(&m.registry, counts.to_vec());
        let pcs_params = union_pcs_params(&union);
        let mut rng = Rng::new(0xE1E_1000 ^ ((n_bool as u64) << 16) ^ n_elem as u64);
        let inputs = random_blake3_inputs(&mut rng, n_bool);
        let z_elem = gate_witness(&m.ty, nu, n_elem, m.w.0, m.w.1, &mut rng);
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, claims) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(&inputs, nu),
                circuit,
            )],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z_elem)
            })],
            &mut ch,
        );
        // A pin is only meaningful for bytes the verifier accepts.
        let mut ch_v = FsChallenger::new(DOMAIN);
        let claims_v = verify_ligerito_union_mixed_class(
            &union,
            &[circuit],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("{label}: pinned proof must verify: {e:?}"));
        assert_eq!(claims_v, claims, "{label}: verifier claims must match");
        check(
            label,
            expected,
            bundle_digest_merged(&proof, &commitment, &claims),
        );
    }
}

/// A mixed boolean+element proof over the MERGED transport — the first thing
/// to drive a packed-direct claim through it.
///
/// The merged (Frobenius) path is the shipped, capacity-free one, but element
/// claims are packed-direct and it had no intake for them, so mixed proofs
/// were confined to the unmerged jagged path and its padded-domain
/// auxiliaries. The intake works by expressing a packed-direct claim's weight
/// as the F₂-linear map `x ↦ γ·x`, which the merged weight builder cannot
/// distinguish from a ring-switched claim's Φ-fold.
///
/// Both transports carry the same claim set, so they must agree on the CLAIMS
/// while their openings differ — and the merged one must still reject
/// tampering, or the packed-direct claims would be riding along unchecked.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mixed_proofs_verify_over_the_merged_transport() {
    let (nu, kappa) = (7usize, 3usize);
    let m = mixed_setup(nu, kappa);
    let (n_bool, n_elem) = (100usize, 90usize);
    let union = UnionInstance::new(&m.registry, vec![n_bool, n_elem]);
    let pcs_params = union_pcs_params(&union);

    let mut rng = Rng::new(0x_4D_47_44_01);
    let inputs = random_blake3_inputs(&mut rng, n_bool);
    let z_elem = gate_witness(&m.ty, nu, n_elem, m.w.0, m.w.1, &mut rng);
    let circuit = m.blake3_r1cs.csc_lincheck_circuit();

    let bool_slot =
        || UnionSlotProverInput::new(generate_witness_batch_major_partial(&inputs, nu), circuit);
    let elem_slot = || {
        let z = z_elem.clone();
        UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
    };

    // Merged.
    let mut ch = FsChallenger::new(DOMAIN);
    let (merged, commitment, claims_m) = prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        vec![bool_slot()],
        vec![elem_slot()],
        &mut ch,
    );
    assert!(merged.boolean.is_some() && merged.element.is_some());
    let mut ch_v = FsChallenger::new(DOMAIN);
    let got = verify_ligerito_union_mixed_class(
        &union,
        &[circuit],
        &commitment,
        &merged,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("merged rejected an honest mixed proof: {e:?}"));
    assert_eq!(got, claims_m);

    // Tampering must still be caught on the merged path.
    for (what, bad) in [
        ("opening", {
            let mut b = merged.clone();
            b.pcs_open.q_eval += F128::ONE;
            b
        }),
        ("element claim", {
            let mut b = merged.clone();
            b.element.as_mut().unwrap().lincheck.z_eval += F128::ONE;
            b
        }),
        ("boolean claim", {
            let mut b = merged.clone();
            b.boolean.as_mut().unwrap().lincheck.z_partial[0] += F128::ONE;
            b
        }),
    ] {
        let mut ch_v = FsChallenger::new(DOMAIN);
        assert!(
            verify_ligerito_union_mixed_class(
                &union,
                &[circuit],
                &commitment,
                &bad,
                &pcs_params,
                &mut ch_v,
            )
            .is_err(),
            "tampered {what} must be rejected by the merged transport"
        );
    }
}

/// Element-only registries produce ZERO ring-switched claims, so the merged
/// open's ring-switch batch is empty — the `!x_outers.is_empty()` guard in
/// `open_batch_merged` is what makes this proof possible at all (the batched
/// ring-switch prover asserts a non-empty batch). This test is the birth
/// certificate of the element-only MERGED transcript, and the
/// cross-transport claims/root equality against the jagged element-only
/// prove is the one differential check this path gets while the jagged
/// transport still exists.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_only_proofs_verify_over_the_merged_transport() {
    let mut rng = Rng::new(0x0E1E_4D47);
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));
    // Full utilization, a partial count, and two slots of different widths —
    // the same shape axes as `element_only_union_roundtrip`.
    let shapes: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![
        (12, vec![3], vec![1 << 12]),
        (12, vec![3], vec![2731]),
        (11, vec![4, 2], vec![1000, 2048]),
    ];
    for (nu, kappas, counts) in shapes {
        let tys: Vec<Arc<ElementTableType>> =
            kappas.iter().map(|&k| gate_block(k, w0, w1)).collect();
        let registry = Registry::new(
            tys.iter().map(|t| TableType::element(t.clone())).collect(),
            nu,
        );
        let slot_tys: Vec<Arc<ElementTableType>> = registry
            .element_types()
            .iter()
            .map(|t| match &t.class {
                TableClass::LargeField(e) => e.clone(),
                _ => unreachable!(),
            })
            .collect();
        let union = UnionInstance::new(&registry, counts.clone());
        let pcs_params = union_pcs_params(&union);
        let witnesses: Vec<Vec<F128>> = slot_tys
            .iter()
            .zip(&counts)
            .map(|(t, &n)| gate_witness(t, nu, n, w0, w1, &mut rng))
            .collect();
        let element_slots = || -> Vec<UnionElementSlotInput<'_>> {
            witnesses
                .iter()
                .map(|w| UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(w)))
                .collect()
        };

        // Merged.
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (merged, commitment, claims_m) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            Vec::new(),
            element_slots(),
            &mut ch_p,
        );
        assert!(merged.boolean.is_none(), "no boolean class");
        assert!(
            merged.pcs_open.ring_switches.is_empty(),
            "element-only: the merged open must carry no ring-switched claims"
        );
        let mut ch_v = FsChallenger::new(DOMAIN);
        let claims_v = verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &merged,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| {
            panic!(
                "merged rejected an honest element-only proof (nu={nu}, counts={counts:?}): {e:?}"
            )
        });
        assert_eq!(claims_v, claims_m);
    }
}
