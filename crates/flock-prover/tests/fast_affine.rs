//! The `fast-affine` profile end to end: a BLAKE3 union prove whose merged
//! open runs the affine F128 inner opening (`MergedInnerOpen::Affine`),
//! verified by the shipped union verifier, plus the rejections that keep the
//! two inner openings apart. m = 24 is the smallest registered affine ladder
//! (`ligerito::affine::affine_ladder_for`).
use flock_core::{
    challenger::FsChallenger, pcs::MergedInnerOpen, pcs::ligerito::LigeritoProfile, test_rng::Rng,
};
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};

const DOMAIN: &[u8] = b"flock-fast-affine-test";
const N_BLOCKS: usize = 1 << 10;

fn inputs() -> Vec<Compression> {
    let mut rng = Rng::new(0xAFF1_E2E0);
    (0..N_BLOCKS)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

#[test]
fn fast_affine_union_roundtrip_and_rejections() {
    let affine = Blake3Setup::with_profile(N_BLOCKS, LigeritoProfile::FastAffine);
    assert_eq!(affine.pcs_params.m, 24, "the registered small affine shape");
    let blocks = inputs();
    let (proof, commitment, _) = affine.prove_fast(&blocks, &mut FsChallenger::new(DOMAIN));
    assert!(
        matches!(proof.pcs_open.inner, MergedInnerOpen::Affine(_)),
        "fast-affine must take the affine inner open"
    );
    affine
        .verify(&commitment, &proof, &mut FsChallenger::new(DOMAIN))
        .expect("fast-affine proof must verify");

    // A tampered affine grid rejects.
    let mut bad = proof.clone();
    let MergedInnerOpen::Affine(inner) = &mut bad.pcs_open.inner else {
        unreachable!()
    };
    inner.grids[0][1] += flock_core::field::F128::ONE;
    assert!(
        affine
            .verify(&commitment, &bad, &mut FsChallenger::new(DOMAIN))
            .is_err(),
        "tampered affine grid"
    );

    // The split verifier refuses the affine proof, and the affine verifier
    // refuses a split proof of the same statement.
    let fast = Blake3Setup::with_profile(N_BLOCKS, LigeritoProfile::Fast);
    assert!(
        fast.verify(&commitment, &proof, &mut FsChallenger::new(DOMAIN))
            .is_err(),
        "fast verifier on a fast-affine proof"
    );
    let (split_proof, split_commitment, _) =
        fast.prove_fast(&blocks, &mut FsChallenger::new(DOMAIN));
    assert!(matches!(
        split_proof.pcs_open.inner,
        MergedInnerOpen::Split(_)
    ));
    assert!(
        affine
            .verify(
                &split_commitment,
                &split_proof,
                &mut FsChallenger::new(DOMAIN)
            )
            .is_err(),
        "fast-affine verifier on a fast proof"
    );
    let mut relabeled = split_commitment.clone();
    relabeled.params.profile = LigeritoProfile::FastAffine;
    assert!(
        affine
            .verify(&relabeled, &split_proof, &mut FsChallenger::new(DOMAIN))
            .is_err(),
        "fast proof under a relabeled commitment"
    );
}

/// `slim-affine` at its one registered shape, the recursion tower's m = 29
/// outer envelope ((22,5), rate 1/4): a union prove whose merged open takes
/// the affine inner opening, verified, and refused by the `slim` verifier.
#[test]
fn slim_affine_union_roundtrip() {
    let n_blocks = 1 << 15;
    let affine = Blake3Setup::with_profile(n_blocks, LigeritoProfile::SlimAffine);
    assert_eq!(affine.pcs_params.m, 29, "the tower's outer envelope shape");
    let mut rng = Rng::new(0x51_1A_FF1E);
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            (cv, m, rng.next_u32() as u64, 64u32, 11u32)
        })
        .collect();
    let (proof, commitment, _) = affine.prove_fast(&blocks, &mut FsChallenger::new(DOMAIN));
    assert!(matches!(proof.pcs_open.inner, MergedInnerOpen::Affine(_)));
    affine
        .verify(&commitment, &proof, &mut FsChallenger::new(DOMAIN))
        .expect("slim-affine proof must verify");
    let slim = Blake3Setup::with_profile(n_blocks, LigeritoProfile::Slim);
    assert!(
        slim.verify(&commitment, &proof, &mut FsChallenger::new(DOMAIN))
            .is_err(),
        "slim verifier on a slim-affine proof"
    );
}
