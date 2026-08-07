//! End-to-end union proofs over the composite Merkle-path table
//! (`r1cs_hashes::merkle_r1cs`) at the real depth 26.
//!
//! Everything here runs through the **walker** `LincheckCircuit` and empty
//! matrix stubs, so no test materializes the ~4.4 GB composite matrices.
//! Geometry at depth 26 (κ = 19), uniform capacity 2^ν:
//!
//! | ν | paths | dense words   | dense_m | union M |
//! |---|-------|---------------|---------|---------|
//! | 3 | 8     | 8·3237 = 25896| 22      | 22      |
//!
//! ν = 3 is the lincheck's floor (`n_outer ≥ 8`) and lands exactly on the
//! smallest embedded Ligerito config, so it is the cheapest real-depth proof.

use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::schedule::{Registry, TableType};
use flock_core::union::UnionInstance;
use flock_core::verifier;
use flock_prover::challenger::FsChallenger;
use flock_prover::mixed::{MerkleMixedCounts, MerkleMixedSetup, MixedRegistryId};
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::merkle_r1cs::MerkleWalkerCircuit;
use flock_prover::r1cs_hashes::merkle_r1cs::{
    MerkleTreeLayout, PathInput, SLOT_WORDS, blake3_spec, reference_root,
};
use std::array::from_fn;
use std::time::Instant;

const DOMAIN: &[u8] = b"flock-merkle-union-e2e-v0";
const DEPTH: usize = 26;
const NU: usize = 3;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn digest(&mut self) -> [u32; SLOT_WORDS] {
        from_fn(|_| self.next_u32())
    }
    fn path(&mut self, depth: usize) -> PathInput {
        let index =
            ((self.next_u32() as u64) << 32 | self.next_u32() as u64) & ((1u64 << depth) - 1);
        PathInput {
            leaf: self.digest(),
            index,
            siblings: (0..depth).map(|_| self.digest()).collect(),
        }
    }
}

/// Everything a Merkle-table union proof needs, built once.
struct Setup {
    layout: MerkleTreeLayout,
    walker: MerkleWalkerCircuit,
    registry: Registry,
}

impl Setup {
    fn new(depth: usize, nu: usize) -> Self {
        let layout = MerkleTreeLayout::new(depth, blake3_spec());
        // Stub matrices: the constraints live on the walker.
        let stub = layout.build_block_r1cs_stub(nu);
        let registry = Registry::new(vec![TableType::from_block_r1cs(&stub)], nu);
        Self {
            walker: layout.build_walker(),
            layout,
            registry,
        }
    }

    fn union(&self, n_paths: usize) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, vec![n_paths])
    }

    fn pcs_params(&self, union: &UnionInstance<'_>) -> PcsParams {
        PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(6),
            merkle_hash: Default::default(),
        }
    }
}

/// Geometry of the depth-26 single-type registry, before any proving.
#[test]
fn depth26_union_geometry() {
    let s = Setup::new(DEPTH, NU);
    assert_eq!(s.registry.num_types(), 1);
    assert_eq!(s.registry.types()[0].k_log, 19);
    assert_eq!(s.registry.types()[0].useful_bits, s.layout.useful_bits);
    assert_eq!(s.registry.types()[0].const_pin, Some(s.layout.const_pos()));
    // One type ⇒ no address-space rounding: M = nu + k_log exactly.
    assert_eq!(s.registry.m_total(), NU + 19);

    let union = s.union(1 << NU);
    assert_eq!(
        union.dense_words(),
        (1 << NU) * s.layout.useful_bits.div_ceil(128)
    );
    assert_eq!(union.dense_m(), 22, "lands on the smallest Ligerito config");
    assert_eq!(s.pcs_params(&union).m, 22);
}

/// **The milestone**: 8 depth-26 Merkle paths proved and verified in one
/// union proof, through the walker circuit and the merged transport.
#[test]
#[ignore] // Real-depth prove; run with `-- --ignored`.
fn depth26_roundtrip() {
    let s = Setup::new(DEPTH, NU);
    let n_paths = 1usize << NU;
    let mut rng = Rng::new(0x_26_00_A7_11);
    let paths: Vec<PathInput> = (0..n_paths).map(|_| rng.path(DEPTH)).collect();

    let union = s.union(n_paths);
    let pcs_params = s.pcs_params(&union);
    let t = Instant::now();
    let witness = s.layout.generate_witness_batch_major_partial(&paths, NU);
    let t_wit = t.elapsed();

    let t = Instant::now();
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, claim) = prover::prove_fast_ligerito_union(
        &union,
        &pcs_params,
        vec![UnionSlotProverInput::new(witness, &s.walker)],
        &mut ch_p,
    );
    let t_prove = t.elapsed();

    let t = Instant::now();
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verifier::verify_ligerito_union(
        &union,
        &[&s.walker],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch_v,
    )
    .expect("the depth-26 Merkle union proof must verify");
    let t_verify = t.elapsed();

    assert_eq!(claim.ab.value, claim_v.ab.value, "ab claim value");
    assert_eq!(claim.c.value, claim_v.c.value, "c claim value");

    println!(
        "depth {DEPTH}, {n_paths} paths (nu={NU}, M={}, dense_m={}): \
         witness {:.0} ms, prove {:.0} ms, verify {:.1} ms",
        s.registry.m_total(),
        pcs_params.m,
        t_wit.as_secs_f64() * 1e3,
        t_prove.as_secs_f64() * 1e3,
        t_verify.as_secs_f64() * 1e3,
    );

    // The roots the proof commits to are the honest ones.
    for (i, p) in paths.iter().enumerate() {
        let [z, _, _] = s.layout.build_witness_zab(p);
        assert_eq!(
            s.layout.read_root(&z),
            reference_root(&s.layout.spec, p),
            "path {i} root"
        );
    }
}

/// A partial slot: 5 of 8 rows declared, the rest dummies. This is the case
/// the count-derived lincheck pin target exercises — a dummy row carrying the
/// constant wire would be rejected.
#[test]
#[ignore] // Real-depth prove; run with `-- --ignored`.
fn depth26_partial_counts_roundtrip() {
    let s = Setup::new(DEPTH, NU);
    let mut rng = Rng::new(0x_2604_0071);
    for n_paths in [1usize, 5, 7] {
        let paths: Vec<PathInput> = (0..n_paths).map(|_| rng.path(DEPTH)).collect();
        let union = s.union(n_paths);
        let pcs_params = s.pcs_params(&union);
        let witness = s.layout.generate_witness_batch_major_partial(&paths, NU);

        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new(witness, &s.walker)],
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union(
            &union,
            &[&s.walker],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("{n_paths}-path proof rejected: {e:?}"));
    }
}

// ---------------------------------------------------------------------------
// The shipped Merkle + BLAKE3 tier
// ---------------------------------------------------------------------------

fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<blake3::Compression> {
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
            (cv, m, counter, 64u32, 11u32)
        })
        .collect()
}

/// **The mixed milestone**: depth-26 Merkle paths and loose BLAKE3
/// compressions in ONE proof, through the shipped tier id.
#[test]
#[ignore] // Real-depth prove; run with `-- --ignored`.
fn merkle_blake3_mixed_roundtrip() {
    let setup = MerkleMixedSetup::new(MixedRegistryId::MerkleBlake3Nu3);
    let mut rng = Rng::new(0x_1E_D0_44_21);

    for (n_merkle, n_blake3) in [(5usize, 6usize), (8, 8), (8, 0), (0, 8)] {
        let paths: Vec<PathInput> = (0..n_merkle).map(|_| rng.path(DEPTH)).collect();
        let blake3_inputs = random_blake3_inputs(&mut rng, n_blake3);
        let counts = MerkleMixedCounts {
            merkle: n_merkle,
            blake3: n_blake3,
        };

        let t = Instant::now();
        let mut ch_p = FsChallenger::new(DOMAIN);
        let (proof, commitment, claim) =
            setup.prove(&paths, &blake3_inputs, LigeritoProfile::Fast, &mut ch_p);
        let t_prove = t.elapsed();

        let t = Instant::now();
        let mut ch_v = FsChallenger::new(DOMAIN);
        let claim_v = setup
            .verify(counts, &commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| {
                panic!("mixed proof ({n_merkle} merkle, {n_blake3} blake3) rejected: {e:?}")
            });
        let t_verify = t.elapsed();

        assert_eq!(claim.ab.value, claim_v.ab.value);
        assert_eq!(claim.c.value, claim_v.c.value);
        println!(
            "tier {}: {n_merkle} merkle + {n_blake3} blake3 \
             (dense_m={}): prove {:.0} ms, verify {:.0} ms",
            setup.id.as_str(),
            setup.pcs_params(counts, LigeritoProfile::Fast).m,
            t_prove.as_secs_f64() * 1e3,
            t_verify.as_secs_f64() * 1e3,
        );
    }
}

/// The tier binds its counts vector: a proof of (5, 6) must not verify as
/// any other count pair.
#[test]
#[ignore] // Real-depth prove; run with `-- --ignored`.
fn merkle_blake3_mixed_wrong_counts_rejected() {
    let setup = MerkleMixedSetup::new(MixedRegistryId::MerkleBlake3Nu3);
    let mut rng = Rng::new(0x_5C_A7_11_09);
    let paths: Vec<PathInput> = (0..5).map(|_| rng.path(DEPTH)).collect();
    let blake3_inputs = random_blake3_inputs(&mut rng, 6);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) =
        setup.prove(&paths, &blake3_inputs, LigeritoProfile::Fast, &mut ch_p);

    for bad in [
        MerkleMixedCounts {
            merkle: 6,
            blake3: 6,
        },
        MerkleMixedCounts {
            merkle: 5,
            blake3: 7,
        },
        MerkleMixedCounts {
            merkle: 6,
            blake3: 5,
        },
    ] {
        let mut ch_v = FsChallenger::new(DOMAIN);
        assert!(
            setup.verify(bad, &commitment, &proof, &mut ch_v).is_err(),
            "a (5, 6) proof verified as {bad:?}"
        );
    }
}

/// A proof must not verify against a different declared count — the union
/// binds the counts vector into the transcript.
#[test]
#[ignore] // Real-depth prove; run with `-- --ignored`.
fn depth26_wrong_count_is_rejected() {
    let s = Setup::new(DEPTH, NU);
    let mut rng = Rng::new(0x_26_7A_00_5D);
    let paths: Vec<PathInput> = (0..5).map(|_| rng.path(DEPTH)).collect();
    let union = s.union(5);
    let pcs_params = s.pcs_params(&union);
    let witness = s.layout.generate_witness_batch_major_partial(&paths, NU);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union(
        &union,
        &pcs_params,
        vec![UnionSlotProverInput::new(witness, &s.walker)],
        &mut ch_p,
    );

    let lying = s.union(6);
    let mut ch_v = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union(
            &lying,
            &[&s.walker],
            &commitment,
            &proof,
            &s.pcs_params(&lying),
            &mut ch_v,
        )
        .is_err(),
        "a proof of 5 paths verified as 6"
    );
}
