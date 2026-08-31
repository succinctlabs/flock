//! kappa = 7 element tables — regression pins for the packed-word width.
//!
//! The 2b close-out session recorded "kappa=7 element tables still die
//! (column-split assert)" and capped the circuit gates at kappa 6. That
//! failure does NOT reproduce on the current tree — nor at the commit that
//! recorded it (`362761e`, retried 2026-08-02 with the residual/prefix/
//! suffix gates bumped to 7) — so it was most likely an artifact of the
//! mid-session state before the rs_eq_ind DeferredDense core fix
//! (`d49cc73`). These tests pin kappa=7 viability at every seam the cap
//! was protecting: element-only union, wide gates (>64 used columns),
//! schema words at columns >= 64, the wired circuit path, and the
//! cross-class crossing. All small shapes — they run un-ignored.

use flock_core::element_r1cs::ElementTableType;
use flock_core::pcs::ligerito::LigeritoProfile;
use prover::prove_fast_ligerito_union_circuit;
use prover::prove_fast_ligerito_union_mixed_class;
use sha2::SHA256_IV;
use sha2::build_block_r1cs;
use sha2::generate_witness_batch_major_partial;
use sha2::sha256_compress;
use std::array::from_fn;
use std::sync::Arc;
use verifier::verify_ligerito_union_circuit;
use verifier::verify_ligerito_union_mixed_class;

use flock_core::circuit::{Cell, Circuit};
use flock_core::element_r1cs::ElementTableBuilder;
use flock_core::field::gf2_128::F128;
use flock_core::schedule::IoWord;
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::PcsParams;
use flock_prover::prover::{self, UnionElementSlotInput};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier;

use flock_core::test_rng::Rng;
use flock_prover::prover::UnionSlotProverInput;
use flock_prover::r1cs_hashes::sha2;
const DOMAIN: &[u8] = b"flock-kappa7-probe";

const EL_A: usize = 0;
const EL_B: usize = 1;
const EL_C: usize = 2;

fn mult_ty(kappa: usize) -> TableType {
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0).free_wire(1).mult(2, 0, 1);
    TableType::element(Arc::new(b.build().expect("mult block is valid"))).with_io_schema(vec![
        IoWord::input(0),
        IoWord::input(1),
        IoWord::output(2),
    ])
}

/// A WIDE kappa=7 gate — `used` live columns (the residual gates' shape:
/// c approaches the column budget), a running product down the row.
fn wide_ty(kappa: usize, used: usize) -> Arc<ElementTableType> {
    assert!(used >= 3 && used <= 1 << kappa);
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0).free_wire(1);
    for c in 2..used {
        b.mult(c, c - 1, 0);
    }
    Arc::new(b.build().expect("wide block is valid"))
}

/// Element-only union with a wide kappa=7 table: >64 used columns.
#[test]
fn kappa7_wide_gate_union() {
    for (kappa, used) in [(7usize, 100usize), (7, 65), (7, 64), (7, 128)] {
        let nu = 8usize;
        let ty = wide_ty(kappa, used);
        let registry = Registry::new(vec![TableType::element(ty.clone())], nu);
        let n = 50usize;
        let mut rng = Rng::new(0x71DE_0001 ^ used as u64);
        let at = |c: usize, j: usize| (c << nu) + j;
        let mut z = vec![F128::ZERO; ty.width() << nu];
        for j in 0..n {
            let (a, b0) = (rng.f128(), rng.f128());
            z[at(0, j)] = a;
            z[at(1, j)] = b0;
            let mut prev = b0;
            for c in 2..used {
                prev *= a;
                z[at(c, j)] = prev;
            }
        }
        assert!(
            ty.satisfies(&z, nu, n),
            "wide witness must satisfy (used={used})"
        );

        let union = UnionInstance::new(&registry, vec![n]);
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        let zc = z.clone();
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prove_fast_ligerito_union_mixed_class(
            &union,
            &pcs_params,
            Vec::new(),
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&zc)
            })],
            &mut ch,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        verify_ligerito_union_mixed_class(&union, &[], &commitment, &proof, &pcs_params, &mut ch)
            .unwrap_or_else(|e| panic!("kappa=7 used={used} rejected: {e:?}"));
    }
}

/// Kappa=7 gate whose SCHEMA words sit at columns >= 64 (bit 7 of the
/// in-slot word coordinate set), wired through the circuit path.
#[test]
fn kappa7_high_column_schema_circuit() {
    let (nu, n) = (8usize, 20usize);
    let (ca, cb, cc) = (80usize, 81, 100);
    let mut b = ElementTableBuilder::new(7);
    b.free_wire(ca).free_wire(cb).mult(cc, ca, cb);
    let ty = Arc::new(b.build().expect("high-column block is valid"));
    let table = TableType::element(ty.clone()).with_io_schema(vec![
        IoWord::input(ca),
        IoWord::input(cb),
        IoWord::output(cc),
    ]);
    let registry = Registry::new(vec![table], nu);
    let ety = registry.types()[0].element_type().expect("element type");

    let mut rng = Rng::new(0xC4A1_0107);
    let seed = rng.f128();
    let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let at = |c: usize, j: usize| (c << nu) + j;
    let mut z = vec![F128::ZERO; ety.width() << nu];
    let mut acc = seed;
    for (j, &aj) in a.iter().enumerate() {
        z[at(ca, j)] = aj;
        z[at(cb, j)] = acc;
        z[at(cc, j)] = aj * acc;
        acc = aj * acc;
    }
    assert!(ety.satisfies(&z, nu, n), "witness must satisfy");
    let result = acc;

    let mut public = vec![seed];
    public.extend_from_slice(&a);
    public.push(result);
    const PUB: usize = 3;

    let mut wires = vec![vec![Cell::new(PUB, 0), Cell::new(EL_B, 0)]];
    for i in 0..n {
        wires.push(vec![Cell::new(PUB, 1 + i), Cell::new(EL_A, i)]);
    }
    for i in 0..n - 1 {
        wires.push(vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)]);
    }
    wires.push(vec![Cell::new(EL_C, n - 1), Cell::new(PUB, 1 + n)]);

    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let circuit = Circuit::new(&registry, vec![n], public.len(), wires).expect("valid");

    let zc = z.clone();
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&zc)
        })],
        &mut ch,
    );
    let mut ch = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("honest high-column kappa-7 chain verifies");
}

/// Cross-class at kappa = 7: a SHA-256 gate's output words feed the mult
/// gate, mirroring `cross_class_hash_into_mult` at the packed-word width.
#[test]
fn kappa7_cross_class_circuit() {
    fn pack_u32_words(u32s: &[u32]) -> Vec<F128> {
        u32s.chunks(4)
            .map(|c| {
                F128::new(
                    c[0] as u64 | ((c[1] as u64) << 32),
                    c[2] as u64 | ((c[3] as u64) << 32),
                )
            })
            .collect()
    }
    fn sha2_schema() -> Vec<IoWord> {
        vec![
            IoWord::input(0),
            IoWord::input(1),
            IoWord::input(4),
            IoWord::input(5),
            IoWord::input(6),
            IoWord::input(7),
            IoWord::output(2),
            IoWord::output(3),
        ]
    }
    const SHA_H0: usize = 0;
    const SHA_H1: usize = 1;
    const SHA_M0: usize = 2;
    const SHA_O0: usize = 6;
    const SHA_O1: usize = 7;

    let (nu, kappa) = (7usize, 7usize);
    let r1cs = build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            mult_ty(kappa),
            TableType::from_block_r1cs(&r1cs).with_io_schema(sha2_schema()),
        ],
        nu,
    );
    assert_eq!(registry.num_boolean(), 1);
    assert!(registry.types()[1].is_element());

    let mut rng = Rng::new(0xC205_0007);
    let m: [u32; 16] = from_fn(|_| rng.next_u64() as u32);
    let h_out = sha256_compress(&SHA256_IV, &m);
    let out_words = pack_u32_words(&h_out);
    let (o0, o1) = (out_words[0], out_words[1]);

    let mut public = pack_u32_words(&SHA256_IV);
    public.extend(pack_u32_words(&m));
    public.push(o0 * o1);

    const EL: usize = 8;
    const PUB: usize = 11;
    let wires = vec![
        vec![Cell::new(PUB, 0), Cell::new(SHA_H0, 0)],
        vec![Cell::new(PUB, 1), Cell::new(SHA_H1, 0)],
        vec![Cell::new(PUB, 2), Cell::new(SHA_M0, 0)],
        vec![Cell::new(PUB, 3), Cell::new(SHA_M0 + 1, 0)],
        vec![Cell::new(PUB, 4), Cell::new(SHA_M0 + 2, 0)],
        vec![Cell::new(PUB, 5), Cell::new(SHA_M0 + 3, 0)],
        vec![Cell::new(SHA_O0, 0), Cell::new(EL + EL_A, 0)],
        vec![Cell::new(SHA_O1, 0), Cell::new(EL + EL_B, 0)],
        vec![Cell::new(EL + EL_C, 0), Cell::new(PUB, 6)],
    ];

    let counts = vec![1usize, 1];
    let union = UnionInstance::new(&registry, counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let circuit = Circuit::new(&registry, counts, public.len(), wires).expect("valid");

    let el_ty = registry.types()[1].element_type().expect("element");
    let mut z = vec![F128::ZERO; el_ty.width() << nu];
    z[0 << nu] = o0;
    z[1 << nu] = o1;
    z[2 << nu] = o0 * o1;
    let circuit_lc = r1cs.csc_lincheck_circuit();

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, claims) = prove_fast_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(&[(SHA256_IV, m)], nu),
            circuit_lc,
        )],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&z)
        })],
        &mut ch,
    );
    assert!(claims.boolean.is_some() && claims.element.is_some());
    let mut ch = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &[circuit_lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("honest kappa-7 cross-class circuit verifies");
}

#[test]
fn kappa7_element_chain_circuit() {
    let (nu, kappa, n) = (8usize, 7usize, 20usize);
    let registry = Registry::new(vec![mult_ty(kappa)], nu);
    assert_eq!(registry.m_total(), 22);
    let ty = registry.types()[0].element_type().expect("element type");

    let mut rng = Rng::new(0xC4A1_0007);
    let seed = rng.f128();
    let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let at = |c: usize, j: usize| (c << nu) + j;
    let mut z = vec![F128::ZERO; ty.width() << nu];
    let mut b = seed;
    for (j, &aj) in a.iter().enumerate() {
        z[at(0, j)] = aj;
        z[at(1, j)] = b;
        z[at(2, j)] = aj * b;
        b = aj * b;
    }
    assert!(ty.satisfies(&z, nu, n), "witness must satisfy");
    let result = b;

    let mut public = vec![seed];
    public.extend_from_slice(&a);
    public.push(result);
    const PUB: usize = 3;

    let mut wires = vec![vec![Cell::new(PUB, 0), Cell::new(EL_B, 0)]];
    for i in 0..n {
        wires.push(vec![Cell::new(PUB, 1 + i), Cell::new(EL_A, i)]);
    }
    for i in 0..n - 1 {
        wires.push(vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)]);
    }
    wires.push(vec![Cell::new(EL_C, n - 1), Cell::new(PUB, 1 + n)]);

    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let circuit = Circuit::new(&registry, vec![n], public.len(), wires).expect("valid");

    let zc = z.clone();
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&zc)
        })],
        &mut ch,
    );
    let mut ch = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("honest kappa-7 element chain verifies");
}
