//! Test fixture for the affine tower gadgets: a direct affine opening at the
//! registered m = 24 fast-affine shape `(17, 6)`, verified through the
//! RECORDING chained-BLAKE3 challenger, with its parsed tape and the core
//! verifier's own trace as the reference.

use flock_core::{
    challenger::FsChallenger,
    lincheck::build_eq_table,
    ntt::additive_ntt_f128::AdditiveNttF128,
    pcs::ligerito::{
        LigeritoProfile,
        affine::{
            AffineProof, AffineVerifyTrace, affine_ladder_for, prove_affine,
            verify_affine_l0_traced,
        },
        ligero_commit,
    },
};
use flock_field::F128;
use flock_transcript::transcript_record::RecordingChallenger;

use crate::tower::{
    fs_chain::flatten_ops,
    tape::{AffineLevel, AffineOodRec, parse_affine_direct},
};

pub(super) struct AffineFixture {
    pub(super) log_n: usize,
    /// The opening point (the basis is `eq(z, ·)`).
    pub(super) z: Vec<F128>,
    pub(super) target: F128,
    pub(super) proof: AffineProof,
    pub(super) trace: AffineVerifyTrace,
    /// Per level: fold count `k` and own column count `c`.
    pub(super) ks: Vec<usize>,
    pub(super) cs: Vec<usize>,
    pub(super) target_v: usize,
    pub(super) initial_ood: Vec<AffineOodRec>,
    pub(super) levels: Vec<AffineLevel>,
    pub(super) vals: Vec<F128>,
    pub(super) chals: Vec<F128>,
}

pub(super) fn rand_vec(len: usize, seed: u64) -> Vec<F128> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let lo = s;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            F128::new(lo, s)
        })
        .collect()
}

pub(super) fn affine_fixture(domain: &[u8]) -> AffineFixture {
    let (log_n, initial_k) = (17, 6);
    let (pc, vc) =
        affine_ladder_for(LigeritoProfile::FastAffine, log_n, initial_k).expect("m24 fast-affine");
    let witness = rand_vec(1 << log_n, 0xA11E);
    let z = rand_vec(log_n, 0x5EED);
    let basis = build_eq_table(&z);
    let target = witness
        .iter()
        .zip(&basis)
        .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);
    let log_cols = log_n - initial_k;
    let rate = pc.log_inv_rates[0];
    let ntt = AdditiveNttF128::standard(log_cols + rate);
    let l0 = ligero_commit(&witness, log_cols, initial_k, rate, &ntt, pc.merkle_hash);
    let mut ch = FsChallenger::with_chained_blake3(domain);
    let proof = prove_affine(&pc, witness, basis, target, &l0.mat, &l0.tree, &mut ch);
    let cap = l0.cap(pc.level_zero_cap_depth()).to_vec();
    let mut rec = RecordingChallenger::new(FsChallenger::with_chained_blake3(domain));
    let trace = verify_affine_l0_traced(
        &vc,
        &proof,
        log_n,
        target,
        &cap,
        (1 << initial_k, None),
        |rs, residual_log| {
            let prefix = z
                .iter()
                .zip(rs)
                .fold(F128::ONE, |acc, (&zi, &ri)| acc * (F128::ONE + zi + ri));
            let mut out = build_eq_table(&z[rs.len()..]);
            assert_eq!(out.len(), 1usize << residual_log);
            out.iter_mut().for_each(|v| *v *= prefix);
            out
        },
        &mut rec,
    )
    .expect("the recorded affine verify accepts");

    let mut ks = vec![pc.initial_k];
    ks.extend(&pc.recursive_ks);
    let mut cs = vec![log_cols];
    cs.extend(&pc.recursive_log_msg_cols);
    let yr_len = (1usize << cs[cs.len() - 1]) + ks[ks.len() - 1];
    let ops = flatten_ops(rec.shape().ops());
    let (target_v, initial_ood, levels) = parse_affine_direct(&ops, &ks, yr_len);
    assert_eq!(trace.queries.len(), levels.len());
    AffineFixture {
        log_n,
        z,
        target,
        proof,
        trace,
        ks,
        cs,
        target_v,
        initial_ood,
        levels,
        vals: rec.values().to_vec(),
        chals: rec.challenges().to_vec(),
    }
}
