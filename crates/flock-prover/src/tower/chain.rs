use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use flock_core::{
    circuit::{
        Circuit, SigmaAssertion, WiringProof,
        builder::{BuiltCircuit, CircuitShape, SlotId},
    },
    element_r1cs::union::Proof as ElementProof,
    lincheck::{LincheckCircuit, LincheckProof},
    pcs::{MergedOpenProof, commit::Commitment},
    proof::{R1csProofCircuitMerged, R1csProofCircuitMergedAg, UnionClassClaims},
    verifier::{DeferredMatrixWork, FlockVerifyError},
};
use flock_hash::blake3_compress;
use flock_transcript::challenger::Challenger;
#[cfg(test)]
use {
    crate::{
        r1cs_hashes::blake3::{Compression, build_block_r1cs},
        tower::{
            ChildSlots, ChildTape, SLOT_WORDS, check_child_region, emit_child_region,
            gates_blake3::Rng, test_config,
        },
    },
    flock_core::{
        circuit::{WiringError, builder::CircuitBuilder},
        pcs::ligerito::LigeritoProfile,
        product_gkr::ProductGkrError,
    },
    std::{any::Any, array::from_fn},
};

#[cfg(target_arch = "aarch64")]
use crate::prover::prove_fast_ligerito_union_circuit_ag;
use crate::{
    prover::prove_fast_ligerito_union_circuit,
    r1cs_hashes::blake3::generate_witness_batch_major_partial,
    tower::{
        Blake3Gate, CHUNK_END, CHUNK_START, DOMAIN, F128, FsChallenger, HashKind, IV, Online,
        PcsParams, ROOT, ShapeBuilder, TowerConfig, UnionInstance, UnionSlotProverInput, Wire,
        chain_blake_r1cs, leaf_zc_ag, pack_params, pack4, pack8, pcs_batch_for, steady_reps,
    },
    verifier::{
        verify_ligerito_union_circuit, verify_ligerito_union_circuit_ag,
        verify_ligerito_union_circuit_ag_deferred, verify_ligerito_union_circuit_deferred,
    },
};

/// A circuit-union proof by boolean-zerocheck FLAVOR — parallel arms so
/// the chain leaf and envelope outers. Both forms share all other regions.
#[derive(serde::Serialize)]
pub(super) enum MixedProof {
    Rs(R1csProofCircuitMerged),
    /// Constructed only where the AG round-1 prover kernel exists
    /// (aarch64); the consuming arms are arch-independent.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Ag(R1csProofCircuitMergedAg),
}

impl MixedProof {
    pub(super) fn wiring(&self) -> &WiringProof {
        match self {
            MixedProof::Rs(p) => &p.wiring,
            MixedProof::Ag(p) => &p.wiring,
        }
    }
    pub(super) fn pcs_open(&self) -> &MergedOpenProof {
        match self {
            MixedProof::Rs(p) => &p.pcs_open,
            MixedProof::Ag(p) => &p.pcs_open,
        }
    }
    pub(super) fn element(&self) -> Option<&ElementProof> {
        match self {
            MixedProof::Rs(p) => p.element.as_ref(),
            MixedProof::Ag(p) => p.element.as_ref(),
        }
    }
    /// The boolean LINCHECK sub-proof — flavor-shared (both boolean proof
    /// structs carry it verbatim; only round 1 differs).
    pub(super) fn boolean_lincheck(&self) -> &LincheckProof {
        match self {
            MixedProof::Rs(p) => &p.boolean.as_ref().expect("boolean side present").lincheck,
            MixedProof::Ag(p) => &p.boolean.as_ref().expect("boolean side present").lincheck,
        }
    }
    /// The plain (assertions-discharged) circuit verify, flavor-dispatched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_circuit<Ch: Challenger>(
        &self,
        union: &UnionInstance<'_>,
        circuit: &Circuit,
        public: &[F128],
        lcs: &[&dyn LincheckCircuit],
        commitment: &Commitment,
        pcs: &PcsParams,
        ch: &mut Ch,
    ) -> Result<UnionClassClaims, FlockVerifyError> {
        match self {
            MixedProof::Rs(p) => {
                verify_ligerito_union_circuit(union, circuit, public, lcs, commitment, p, pcs, ch)
            }
            MixedProof::Ag(p) => verify_ligerito_union_circuit_ag(
                union, circuit, public, lcs, commitment, p, pcs, ch,
            ),
        }
    }
    /// The DEFERRED circuit verify (assertions returned, not discharged),
    /// flavor-dispatched — what the recursion tapes record.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_circuit_deferred<Ch: Challenger>(
        &self,
        union: &UnionInstance<'_>,
        circuit: &Circuit,
        public: &[F128],
        lcs: &[&dyn LincheckCircuit],
        commitment: &Commitment,
        pcs: &PcsParams,
        ch: &mut Ch,
    ) -> Result<(UnionClassClaims, DeferredMatrixWork, SigmaAssertion), FlockVerifyError> {
        match self {
            MixedProof::Rs(p) => verify_ligerito_union_circuit_deferred(
                union, circuit, public, lcs, commitment, p, pcs, ch,
            ),
            MixedProof::Ag(p) => verify_ligerito_union_circuit_ag_deferred(
                union, circuit, public, lcs, commitment, p, pcs, ch,
            ),
        }
    }
}

/// A mixed circuit proof and its deferred verification data.
pub(super) struct MixedInner {
    pub(super) nu: usize,
    pub(super) built: BuiltCircuit,
    pub(super) proof: MixedProof,
    pub(super) commitment: Commitment,
    pub(super) pcs: PcsParams,
    pub(super) work: DeferredMatrixWork,
    pub(super) sigma: SigmaAssertion,
}

/// Check a single-slot Boolean union whose rows form a BLAKE3 hash chain.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn chain_probe_boolean_only_wired_union() {
    let n_blocks = 256usize;
    let nu = 8usize;
    let mut rng = Rng(0xC4A1_0001);
    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let iv = pack8(&IV);
    let mut cv = [b.public_value(iv[0]), b.public_value(iv[1])];
    for i in 0..n_blocks {
        let m: [u32; 16] = from_fn(|_| rng.next_u32());
        let mut hash_in = vec![cv[0], cv[1]];
        for j in 0..4 {
            hash_in.push(b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())));
        }
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == n_blocks {
            flags |= CHUNK_END;
        }
        hash_in.push(b.public_value(pack_params(0, 64, flags)));
        let out = b.gate(hash, &hash_in);
        cv = [out[0], out[1]];
    }
    b.publish(cv[0]);
    b.publish(cv[1]);
    let built = b.finish().expect("the chain circuit builds");

    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    assert!(!union.has_element(), "one boolean slot, no element class");
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: pcs_batch_for(&union, LigeritoProfile::Fast),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(pcs_batch_for(&union, LigeritoProfile::Fast)),
        merkle_hash: HashKind::Blake3,
    };
    let blake_r1cs = build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let prove = |rows: &[Compression]| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        prove_fast_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(rows, nu),
                blake_lc,
            )],
            Vec::new(),
            &mut ch,
        )
    };
    let (proof, commitment, _) = prove(built.rows::<Blake3Gate>(hash));

    // The deferred verify — what a first-level node runs per chain child.
    let lcs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (_claims, work, sigma) = verify_ligerito_union_circuit_deferred(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the deferred verify accepts an honest boolean-only chain");
    let matrix = work
        .boolean
        .expect("the b3 slot yields boolean matrix work");
    assert!(work.element.is_none(), "no element class, no element work");
    matrix
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges against the b3 matrices");
    assert!(
        sigma.check(&built.shape.circuit),
        "the sigma assertion discharges against the chain's own sigma table"
    );

    // The chain-link tamper: row 17 re-witnessed from a cv one bit off.
    // Every gate still satisfies the b3 relation on its own row; only the
    // copy constraints around row 17 break — the wiring product must be
    // what catches it.
    let mut bad: Vec<Compression> = built.rows::<Blake3Gate>(hash).to_vec();
    bad[17].0[0] ^= 1;
    let (p, cm, _) = prove(&bad);
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    assert!(
        matches!(
            verify_ligerito_union_circuit(
                &union,
                &built.shape.circuit,
                &built.witness.public,
                &lcs,
                &cm,
                &p,
                &pcs_params,
                &mut ch,
            ),
            Err(FlockVerifyError::Wiring(WiringError::Gkr(
                ProductGkrError::ProductMismatch
            )))
        ),
        "a broken chain link must die on the wiring product"
    );
}

// ---------------------------------------------------------------------------
// The hash-chain PoC's LEAF (task 2): the message-chain proof.
//
// h_{i+1} = compress(IV, h_i, counter = 0, block_len = 64,
//                    CHUNK_START | CHUNK_END | ROOT)
//
// — the full 64-byte output fed back as the next MESSAGE block (Ron's call:
// no truncation to a 32-byte cv between steps), cv pinned to the IV, the
// single-block-root flag flavor, so one step reads as a standalone
// blake3-of-64-bytes. One b3 slot, rows chained out(i) → m(i+1) by copy
// constraints (task 1 pinned that boolean-only unions take wiring); the
// statement is 11 words: iv (2) + params (1) + h_start (4) declared, h_end
// (4) published last — DECLARATION-ordered, so the public tail IS h_end.

/// The chain statement's flag word: a standalone single-block root.
pub(super) const CHAIN_FLAGS: u32 = CHUNK_START | CHUNK_END | ROOT;

/// The native reference: `n_blocks` message-chain steps from `h_start`.
pub(super) fn native_chain(h_start: &[u32; 16], n_blocks: usize) -> [u32; 16] {
    let mut h = *h_start;
    for _ in 0..n_blocks {
        h = blake3_compress(&IV, &h, 0, 64, CHAIN_FLAGS);
    }
    h
}

/// The chain circuit alone (shared by the honest builder and the tamper
/// legs): one b3 slot, message-chain wiring, the 11-word statement.
/// The chain circuit's SHAPE, separated from the statement it runs on.
/// The shape does not depend on `h_start` — that is the digest-determinism
/// pin — so a chain prover builds this ONCE and pays only the per-segment
/// walk afterwards. The split is also what makes a leaf's ONLINE cost
/// measurable: the walk is per-statement (it computes the chain and
/// materialises the rows), the shape is not.
#[derive(Clone)]
pub(super) struct ChainShape {
    pub(super) shape: CircuitShape,
    pub(super) hash: SlotId,
    pub(super) nu: usize,
}

/// The chain SHAPE per n_blocks, cached process-wide: the emission+finish
/// (~1.4 s at m32) is statement-independent — that is the digest pin — so
/// the tower's material proofs CLONE the cached shape (Registry + Circuit
/// memcpy, ~an order of magnitude cheaper) instead of re-emitting it.
/// `build_chain_proof`'s setup_ms honestly reflects whichever it paid.
pub(super) fn chain_shape_cached(n_blocks: usize) -> Arc<ChainShape> {
    type Cache = Mutex<Vec<(usize, Arc<ChainShape>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, s)) = g.iter().find(|(k, _)| *k == n_blocks) {
        return s.clone();
    }
    let s = Arc::new(build_chain_shape(n_blocks));
    g.push((n_blocks, s.clone()));
    s
}

pub(super) fn build_chain_shape(n_blocks: usize) -> ChainShape {
    let nu = n_blocks.trailing_zeros() as usize;
    assert_eq!(1usize << nu, n_blocks, "block count is a power of two");
    let mut sb = ShapeBuilder::new(nu);
    let hash = sb.slot(Blake3Gate { nu });
    let cv = [sb.public_input(), sb.public_input()];
    let params = sb.public_input();
    let mut m: Vec<Wire> = (0..4).map(|_| sb.public_input()).collect();
    let mut out = Vec::new();
    for _ in 0..n_blocks {
        let mut hash_in = vec![cv[0], cv[1]];
        hash_in.extend_from_slice(&m);
        hash_in.push(params);
        out = sb.gate(hash, &hash_in);
        m = out.clone();
    }
    for w in &out {
        sb.publish(*w);
    }
    ChainShape {
        shape: sb.finish().expect("the chain circuit builds"),
        hash,
        nu,
    }
}

/// The statement a chain shape runs on, in declaration order: the IV pair,
/// the params word, then `h_start`. (`h_end` is PUBLISHED, so it is the
/// walk's output, not an input.)
pub(super) fn chain_vals(h_start: &[u32; 16]) -> Vec<F128> {
    let iv = pack8(&IV);
    let mut v = vec![iv[0], iv[1], pack_params(0, 64, CHAIN_FLAGS)];
    v.extend((0..4).map(|j| pack4(h_start[4 * j..4 * j + 4].try_into().unwrap())));
    v
}

#[cfg(test)]
pub(super) fn build_chain_circuit(h_start: &[u32; 16], n_blocks: usize) -> (BuiltCircuit, SlotId) {
    let cs = build_chain_shape(n_blocks);
    let witness = cs.shape.run(&chain_vals(h_start), &[]);
    (
        BuiltCircuit {
            shape: cs.shape,
            witness,
        },
        cs.hash,
    )
}

/// The chain-PoC leaf, end to end: FAST profile (the B-fast decision),
/// BLAKE3 for Merkle and FS (recursable), proven over the circuit path
/// with NO element slots, deferred-verified with both assertion families
/// discharged, and h_end cross-checked against the native chain. The
/// [`MixedInner`] embedding is deliberate: a chain proof is a circuit
/// proof, so [`ChildTape`] consumes it directly (element side `None`).
pub struct ChainProof {
    pub(super) inner: MixedInner,
    pub(super) h_start: [u32; 16],
    pub(super) h_end: [u32; 16],
    /// What the leaf cost, split SETUP vs ONLINE — see [`Online`]. The
    /// LAST online iteration under steady repetition. Read by the in-file
    /// `#[test]` benches only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) t: Online,
    /// One record per online iteration (1 + steady_reps of them).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) onlines: Vec<Online>,
}

/// A chain leaf. The SHAPE build is per-shape setup (statement-independent
/// — the digest pin), the WALK is per-statement and is the chain compute
/// itself, so it is reported apart from the proving phases.
pub fn build_chain_proof(cfg: TowerConfig, h_start: [u32; 16], n_blocks: usize) -> ChainProof {
    let t_shape = Instant::now();
    let cs: ChainShape = chain_shape_cached(n_blocks).as_ref().clone();
    let shape_ms = t_shape.elapsed().as_secs_f64() * 1e3;
    let (nu, hash) = (cs.nu, cs.hash);
    let t_setup = Instant::now();
    let blake_r1cs = chain_blake_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let setup_ms = shape_ms + t_setup.elapsed().as_secs_f64() * 1e3;

    // ONLINE, `1 + steady_reps()` iterations over the ONE shape: walk (the
    // chain compute itself), witgen, prove. Identical inputs, so every
    // iteration's outputs match and the last one ships.
    let reps = 1 + steady_reps();
    let mut onlines: Vec<Online> = Vec::with_capacity(reps);
    let mut fin = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let witness = cs.shape.run(&chain_vals(&h_start), &[]);
        let walk_ms = t0.elapsed().as_secs_f64() * 1e3;
        let union = UnionInstance::new(&cs.shape.registry, cs.shape.counts.clone());
        assert!(!union.has_element(), "a chain proof is boolean-only");
        let pcs_params = PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            // The chain leaf's WORKLOAD inner: the tower's security level,
            // rate 1/2 (a 100-bit recursion carries a 100-bit leaf). The
            // batch is keyed by the SAME profile as the params — the old
            // Fast-keyed batch only worked because the Fast twins share
            // initial_k at these m's.
            profile: cfg.leaf_profile(),
            log_batch_size: pcs_batch_for(&union, cfg.leaf_profile()),
            num_lanes: union.commit_lanes(pcs_batch_for(&union, cfg.leaf_profile())),
            merkle_hash: HashKind::Blake3,
        };
        let t1 = Instant::now();
        let wit = generate_witness_batch_major_partial(witness.rows::<Blake3Gate>(hash), nu);
        let witgen_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (proof, commitment) = if leaf_zc_ag() {
            #[cfg(target_arch = "aarch64")]
            {
                let (p, c, _) = prove_fast_ligerito_union_circuit_ag(
                    &union,
                    &cs.shape.circuit,
                    &witness.public,
                    &pcs_params,
                    vec![UnionSlotProverInput::new(wit, blake_lc)],
                    Vec::new(),
                    &mut ch,
                );
                (MixedProof::Ag(p), c)
            }
            #[cfg(not(target_arch = "aarch64"))]
            unreachable!("leaf_zc_ag() is false off aarch64")
        } else {
            let (p, c, _) = prove_fast_ligerito_union_circuit(
                &union,
                &cs.shape.circuit,
                &witness.public,
                &pcs_params,
                vec![UnionSlotProverInput::new(wit, blake_lc)],
                Vec::new(),
                &mut ch,
            );
            (MixedProof::Rs(p), c)
        };
        let prove_ms = t2.elapsed().as_secs_f64() * 1e3;
        // `t0` opened before the walk, so this is the whole online span in
        // ONE timer — the honest per-leaf number, against which the phase
        // sum is only a lower bound.
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        onlines.push(Online {
            setup_ms,
            walk_ms,
            witgen_ms,
            prove_ms,
            wall_ms,
            ..Online::default()
        });
        fin = Some((witness, proof, commitment, pcs_params));
    }
    let (witness, proof, commitment, pcs_params) = fin.expect("one online iteration at least");
    let built = BuiltCircuit {
        shape: cs.shape,
        witness,
    };
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());

    let lcs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (_claims, work, sigma) = match &proof {
        MixedProof::Rs(p) => verify_ligerito_union_circuit_deferred(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &commitment,
            p,
            &pcs_params,
            &mut ch,
        ),
        MixedProof::Ag(p) => verify_ligerito_union_circuit_ag_deferred(
            &union,
            &built.shape.circuit,
            &built.witness.public,
            &lcs,
            &commitment,
            p,
            &pcs_params,
            &mut ch,
        ),
    }
    .expect("the deferred verify accepts an honest chain proof");
    work.boolean
        .as_ref()
        .expect("the b3 slot yields boolean matrix work")
        .check(&union, &lcs)
        .expect("the boolean matrix work discharges");
    assert!(work.element.is_none(), "no element class, no element work");
    assert!(sigma.check(&built.shape.circuit), "sigma discharges");

    // The statement is bound end to end: publics[3..7] are h_start, the
    // published tail is h_end, and h_end equals the native chain.
    let h_end = native_chain(&h_start, n_blocks);
    let public = &built.witness.public;
    for j in 0..4 {
        assert_eq!(
            public[3 + j],
            pack4(h_start[4 * j..4 * j + 4].try_into().unwrap()),
            "public word {} is h_start[{}]",
            3 + j,
            j
        );
        assert_eq!(
            public[public.len() - 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "the published tail is the native h_end"
        );
    }

    ChainProof {
        inner: MixedInner {
            nu,
            built,
            proof,
            commitment,
            pcs: pcs_params,
            work,
            sigma,
        },
        h_start,
        h_end,
        t: *onlines.last().expect("one online iteration"),
        onlines,
    }
}

/// **Phase B slice 1 (docs/ag-recursion-plan.md): the REAL chain-leaf shape
/// through the AG-skip circuit entries.** Same cached chain shape, same
/// statement, same leaf profile — prove with
/// `prove_fast_ligerito_union_circuit_ag`, deferred-verify with the AG twin,
/// discharge both assertion families, pin h_end against the native chain,
/// reject the fused-nonce tampers, and print the same-shape RS vs AG leaf
/// prove times (the workload number the AG leaf exists for). No tower
/// plumbing is touched yet — `MixedInner`/`ChildTape` stay RS until the FL
/// walker grows its AG arm.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn chain_leaf_ag_roundtrip() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_00A6);
    let h_start: [u32; 16] = from_fn(|_| rng.next_u32());

    let cs: ChainShape = chain_shape_cached(n_blocks).as_ref().clone();
    let (nu, hash) = (cs.nu, cs.hash);
    let blake_r1cs = chain_blake_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();

    let witness = cs.shape.run(&chain_vals(&h_start), &[]);
    let union = UnionInstance::new(&cs.shape.registry, cs.shape.counts.clone());
    assert!(!union.has_element(), "a chain proof is boolean-only");
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        profile: cfg.leaf_profile(),
        log_batch_size: pcs_batch_for(&union, cfg.leaf_profile()),
        num_lanes: union.commit_lanes(pcs_batch_for(&union, cfg.leaf_profile())),
        merkle_hash: HashKind::Blake3,
    };
    let wit_rows = witness.rows::<Blake3Gate>(hash);

    // Same-shape RS baseline (the flavor delta, not a cross-shape number).
    let t0 = Instant::now();
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (_rs_proof, _, _) = prove_fast_ligerito_union_circuit(
        &union,
        &cs.shape.circuit,
        &witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(wit_rows, nu),
            blake_lc,
        )],
        Vec::new(),
        &mut ch,
    );
    let rs_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit_ag(
        &union,
        &cs.shape.circuit,
        &witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(wit_rows, nu),
            blake_lc,
        )],
        Vec::new(),
        &mut ch,
    );
    let ag_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[chain-leaf] prove RS {rs_ms:7.2} ms vs AG {ag_ms:7.2} ms (same shape, m = {})",
        union.dense_m()
    );

    let lcs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let verify = |p: &R1csProofCircuitMergedAg| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        verify_ligerito_union_circuit_ag_deferred(
            &union,
            &cs.shape.circuit,
            &witness.public,
            &lcs,
            &commitment,
            p,
            &pcs_params,
            &mut ch,
        )
    };
    let (_claims, work, sigma) = verify(&proof).expect("deferred AG chain leaf verifies");
    work.boolean
        .as_ref()
        .expect("boolean matrix work")
        .check(&union, &lcs)
        .expect("boolean matrix work discharges");
    assert!(work.element.is_none(), "no element class, no element work");
    assert!(sigma.check(&cs.shape.circuit), "sigma discharges");

    // The statement is bound end to end: the published tail is h_end.
    let h_end = native_chain(&h_start, n_blocks);
    let public = &witness.public;
    for j in 0..4 {
        assert_eq!(
            public[public.len() - 4 + j],
            pack4(h_end[4 * j..4 * j + 4].try_into().unwrap()),
            "the published tail is the native h_end"
        );
    }

    // Fused-nonce tampers on the REAL chain shape.
    let mut bad = proof.clone();
    let n = &mut bad.boolean.as_mut().expect("boolean").ag.r1_nonce;
    *n = n.wrapping_add(1);
    assert!(verify(&bad).is_err(), "tampered fused r1 nonce accepted");
    let mut bad = proof.clone();
    *bad.boolean
        .as_mut()
        .expect("boolean")
        .lincheck
        .grinding_nonces
        .last_mut()
        .expect("fused skip nonce") ^= 1;
    assert!(verify(&bad).is_err(), "tampered fused skip nonce accepted");
}

/// **Task 2's pin: the message-chain leaf, honest + the tamper matrix.**
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn chain_proof_message_chain_roundtrip_and_tampers() {
    let cfg = test_config();
    let n_blocks = 256usize;
    let mut rng = Rng(0xC4A1_0002);
    let h_start: [u32; 16] = from_fn(|_| rng.next_u32());

    // Honest: build_chain_proof internally deferred-verifies, discharges
    // both assertion families and cross-checks h_end against the native
    // chain. Determinism of the statement: a second build from the same
    // h_start yields the same h_end.
    let cp = build_chain_proof(cfg, h_start, n_blocks);
    assert_eq!(cp.h_end, native_chain(&h_start, n_blocks));
    assert_eq!(cp.inner.nu, 8);

    // The tamper legs run on a fresh circuit build (the honest one's rows,
    // modified), against the PLAIN verifier so every check is in force.
    let (built, hash) = build_chain_circuit(&h_start, n_blocks);
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let blake_r1cs = build_block_r1cs(cp.inner.nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let lcs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
    let prove = |rows: &[Compression], public: &[F128]| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        prove_fast_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            public,
            &cp.inner.pcs,
            vec![UnionSlotProverInput::new(
                generate_witness_batch_major_partial(rows, cp.inner.nu),
                blake_lc,
            )],
            Vec::new(),
            &mut ch,
        )
    };
    let verify = |public: &[F128], cm: &Commitment, p: &MixedProof| {
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        match p {
            MixedProof::Rs(p) => verify_ligerito_union_circuit(
                &union,
                &built.shape.circuit,
                public,
                &lcs,
                cm,
                p,
                &cp.inner.pcs,
                &mut ch,
            ),
            MixedProof::Ag(p) => verify_ligerito_union_circuit_ag(
                &union,
                &built.shape.circuit,
                public,
                &lcs,
                cm,
                p,
                &cp.inner.pcs,
                &mut ch,
            ),
        }
    };

    // (a) A broken chain link: row 100 re-witnessed from a message one bit
    //     off. Its own b3 relation holds; the copy constraint out(99) ==
    //     m(100) breaks, and the wiring product is what must catch it.
    {
        let mut bad: Vec<Compression> = built.rows::<Blake3Gate>(hash).to_vec();
        bad[100].1[0] ^= 1;
        let (p, cm, _) = prove(&bad, &built.witness.public);
        assert!(
            matches!(
                verify(&built.witness.public, &cm, &MixedProof::Rs(p)),
                Err(FlockVerifyError::Wiring(WiringError::Gkr(
                    ProductGkrError::ProductMismatch
                )))
            ),
            "a broken chain link must die on the wiring product"
        );
    }

    // (b) A tampered STATEMENT word — h_start (public 3) and h_end (the
    //     tail): the honest proof must not verify against it, and a prover
    //     honestly proving the tampered statement must be rejected too.
    let plen = built.witness.public.len();
    for i in [3usize, plen - 1] {
        let mut bad = built.witness.public.clone();
        bad[i] += F128::ONE;
        assert!(
            verify(&bad, &cp.inner.commitment, &cp.inner.proof).is_err(),
            "statement word {i} must be bound to the transcript"
        );
        let (p, cm, _) = prove(built.rows::<Blake3Gate>(hash), &bad);
        assert!(
            verify(&bad, &cm, &MixedProof::Rs(p)).is_err(),
            "statement word {i} must be enforced by the wiring"
        );
    }

    // (c') see chain_tape_regions_pinned for the tape-side continuation.
    // (c) Shape determinism — the foldability key. The chain circuit is
    //     h_start-INDEPENDENT (h_start only moves public VALUES), so every
    //     segment of a long chain proves against ONE circuit digest, and
    //     the accumulator folds their assertions under one sigma key.
    assert_eq!(cp.h_start, h_start);
    assert_eq!(
        cp.inner.built.shape.circuit.digest(),
        built.shape.circuit.digest(),
        "two builds from the same h_start agree"
    );
    let other_start: [u32; 16] = from_fn(|_| rng.next_u32());
    let (other, _) = build_chain_circuit(&other_start, n_blocks);
    assert_eq!(
        cp.inner.built.shape.circuit.digest(),
        other.shape.circuit.digest(),
        "a DIFFERENT segment's chain circuit is digest-equal"
    );
    // The deferred work/sigma are verifier-exported references: both must
    // discharge against the REBUILT circuit too (same digest, same tables).
    cp.inner
        .work
        .boolean
        .as_ref()
        .expect("boolean matrix work travels with the chain proof")
        .check(&union, &lcs)
        .expect("the exported matrix work discharges against the rebuild");
    assert!(
        cp.inner.sigma.check(&built.shape.circuit),
        "the exported sigma discharges against the rebuild's sigma table"
    );
}

/// **Task 3: the chain tape.** [`ChildTape::new`] — the SAME constructor the
/// merge machinery instantiates per mixed child — walks the hash-chain
/// leaf's tape with the element side `None`. Its class-agnostic pins all
/// re-assert on the boolean-only shape: the duplex chain trace, the GKR
/// walk + masked input checks, rs×2, the pd census, the R=2+P schedule
/// replaying to anchor.v, the W rounds, the stratified query geometry, the
/// spine/residual natives, the recombination, and the full anchor-expect
/// replica. This test adds the chain-shape facts on top.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
pub(super) fn chain_tape_regions_pinned() {
    let cfg = test_config();
    let mut rng = Rng(0xC4A1_0003);
    let h_start: [u32; 16] = from_fn(|_| rng.next_u32());
    let n_blocks = 256usize;
    let cp = build_chain_proof(cfg, h_start, n_blocks);
    let ct = ChildTape::new(&cp.inner, DOMAIN);

    // The boolean-only shape facts.
    assert!(ct.el.is_none(), "no element PIOP region on a chain tape");
    assert!(ct.el_assert.is_none(), "no element assertion travels");
    assert_eq!(
        ct.n_pd,
        cp.inner.proof.wiring().gather.len(),
        "the pd claims are the wiring gathers ONLY"
    );
    assert!(ct.n_p > 0, "the gathers form scalar groups");
    assert_eq!(
        ct.sigma_native.value, cp.inner.sigma.value,
        "the tape's sigma reference is the deferred verify's"
    );
    assert_eq!(
        ct.bool_assert.target,
        cp.inner.work.boolean.as_ref().expect("boolean work").target,
        "the tape's boolean reference is the deferred verify's"
    );

    let union = UnionInstance::new(
        &cp.inner.built.shape.registry,
        cp.inner.built.shape.counts.clone(),
    );
    println!(
        "\nCHAIN TAPE (boolean-only wired leaf, message-chain)\n  \
         inner: nu {} | dense_m {} | pd claims {} (ALL gathers) | P {} | mu {}\n  \
         GKR layers {} | b3 rows (tape model) {} | L0 lanes {} x {} words\n",
        cp.inner.nu,
        union.dense_m(),
        ct.n_pd,
        ct.n_p,
        ct.mu_i,
        cp.inner.proof.wiring().gkr.layers.len(),
        ct.b3_rows,
        ct.geo[0].lanes,
        ct.geo[0].row_words,
    );
}

/// Bisect probe: ONE chain child region alone — emit, run, check.
#[test]
#[ignore]
pub(super) fn chain_child_region_emits_alone() {
    let cfg = test_config();
    let mut rng = Rng(0xC4A1_0005);
    let h_start: [u32; 16] = from_fn(|_| rng.next_u32());
    let cp = build_chain_proof(cfg, h_start, 256);
    let ct = ChildTape::new(&cp.inner, DOMAIN);
    let nu2 = (ct.b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let mut sb = ShapeBuilder::new(nu2);
    let mut cs = ChildSlots::new(&mut sb, nu2, ct.spread_w);
    let mut vals: Vec<F128> = Vec::new();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut consts: Vec<(F128, Wire)> = Vec::new();
    let b3_slot = cs.q.b3;
    let region = emit_child_region(
        &mut sb,
        &mut cs,
        b3_slot,
        &ct,
        &mut vals,
        &mut hints,
        &mut consts,
    );
    let shape2 = sb.finish().expect("the chain child circuit builds");
    let hint_refs: Vec<&(dyn Any + Sync)> = hints.iter().map(|h| h as &(dyn Any + Sync)).collect();
    let built2 = shape2.run(&vals, &hint_refs);
    let consumed = check_child_region(&built2.public, &ct, &region);
    assert_eq!(
        region.pub_base + consumed,
        built2.public.len(),
        "the region's publics are the whole tail"
    );
}
