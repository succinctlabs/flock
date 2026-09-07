//! M2 oracles for the union-column lincheck (`lincheck::prove_union_capture_z_vec`
//! / `lincheck::verify_union`).
//!
//! 1. **T = 1 degeneration**: on a single-type registry the union-column
//!    lincheck IS today's lincheck — byte-identical proof messages, claims,
//!    and captured fold (the heavy BLAKE3/SHA-256 instances of this oracle
//!    run through the full union prove entry in flock-prover's ignored
//!    `tests/union_roundtrip.rs`).
//! 2. **T = 2 vs brute force**: a synthetic two-type registry small enough
//!    for brute-force MLEs — union zerocheck, then the union-column
//!    lincheck, with (a) the initial claim, (b) the final witness claim,
//!    and (c) the verifier's closed-form Comb-hat collapse each
//!    cross-checked against dense recomputations, plus tamper rejection
//!    (corrupted round message / z_partial / comb-affecting declared count).

use std::collections::BTreeSet;

use aggregate::{Accumulator, prove_aggregate, verify_aggregate};
use bincode::serialize;
use flock_core::{
    aggregate::{self, AggregateError},
    challenger::{Challenger, FsChallenger},
    field::F128,
    lincheck::{
        self, LincheckCircuit, QuirkyPoint, SkipPoint, SparseMatrixCircuit, UnionLincheckSlot,
        build_eq_table, build_quirky_eq_table, pack_z_lincheck,
    },
    matrix_fold::{self, FoldError, MatrixClaim, Weight},
    r1cs::SparseBinaryMatrix,
    schedule::{Registry, TableClass, TableType},
    test_rng::Rng,
    union::UnionInstance,
    zerocheck::{self, K_SKIP, multilinear::lagrange_weights_naive, univariate_skip::pack_bits},
};
use lincheck::{
    LincheckClaim, LincheckError, LincheckGrinding, LincheckProof, MatrixAssertion, eq_prefix_sum,
    eq_prefix_weight, prove_padded_capture_z_vec, prove_union_capture_z_vec,
    prove_union_capture_z_vec_with_grinding, union_comb_partial, verify as verify_lincheck,
    verify_union, verify_union_deferred, verify_union_with_grinding,
};
use matrix_fold::{col_marginal, prove_fold, verify_fold};
use zerocheck::{prove_packed_padded, verify as verify_zerocheck};
const DOMAIN: &[u8] = b"flock-union-lincheck-test-v0";

fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

/// Random sparse `k × k` matrix supported on the useful square: rows
/// `[0, useful)` get ~`per_row` entries in columns `[0, useful)`; rows
/// `[useful, k)` are empty. Keeps `M·z` honestly zero on the padding
/// columns of every block (the same shape as the hash encoders' matrices).
fn random_useful_matrix(
    k: usize,
    useful: usize,
    per_row: usize,
    rng: &mut Rng,
) -> SparseBinaryMatrix {
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
    for row in rows.iter_mut().take(useful) {
        let mut cols = BTreeSet::new();
        for _ in 0..per_row {
            cols.insert((rng.next_u64() as usize) % useful);
        }
        *row = cols.into_iter().collect();
    }
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows,
    }
}

/// One synthetic slot: matrices, declared count, and the semantic
/// (column-inner, row-outer) witness triple `z, a = A·z, b = B·z` over the
/// full capacity (dummy rows all-zero — pin included).
struct SlotData {
    k_log: usize,
    useful: usize,
    n: usize,
    pin: Option<usize>,
    a0: SparseBinaryMatrix,
    b0: SparseBinaryMatrix,
    /// `z[v + row·k]` — trace position `v`, invocation `row`.
    z_sem: Vec<bool>,
    a_sem: Vec<bool>,
    b_sem: Vec<bool>,
}

fn build_slot(
    k_log: usize,
    useful: usize,
    nu: usize,
    n: usize,
    pin: Option<usize>,
    seed: u64,
) -> SlotData {
    let mut rng = Rng::new(seed);
    let k = 1usize << k_log;
    let rows = 1usize << nu;
    assert!(n <= rows);
    let a0 = random_useful_matrix(k, useful, 4, &mut rng);
    let b0 = random_useful_matrix(k, useful, 4, &mut rng);

    let mut z_sem = vec![false; k * rows];
    for row in 0..n {
        for v in 0..useful {
            z_sem[v + row * k] = rng.bit();
        }
        if let Some(p) = pin {
            z_sem[p + row * k] = true; // declared rows carry the pin at 1
        }
    }
    // a = A·z, b = B·z per invocation (dummy rows: zero in, zero out).
    let apply = |m: &SparseBinaryMatrix| -> Vec<bool> {
        let mut out = vec![false; k * rows];
        for row in 0..rows {
            for (i, cols) in m.rows.iter().enumerate() {
                let mut acc = false;
                for &c in cols {
                    acc ^= z_sem[c + row * k];
                }
                out[i + row * k] = acc;
            }
        }
        out
    };
    let a_sem = apply(&a0);
    let b_sem = apply(&b0);
    SlotData {
        k_log,
        useful,
        n,
        pin,
        a0,
        b0,
        z_sem,
        a_sem,
        b_sem,
    }
}

fn table_type(slot: &SlotData) -> TableType {
    TableType {
        k_log: slot.k_log,
        useful_bits: slot.useful,
        a_0: slot.a0.clone(),
        b_0: slot.b0.clone(),
        c_0: identity(1 << slot.k_log),
        const_pin: slot.pin,
        class: TableClass::Boolean,
        io_schema: Vec::new(),
    }
}

/// Scatter a slot's semantic vector into a union address-space vector at
/// the slot's offset, under the BatchMajor address split
/// `[7 in-word | ν row | κ−7 chunk]`.
fn place_addr(dst: &mut [bool], sem: &[bool], k_log: usize, nu: usize, offset: usize) {
    let k = 1usize << k_log;
    for row in 0..(1usize << nu) {
        for v in 0..k {
            let addr = offset + (v & 127) + (row << 7) + ((v >> 7) << (7 + nu));
            dst[addr] = sem[v + row * k];
        }
    }
}

/// Brute-force quirky MLE of an address-ordered Boolean vector at
/// `(z_skip, rest)`: φ8 Lagrange on the low `K_SKIP` address bits,
/// multilinear eq on the rest (LSB-first, `rest[j]` binds address bit
/// `K_SKIP + j`).
fn quirky_eval_addr(f: &[bool], z_skip: F128, rest: &[F128]) -> F128 {
    let lambda = lagrange_weights_naive(K_SKIP, z_skip);
    let eq = build_eq_table(rest);
    assert_eq!(f.len(), lambda.len() * eq.len());
    let mask = (1usize << K_SKIP) - 1;
    let mut acc = F128::ZERO;
    for (i, &bit) in f.iter().enumerate() {
        if bit {
            acc += lambda[i & mask] * eq[i >> K_SKIP];
        }
    }
    acc
}

/// Brute-force quirky MLE of a semantic (column-inner, row-outer) Boolean
/// vector at a semantic [`QuirkyPoint`].
fn quirky_eval_sem(f: &[bool], k_log: usize, p: &QuirkyPoint) -> F128 {
    let lambda = lagrange_weights_naive(K_SKIP, p.z_skip.phi8());
    let eq_rest = build_eq_table(&p.x_inner_rest);
    let eq_outer = build_eq_table(&p.x_outer);
    let skip_mask = (1usize << K_SKIP) - 1;
    let rest_mask = (1usize << (k_log - K_SKIP)) - 1;
    let mut acc = F128::ZERO;
    for (i, &bit) in f.iter().enumerate() {
        if bit {
            acc +=
                lambda[i & skip_mask] * eq_rest[(i >> K_SKIP) & rest_mask] * eq_outer[i >> k_log];
        }
    }
    acc
}

/// T = 1 degeneration: on a single-type registry at full utilization the
/// union-column lincheck produces byte-identical proof messages, claims,
/// captured fold, and post-proof transcript state to today's single-table
/// lincheck — and both verifiers accept with the same output claim.
#[test]
fn single_type_union_lincheck_is_byte_identical() {
    let (k_log, useful, nu) = (8usize, 200usize, 4usize);
    let n = 1usize << nu; // full utilization: every row declared, pin = 1
    let slot = build_slot(k_log, useful, nu, n, Some(0), 0x51_46_1E);
    let m = nu + k_log;
    let stripe = pack_z_lincheck(&slot.z_sem, m, k_log);
    let circuit = SparseMatrixCircuit::new(&slot.a0, &slot.b0).with_const_pin(slot.pin);

    let mut rng = Rng::new(0x0DD_0);
    let x_ab = QuirkyPoint {
        z_skip: SkipPoint::Phi8(rng.f128()),
        x_inner_rest: rng.f128_vec(k_log - K_SKIP),
        x_outer: rng.f128_vec(nu),
    };

    // Today's single-table lincheck.
    let mut ch1 = FsChallenger::new(DOMAIN);
    let (proof1, claim1, zvec1) =
        prove_padded_capture_z_vec(&stripe, m, k_log, K_SKIP, useful, &circuit, &x_ab, &mut ch1);

    // The union-column lincheck on the one-slot registry.
    let registry = Registry::new(vec![table_type(&slot)], nu);
    let union = UnionInstance::new(&registry, vec![n]);
    let mut ch2 = FsChallenger::new(DOMAIN);
    let (proof2, claim2, zvec2) = prove_union_capture_z_vec(
        &union,
        &[UnionLincheckSlot {
            z_lincheck: &stripe,
            circuit: &circuit,
        }],
        &x_ab,
        &mut ch2,
    );

    // The SUMCHECK is byte-identical — that is what "T = 1 degenerates to the
    // single-table lincheck" means. The union path additionally reports the
    // per-matrix bilinear values for accumulation; the single-table path does
    // not accumulate and leaves them empty, so they are compared separately
    // (below) rather than papered over.
    assert_eq!(
        serialize(&(&proof1.rounds, &proof1.z_partial)).unwrap(),
        serialize(&(&proof2.rounds, &proof2.z_partial)).unwrap(),
        "union lincheck sumcheck must be byte-identical to the single-table one"
    );
    assert!(
        proof1.matrix_evals.is_empty(),
        "the single-table path does not accumulate"
    );
    assert_eq!(proof2.matrix_evals.len(), 1, "one boolean type, one report");
    assert_eq!(claim1, claim2, "claims diverged");
    assert_eq!(zvec1, zvec2, "captured pre-sumcheck folds diverged");
    assert_eq!(
        ch1.sample_f128(),
        ch2.sample_f128(),
        "post-proof transcript state diverged"
    );

    // Both verifiers accept the honest proof and agree on the claim.
    let v_a = quirky_eval_sem(&slot.a_sem, k_log, &x_ab);
    let v_b = quirky_eval_sem(&slot.b_sem, k_log, &x_ab);
    let mut chv1 = FsChallenger::new(DOMAIN);
    let vclaim1 = verify_lincheck(
        m, k_log, K_SKIP, &circuit, &x_ab, v_a, v_b, &proof1, &mut chv1,
    )
    .expect("single-table verifier must accept");
    let mut chv2 = FsChallenger::new(DOMAIN);
    let vclaim2 = verify_union(&union, &[&circuit], &x_ab, v_a, v_b, &proof2, &mut chv2)
        .expect("union verifier must accept");
    assert_eq!(vclaim1, claim1);
    assert_eq!(vclaim2, claim2);
    assert_eq!(
        chv1.sample_f128(),
        chv2.sample_f128(),
        "post-verify transcript state diverged"
    );
}

/// T = 2 vs brute force: union zerocheck → union-column lincheck on a
/// synthetic two-type registry with partial counts and one const-pinned
/// type, cross-checked against dense recomputations, plus tamper rejection.
#[test]
fn two_type_union_lincheck_matches_brute_force() {
    let nu = 4usize;
    // Type A: κ = 9, 300 useful bits (3 of 4 chunk-columns), pinned at
    // column 0, 11 of 16 rows declared. Type B: κ = 8, 120 useful bits
    // (1 of 2 chunk-columns), unpinned, 13 of 16 rows declared.
    let slot_a = build_slot(9, 300, nu, 11, Some(0), 0x2A_2A_01);
    let slot_b = build_slot(8, 120, nu, 13, None, 0x2B_2B_02);
    let registry = Registry::new(vec![table_type(&slot_a), table_type(&slot_b)], nu);
    assert_eq!(registry.types()[0].k_log, 9, "slot order: A (wider) first");
    let m = registry.m_total();
    assert_eq!(m, 14); // 2^13 + 2^12 rounded up to 2^14 (with a gap)
    let union = UnionInstance::new(&registry, vec![slot_a.n, slot_b.n]);

    // Dense union address-space buffers; c = a ∘ b pointwise.
    let mut z_addr = vec![false; 1 << m];
    let mut a_addr = vec![false; 1 << m];
    let mut b_addr = vec![false; 1 << m];
    for (slot, layout) in [&slot_a, &slot_b].into_iter().zip(registry.slots()) {
        place_addr(&mut z_addr, &slot.z_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut a_addr, &slot.a_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut b_addr, &slot.b_sem, slot.k_log, nu, layout.offset);
    }
    let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
    let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));
    let padding = union.padding_spec();
    assert!(padding.as_single_run().is_none(), "must exercise multi-run");

    // ---- Prove: union zerocheck, then the union-column lincheck.
    let stripe_a = pack_z_lincheck(&slot_a.z_sem, nu + slot_a.k_log, slot_a.k_log);
    let stripe_b = pack_z_lincheck(&slot_b.z_sem, nu + slot_b.k_log, slot_b.k_log);
    let circ_a = SparseMatrixCircuit::new(&slot_a.a0, &slot_a.b0).with_const_pin(slot_a.pin);
    let circ_b = SparseMatrixCircuit::new(&slot_b.a0, &slot_b.b0);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (zc_proof, zc_claim) = prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch_p);
    let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);
    let lc_slots = [
        UnionLincheckSlot {
            z_lincheck: &stripe_a,
            circuit: &circ_a,
        },
        UnionLincheckSlot {
            z_lincheck: &stripe_b,
            circuit: &circ_b,
        },
    ];
    let (lc_proof, lc_claim, _g_vec) =
        prove_union_capture_z_vec(&union, &lc_slots, &x_ab, &mut ch_p);

    // ---- Verify (and probe the transcript for α, β_A, and the round
    // challenges — cloned BEFORE verify_union consumes them).
    let mut ch_v = FsChallenger::new(DOMAIN);
    let zc_claim_v = verify_zerocheck(m, &zc_proof, &mut ch_v).expect("zerocheck must accept");
    assert_eq!(zc_claim_v, zc_claim);
    let x_ab_v = union.x_ab_from_mlv(SkipPoint::Phi8(zc_claim_v.z), &zc_claim_v.mlv_challenges);
    assert_eq!(x_ab_v, x_ab);

    let mut probe = ch_v.clone();
    probe.observe_label(b"flock-lincheck-v0");
    let alpha = probe.sample_f128();
    let beta_a = probe.sample_f128(); // slot A's pin; B has none
    let r_rounds: Vec<F128> = lc_proof
        .rounds
        .iter()
        .map(|&(e1, einf)| {
            probe.observe_f128(e1);
            probe.observe_f128(einf);
            probe.sample_f128()
        })
        .collect();
    let mut rr = r_rounds.clone();
    rr.reverse();

    let circuits: [&dyn LincheckCircuit; 2] = [&circ_a, &circ_b];
    let lc_claim_v = verify_union(
        &union,
        &circuits,
        &x_ab_v,
        zc_claim.a_eval,
        zc_claim.b_eval,
        &lc_proof,
        &mut ch_v,
    )
    .expect("union lincheck verifier must accept the honest proof");
    assert_eq!(lc_claim_v, lc_claim);

    // ---- (a) The lincheck's initial claim α·â(r) + b̂(r) equals the
    // brute-force quirky MLEs of the dense union buffers at the zerocheck
    // point.
    let v_a_bf = quirky_eval_addr(&a_addr, zc_claim.z, &zc_claim.mlv_challenges);
    let v_b_bf = quirky_eval_addr(&b_addr, zc_claim.z, &zc_claim.mlv_challenges);
    assert_eq!(zc_claim.a_eval, v_a_bf, "â(r) != brute force");
    assert_eq!(zc_claim.b_eval, v_b_bf, "b̂(r) != brute force");
    assert_eq!(
        alpha * zc_claim.a_eval + zc_claim.b_eval,
        alpha * v_a_bf + v_b_bf,
        "initial lincheck claim != brute force"
    );

    // ---- (b) The final witness claim equals the brute-force MLE of the
    // union witness at the (address-ordered) claim point.
    let point = union.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer);
    let rest: Vec<F128> = point
        .x_inner_rest
        .iter()
        .chain(&point.x_outer)
        .copied()
        .collect();
    assert_eq!(
        lc_claim.w,
        quirky_eval_addr(&z_addr, point.z_skip.phi8(), &rest),
        "final witness claim != brute-force union witness MLE"
    );

    // ---- (c) The verifier's closed-form Comb-hat collapse equals a dense
    // union Comb MLE at the bound point.
    // Verifier-identical per-type combs: circuit fold + w_t scale + β pin.
    let mut combs_v: Vec<Vec<F128>> = registry
        .types()
        .iter()
        .zip(registry.slots())
        .zip(circuits)
        .map(|((ty, layout), circuit)| {
            let inner = ty.k_log - K_SKIP;
            let eq_inner =
                build_quirky_eq_table(x_ab.z_skip.phi8(), &x_ab.x_inner_rest[..inner], K_SKIP);
            let mut comb = circuit.fold_alpha_batched(alpha, &eq_inner);
            let w_t = eq_prefix_weight(&x_ab.x_inner_rest[inner..], layout.prefix);
            for v in &mut comb {
                *v *= w_t;
            }
            comb
        })
        .collect();
    combs_v[0][slot_a.pin.unwrap()] += beta_a;
    let closed = union_comb_partial(&registry, &combs_v, &rr, K_SKIP);

    // Dense: per-type ξ recomputed from the raw matrix entries with explicit
    // Lagrange × eq products, w_t-scaled, placed at the aligned column
    // offsets (zero on gaps), pin added, then folded with the full eq tensor
    // over the bound challenges.
    let col_vars = m - nu;
    let lambda = lagrange_weights_naive(K_SKIP, x_ab.z_skip.phi8());
    let mut dense = vec![F128::ZERO; 1 << col_vars];
    for (slot, layout) in [&slot_a, &slot_b].into_iter().zip(registry.slots()) {
        let inner = slot.k_log - K_SKIP;
        let eq_rest_t = build_eq_table(&x_ab.x_inner_rest[..inner]);
        let mut w_t = F128::ONE;
        for (j, &x) in x_ab.x_inner_rest[inner..].iter().enumerate() {
            w_t *= if (layout.prefix >> j) & 1 == 1 {
                x
            } else {
                F128::ONE + x
            };
        }
        let off = layout.prefix << slot.k_log;
        for (i, cols) in slot.a0.rows.iter().enumerate() {
            let wq = alpha * w_t * lambda[i & 63] * eq_rest_t[i >> K_SKIP];
            for &c in cols {
                dense[off + c] += wq;
            }
        }
        for (i, cols) in slot.b0.rows.iter().enumerate() {
            let wq = w_t * lambda[i & 63] * eq_rest_t[i >> K_SKIP];
            for &c in cols {
                dense[off + c] += wq;
            }
        }
        if let Some(p) = slot.pin {
            dense[off + p] += beta_a; // only slot A is pinned
        }
    }
    let eq_rr = build_eq_table(&rr);
    let n_skip = 1usize << K_SKIP;
    let mut dense_partial = vec![F128::ZERO; n_skip];
    for (j, &e) in eq_rr.iter().enumerate() {
        for (i, x) in dense_partial.iter_mut().enumerate() {
            *x += e * dense[(j << K_SKIP) + i];
        }
    }
    assert_eq!(
        closed, dense_partial,
        "closed-form Comb-hat collapse != brute-force dense Comb MLE"
    );

    // Final-check ledger: replaying the rounds from the brute-force target
    // (with the count-derived pin term) must land exactly on
    // Σ comb_partial · z_partial — the verifier's accepted equation.
    let mut running = alpha * v_a_bf + v_b_bf + beta_a * eq_prefix_sum(&x_ab.x_outer, slot_a.n);
    for (&(e1, einf), &r) in lc_proof.rounds.iter().zip(&r_rounds) {
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
    }
    let final_sum = closed
        .iter()
        .zip(&lc_proof.z_partial)
        .fold(F128::ZERO, |acc, (c, z)| acc + *c * *z);
    assert_eq!(running, final_sum, "brute-force sumcheck ledger broke");

    // ---- Tampers. Each replays the verifier from a fresh transcript.
    let verify_with = |union: &UnionInstance<'_>,
                       proof: &LincheckProof|
     -> Result<LincheckClaim, LincheckError> {
        let mut ch = FsChallenger::new(DOMAIN);
        let zc = verify_zerocheck(m, &zc_proof, &mut ch).expect("zerocheck side is untampered");
        let x = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        verify_union(union, &circuits, &x, zc.a_eval, zc.b_eval, proof, &mut ch)
    };

    // Corrupted round message.
    let mut bad = lc_proof.clone();
    bad.rounds[1].0.lo ^= 1;
    assert!(
        matches!(
            verify_with(&union, &bad),
            Err(LincheckError::ConsistencyFailed { .. })
        ),
        "corrupted round message must be rejected"
    );

    // Corrupted z_partial entry.
    let mut bad = lc_proof.clone();
    bad.z_partial[3].hi ^= 1;
    assert!(
        matches!(
            verify_with(&union, &bad),
            Err(LincheckError::ConsistencyFailed { .. })
        ),
        "corrupted z_partial must be rejected"
    );

    // Comb-affecting count: the pinned slot's declared count enters the
    // verifier's target through the const-pin term β·Σ_{row<n}eq(x_outer,row),
    // so a corrupted count must fail the final consistency check.
    let union_bad = UnionInstance::new(&registry, vec![slot_a.n - 1, slot_b.n]);
    assert!(
        matches!(
            verify_with(&union_bad, &lc_proof),
            Err(LincheckError::ConsistencyFailed { .. })
        ),
        "corrupted pinned-slot count must be rejected"
    );

    // The grinded multi-table path has the same relation but a different
    // transcript: α, the one pinned table's β, every sumcheck round, then
    // the φ8 skip point are all preceded by a PoW witness.
    let grinding = LincheckGrinding::per_challenge_128();
    let mut ch_p_secure = FsChallenger::new(b"flock-union-lincheck-secure-v0");
    let (zc_proof_secure, zc_claim_secure) =
        prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch_p_secure);
    let x_ab_secure = union.x_ab_from_mlv(
        SkipPoint::Phi8(zc_claim_secure.z),
        &zc_claim_secure.mlv_challenges,
    );
    let (lc_proof_secure, lc_claim_secure, _) = prove_union_capture_z_vec_with_grinding(
        &union,
        &lc_slots,
        &x_ab_secure,
        grinding,
        &mut ch_p_secure,
    );
    assert_eq!(
        lc_proof_secure.grinding_nonces.len(),
        grinding.nonce_count(registry.m_bool() - nu - K_SKIP, 1, K_SKIP),
    );

    let mut ch_v_secure = FsChallenger::new(b"flock-union-lincheck-secure-v0");
    let zc_claim_secure_v = verify_zerocheck(m, &zc_proof_secure, &mut ch_v_secure)
        .expect("ungrinded zerocheck side must verify");
    let x_ab_secure_v = union.x_ab_from_mlv(
        SkipPoint::Phi8(zc_claim_secure_v.z),
        &zc_claim_secure_v.mlv_challenges,
    );
    let lc_claim_secure_v = verify_union_with_grinding(
        &union,
        &circuits,
        &x_ab_secure_v,
        zc_claim_secure_v.a_eval,
        zc_claim_secure_v.b_eval,
        &lc_proof_secure,
        grinding,
        &mut ch_v_secure,
    )
    .expect("grinded union lincheck must verify");
    assert_eq!(lc_claim_secure_v, lc_claim_secure);

    let mut missing_nonce = lc_proof_secure.clone();
    missing_nonce.grinding_nonces.pop();
    let mut ch_missing = FsChallenger::new(b"flock-union-lincheck-secure-v0");
    let zc_missing = verify_zerocheck(m, &zc_proof_secure, &mut ch_missing)
        .expect("ungrinded zerocheck side must verify");
    let x_missing = union.x_ab_from_mlv(SkipPoint::Phi8(zc_missing.z), &zc_missing.mlv_challenges);
    assert!(matches!(
        verify_union_with_grinding(
            &union,
            &circuits,
            &x_missing,
            zc_missing.a_eval,
            zc_missing.b_eval,
            &missing_nonce,
            grinding,
            &mut ch_missing,
        ),
        Err(LincheckError::BadGrindingNonceCount { .. })
    ));
}

/// The deferred split is behaviour-preserving, and the deferred half really
/// does leave the matrix work undone.
///
/// `verify_union` is `verify_union_deferred` + `MatrixAssertion::check`, so the
/// two must agree on every input — including tampered ones, where the
/// interesting property is *which* half rejects. A corrupted round message
/// perturbs `running`, which the assertion carries as `target`, so the deferred
/// pass still returns `Ok` and only `check` catches it. That is the point of
/// the split: nothing before `check` reads a base matrix, which is why a
/// circuit can replay the deferred half and carry the assertion out as a claim
/// instead of paying `O(nnz)`.
#[test]
fn deferred_lincheck_matches_and_defers_the_matrix_work() {
    let nu = 4usize;
    let slot_a = build_slot(9, 300, nu, 11, Some(0), 0x2A_2A_01);
    let slot_b = build_slot(8, 120, nu, 13, None, 0x2B_2B_02);
    let registry = Registry::new(vec![table_type(&slot_a), table_type(&slot_b)], nu);
    let m = registry.m_total();
    let union = UnionInstance::new(&registry, vec![slot_a.n, slot_b.n]);

    let mut z_addr = vec![false; 1 << m];
    let mut a_addr = vec![false; 1 << m];
    let mut b_addr = vec![false; 1 << m];
    for (slot, layout) in [&slot_a, &slot_b].into_iter().zip(registry.slots()) {
        place_addr(&mut z_addr, &slot.z_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut a_addr, &slot.a_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut b_addr, &slot.b_sem, slot.k_log, nu, layout.offset);
    }
    let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
    let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));

    let stripe_a = pack_z_lincheck(&slot_a.z_sem, nu + slot_a.k_log, slot_a.k_log);
    let stripe_b = pack_z_lincheck(&slot_b.z_sem, nu + slot_b.k_log, slot_b.k_log);
    let circ_a = SparseMatrixCircuit::new(&slot_a.a0, &slot_a.b0).with_const_pin(slot_a.pin);
    let circ_b = SparseMatrixCircuit::new(&slot_b.a0, &slot_b.b0);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (zc_proof, zc_claim) =
        prove_packed_padded(&a_p, &b_p, &c_p, m, &union.padding_spec(), &mut ch_p);
    let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);
    let lc_slots = [
        UnionLincheckSlot {
            z_lincheck: &stripe_a,
            circuit: &circ_a,
        },
        UnionLincheckSlot {
            z_lincheck: &stripe_b,
            circuit: &circ_b,
        },
    ];
    let (lc_proof, _lc_claim, _g) = prove_union_capture_z_vec(&union, &lc_slots, &x_ab, &mut ch_p);

    let circuits: Vec<&dyn LincheckCircuit> = vec![&circ_a, &circ_b];
    // The transcript up to the lincheck does not depend on the lincheck proof.
    let replay = || {
        let mut ch = FsChallenger::new(DOMAIN);
        let zc = verify_zerocheck(m, &zc_proof, &mut ch).expect("zerocheck untampered");
        let x = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        (x, zc.a_eval, zc.b_eval, ch)
    };

    // Honest: both entries agree, and the assertion discharges.
    let (x, va, vb, mut ch) = replay();
    let direct = verify_union(&union, &circuits, &x, va, vb, &lc_proof, &mut ch)
        .expect("direct verify accepts");
    let (x, va, vb, mut ch) = replay();
    let (deferred, assertion) =
        verify_union_deferred(&union, &circuits, &x, va, vb, &lc_proof, &mut ch)
            .expect("deferred verify accepts");
    assert_eq!(
        direct, deferred,
        "the split must not change the witness claim"
    );
    assert!(
        assertion.check(&union, &circuits).is_ok(),
        "an honest assertion must discharge"
    );

    // Tampered: the composition still rejects, but the deferred half does not —
    // it never reads a matrix, so `check` is what catches it.
    let mut bad = lc_proof.clone();
    bad.rounds[1].0.lo ^= 1;
    let (x, va, vb, mut ch) = replay();
    assert!(
        matches!(
            verify_union(&union, &circuits, &x, va, vb, &bad, &mut ch),
            Err(LincheckError::ConsistencyFailed { .. })
        ),
        "composed verifier must still reject a corrupted round message"
    );
    let (x, va, vb, mut ch) = replay();
    let (_, tampered) = verify_union_deferred(&union, &circuits, &x, va, vb, &bad, &mut ch)
        .expect("the deferred half accepts — it defers");
    assert!(
        matches!(
            tampered.check(&union, &circuits),
            Err(LincheckError::ConsistencyFailed { .. })
        ),
        "the assertion is what must catch it"
    );
}

/// The bridge between the two halves of the accumulation route: a real
/// `MatrixAssertion` decomposes into per-matrix claims that `matrix_fold`
/// can fold.
///
/// The decomposition is the load-bearing algebra. For a single boolean type
/// filling the region, `fold_alpha_batched`'s contract
/// `comb[c] = α·Σ_{r∈colA(c)} eq_inner[r] + Σ_{r∈colB(c)} eq_inner[r]`
/// and the closed-form collapse give
///
/// ```text
///   target = α·⟨W_row⊗W_col, A₀⟩ + ⟨W_row⊗W_col, B₀⟩ + β·W_col[pin]
/// ```
///
/// with `W_row = λ(z_skip) ⊗ eq(x_inner_rest)` and `W_col = z_partial ⊗ eq(rr)`
/// — exactly `matrix_fold::Weight`'s shape. Pinning this is what says the
/// verifier's matrix work really is two evaluations of two *static* matrices,
/// which is what makes accumulating them (rather than delegating, or paying
/// them in-circuit) sound. It also shows why A and B stay separate claims:
/// only their α-combination appears in the target, and α is per-proof.
#[test]
fn matrix_assertion_decomposes_into_foldable_claims() {
    let nu = 4usize;
    let slot = build_slot(9, 300, nu, 11, Some(0), 0xF0_0D_01);
    let registry = Registry::new(vec![table_type(&slot)], nu);
    let m = registry.m_total();
    let union = UnionInstance::new(&registry, vec![slot.n]);

    let mut z_addr = vec![false; 1 << m];
    let mut a_addr = vec![false; 1 << m];
    let mut b_addr = vec![false; 1 << m];
    place_addr(
        &mut z_addr,
        &slot.z_sem,
        slot.k_log,
        nu,
        registry.slots()[0].offset,
    );
    place_addr(
        &mut a_addr,
        &slot.a_sem,
        slot.k_log,
        nu,
        registry.slots()[0].offset,
    );
    place_addr(
        &mut b_addr,
        &slot.b_sem,
        slot.k_log,
        nu,
        registry.slots()[0].offset,
    );
    let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
    let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));

    let stripe = pack_z_lincheck(&slot.z_sem, nu + slot.k_log, slot.k_log);
    let circ = SparseMatrixCircuit::new(&slot.a0, &slot.b0).with_const_pin(slot.pin);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (zc_proof, zc_claim) =
        prove_packed_padded(&a_p, &b_p, &c_p, m, &union.padding_spec(), &mut ch_p);
    let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);
    let lc_slots = [UnionLincheckSlot {
        z_lincheck: &stripe,
        circuit: &circ,
    }];
    let (lc_proof, _claim, _g) = prove_union_capture_z_vec(&union, &lc_slots, &x_ab, &mut ch_p);

    let circuits: Vec<&dyn LincheckCircuit> = vec![&circ];
    let mut ch_v = FsChallenger::new(DOMAIN);
    let zc = verify_zerocheck(m, &zc_proof, &mut ch_v).expect("zerocheck accepts");
    let x = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
    let (_, assertion) = verify_union_deferred(
        &union, &circuits, &x, zc.a_eval, zc.b_eval, &lc_proof, &mut ch_v,
    )
    .expect("deferred verify accepts");
    assert!(
        assertion.check(&union, &circuits).is_ok(),
        "assertion is honest"
    );

    // ---- Decompose into the two per-matrix claims.
    let inner = slot.k_log - K_SKIP;
    let row = Weight::low_eq(
        assertion.z_skip.weights(K_SKIP),
        assertion.x_inner_rest[..inner].to_vec(),
    );
    let col = Weight::low_eq(assertion.z_partial.clone(), assertion.rr.clone());
    assert_eq!(row.n_vars(), slot.k_log, "row weight spans the block");
    assert_eq!(col.n_vars(), slot.k_log, "column weight spans the block");

    let claim_a = MatrixClaim::honest(row.clone(), col.clone(), &slot.a0);
    let claim_b = MatrixClaim::honest(row, col.clone(), &slot.b0);
    let beta = assertion.betas[0].expect("this slot is pinned");
    let pin_term = beta * col.materialize()[slot.pin.unwrap()];

    assert_eq!(
        assertion.alpha * claim_a.value + claim_b.value + pin_term,
        assertion.target,
        "the assertion must decompose as α·⟨W,A⟩ + ⟨W,B⟩ + β·W_col[pin]"
    );

    // ---- And the claims fold. A second honest claim about the SAME matrix
    // stands in for a second proof's claim; folding is over one matrix, which
    // is why A and B get their own accumulators.
    let other = MatrixClaim::honest(
        Weight::eq(
            (0..slot.k_log)
                .map(|i| F128::new(0x51 + i as u64, 7))
                .collect(),
        ),
        Weight::eq(
            (0..slot.k_log)
                .map(|i| F128::new(0x77 + i as u64, 3))
                .collect(),
        ),
        &slot.a0,
    );
    let pair = [claim_a, other];
    let combs: Vec<Vec<F128>> = pair
        .iter()
        .map(|c| col_marginal(&slot.a0, &c.row.materialize(), slot.a0.num_cols))
        .collect();
    let mut ch = FsChallenger::new(b"fold");
    let (fold_proof, _) = prove_fold(&slot.a0, &combs, &pair, &mut ch);
    let mut chv = FsChallenger::new(b"fold");
    let acc = verify_fold(&pair, &fold_proof, &mut chv).expect("fold verifies");
    assert!(
        acc.check_direct(&slot.a0),
        "the accumulated claim must discharge against the real base matrix"
    );
}

/// The accumulated path agrees with the direct one, on a TWO-type registry
/// where the slot prefix weights are not 1.
///
/// `check` reads the matrices (`O(Σ_t nnz_t)`); `check_reported` reads only
/// the prover's reported per-matrix values and the verifier's own scalars.
/// They must accept the same proofs — and the reported values must be the
/// honest bilinear forms, which is what `claims()` hands to the accumulator.
/// Corrupting a reported value has to be caught by one side or the other:
/// the equation if it is inconsistent, the claim's own discharge if it is
/// consistent-but-wrong.
#[test]
fn reported_matrix_evals_agree_with_reading_the_matrices() {
    let nu = 4usize;
    let slot_a = build_slot(9, 300, nu, 11, Some(0), 0x2A_2A_01);
    let slot_b = build_slot(8, 120, nu, 13, None, 0x2B_2B_02);
    let registry = Registry::new(vec![table_type(&slot_a), table_type(&slot_b)], nu);
    let m = registry.m_total();
    let union = UnionInstance::new(&registry, vec![slot_a.n, slot_b.n]);

    let mut z_addr = vec![false; 1 << m];
    let mut a_addr = vec![false; 1 << m];
    let mut b_addr = vec![false; 1 << m];
    for (slot, layout) in [&slot_a, &slot_b].into_iter().zip(registry.slots()) {
        place_addr(&mut z_addr, &slot.z_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut a_addr, &slot.a_sem, slot.k_log, nu, layout.offset);
        place_addr(&mut b_addr, &slot.b_sem, slot.k_log, nu, layout.offset);
    }
    let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
    let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));

    let stripe_a = pack_z_lincheck(&slot_a.z_sem, nu + slot_a.k_log, slot_a.k_log);
    let stripe_b = pack_z_lincheck(&slot_b.z_sem, nu + slot_b.k_log, slot_b.k_log);
    let circ_a = SparseMatrixCircuit::new(&slot_a.a0, &slot_a.b0).with_const_pin(slot_a.pin);
    let circ_b = SparseMatrixCircuit::new(&slot_b.a0, &slot_b.b0);
    let mut ch_p = FsChallenger::new(DOMAIN);
    let (zc_proof, zc_claim) =
        prove_packed_padded(&a_p, &b_p, &c_p, m, &union.padding_spec(), &mut ch_p);
    let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);
    let lc_slots = [
        UnionLincheckSlot {
            z_lincheck: &stripe_a,
            circuit: &circ_a,
        },
        UnionLincheckSlot {
            z_lincheck: &stripe_b,
            circuit: &circ_b,
        },
    ];
    let (lc_proof, _c, _g) = prove_union_capture_z_vec(&union, &lc_slots, &x_ab, &mut ch_p);

    let circuits: Vec<&dyn LincheckCircuit> = vec![&circ_a, &circ_b];
    let assertion_for = |proof: &LincheckProof| {
        let mut ch = FsChallenger::new(DOMAIN);
        let zc = verify_zerocheck(m, &zc_proof, &mut ch).expect("zerocheck accepts");
        let x = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        verify_union_deferred(&union, &circuits, &x, zc.a_eval, zc.b_eval, proof, &mut ch)
            .map(|(_, a)| a)
    };

    // Honest: both routes accept, and every reported value is the honest
    // bilinear form of a real base matrix.
    let assertion = assertion_for(&lc_proof).expect("deferred verify accepts");
    assert!(
        assertion.check(&union, &circuits).is_ok(),
        "matrix route accepts"
    );
    assert!(
        assertion.check_reported(&registry).is_ok(),
        "reported route accepts"
    );
    for ((ca, cb), slot) in assertion
        .claims(&registry)
        .into_iter()
        .zip([&slot_a, &slot_b])
    {
        assert!(ca.check_direct(&slot.a0), "reported A eval must be honest");
        assert!(cb.check_direct(&slot.b0), "reported B eval must be honest");
    }

    // The claims fold, and the accumulator stays true.
    let (ca0, _) = assertion.claims(&registry).swap_remove(0);
    let other = MatrixClaim::honest(
        Weight::eq(
            (0..slot_a.k_log)
                .map(|i| F128::new(9 + i as u64, 4))
                .collect(),
        ),
        Weight::eq(
            (0..slot_a.k_log)
                .map(|i| F128::new(5 + i as u64, 6))
                .collect(),
        ),
        &slot_a.a0,
    );
    let pair = [ca0, other];
    let combs: Vec<Vec<F128>> = pair
        .iter()
        .map(|c| col_marginal(&slot_a.a0, &c.row.materialize(), slot_a.a0.num_cols))
        .collect();
    let mut ch = FsChallenger::new(b"fold");
    let (fp, _) = prove_fold(&slot_a.a0, &combs, &pair, &mut ch);
    let mut chv = FsChallenger::new(b"fold");
    let acc = verify_fold(&pair, &fp, &mut chv).expect("fold verifies");
    assert!(acc.check_direct(&slot_a.a0), "accumulator must stay true");

    // Tampered report: caught by the equation, or by the claim's discharge.
    for idx in 0..2 {
        for which in 0..2 {
            let mut bad = lc_proof.clone();
            if which == 0 {
                bad.matrix_evals[idx].0 += F128::ONE;
            } else {
                bad.matrix_evals[idx].1 += F128::ONE;
            }
            let a = assertion_for(&bad).expect("shape is still fine");
            let equation = a.check_reported(&registry).is_err();
            let claims = a.claims(&registry);
            let slot = if idx == 0 { &slot_a } else { &slot_b };
            let honest =
                claims[idx].0.check_direct(&slot.a0) && claims[idx].1.check_direct(&slot.b0);
            assert!(
                equation || !honest,
                "a tampered report survived both the equation and the claim (slot {idx}, {which})"
            );
        }
    }
}

/// The actual 2→1 step: two INDEPENDENT proofs of the same circuit, their A
/// and B claims folded into one accumulator each, and a third proof folded in
/// on top.
///
/// The other fold tests use synthetic claims; these come from two real
/// lincheck runs at unrelated points, which is what aggregation actually
/// presents. Both accumulators must stay discharge-true, and a corrupted
/// report must not yield a true one — either the fold rejects it outright or
/// the accumulator it produces fails the root discharge. Both outcomes are
/// sound; only a true-looking accumulator would not be.
#[test]
fn two_proofs_fold_two_to_one_per_matrix() {
    const SEED: u64 = 0x5EED_5EED;
    let nu = 4usize;
    // Same seed ⇒ same base matrices (drawn before the witness); different
    // declared counts ⇒ genuinely different proofs of the same circuit.
    let slots = [
        build_slot(9, 300, nu, 11, Some(0), SEED),
        build_slot(9, 300, nu, 7, Some(0), SEED),
        build_slot(9, 300, nu, 14, Some(0), SEED),
    ];
    for s in &slots[1..] {
        assert_eq!(slots[0].a0.rows, s.a0.rows, "same circuit, by seed");
        assert_eq!(slots[0].b0.rows, s.b0.rows, "same circuit, by seed");
    }
    let (a0, b0) = (&slots[0].a0, &slots[0].b0);

    let registry = Registry::new(vec![table_type(&slots[0])], nu);
    let m = registry.m_total();

    // Prove one instance and return its (A, B) claim pair.
    let claims_of = |slot: &SlotData, tamper: bool| -> (MatrixClaim, MatrixClaim) {
        let union = UnionInstance::new(&registry, vec![slot.n]);
        let off = registry.slots()[0].offset;
        let mut z_addr = vec![false; 1 << m];
        let mut a_addr = vec![false; 1 << m];
        let mut b_addr = vec![false; 1 << m];
        place_addr(&mut z_addr, &slot.z_sem, slot.k_log, nu, off);
        place_addr(&mut a_addr, &slot.a_sem, slot.k_log, nu, off);
        place_addr(&mut b_addr, &slot.b_sem, slot.k_log, nu, off);
        let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));
        let stripe = pack_z_lincheck(&slot.z_sem, nu + slot.k_log, slot.k_log);
        let circ = SparseMatrixCircuit::new(&slot.a0, &slot.b0).with_const_pin(slot.pin);

        let mut ch = FsChallenger::new(DOMAIN);
        let (zc_proof, zc) =
            prove_packed_padded(&a_p, &b_p, &c_p, m, &union.padding_spec(), &mut ch);
        let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        let (mut proof, _c, _g) = prove_union_capture_z_vec(
            &union,
            &[UnionLincheckSlot {
                z_lincheck: &stripe,
                circuit: &circ,
            }],
            &x_ab,
            &mut ch,
        );
        if tamper {
            proof.matrix_evals[0].0 += F128::ONE;
        }

        let circuits: Vec<&dyn LincheckCircuit> = vec![&circ];
        let mut chv = FsChallenger::new(DOMAIN);
        let zcv = verify_zerocheck(m, &zc_proof, &mut chv).expect("zerocheck accepts");
        let xv = union.x_ab_from_mlv(SkipPoint::Phi8(zcv.z), &zcv.mlv_challenges);
        let (_, assertion) = verify_union_deferred(
            &union, &circuits, &xv, zcv.a_eval, zcv.b_eval, &proof, &mut chv,
        )
        .expect("deferred verify accepts");
        assertion.claims(&registry).swap_remove(0)
    };

    // One accumulator per matrix — A and B never mix.
    let fold2 =
        |mat: &SparseBinaryMatrix, pair: [MatrixClaim; 2]| -> Result<MatrixClaim, FoldError> {
            let combs: Vec<Vec<F128>> = pair
                .iter()
                .map(|c| col_marginal(mat, &c.row.materialize(), mat.num_cols))
                .collect();
            let mut chp = FsChallenger::new(b"acc");
            let (proof, _) = prove_fold(mat, &combs, &pair, &mut chp);
            let mut chv = FsChallenger::new(b"acc");
            verify_fold(&pair, &proof, &mut chv)
        };

    let (a1, b1) = claims_of(&slots[0], false);
    let (a2, b2) = claims_of(&slots[1], false);
    assert_ne!(a1.row.point, a2.row.point, "two proofs, two points");

    let acc_a = fold2(a0, [a1, a2]).expect("A fold verifies");
    let acc_b = fold2(b0, [b1, b2]).expect("B fold verifies");
    assert!(acc_a.check_direct(a0), "A accumulator must be true");
    assert!(acc_b.check_direct(b0), "B accumulator must be true");

    // A third proof folds in on top — the tree continues, and an accumulated
    // claim (plain eq ⊗ eq) folds against a fresh one (λ ⊗ eq shape).
    let (a3, b3) = claims_of(&slots[2], false);
    let acc_a = fold2(a0, [acc_a, a3]).expect("A fold verifies at depth 2");
    let acc_b = fold2(b0, [acc_b, b3]).expect("B fold verifies at depth 2");
    assert!(acc_a.check_direct(a0), "A accumulator survives depth 2");
    assert!(acc_b.check_direct(b0), "B accumulator survives depth 2");

    // A corrupted report must not yield a true accumulator — and must not
    // disturb the other matrix, since A and B accumulate separately.
    let (bad_a, good_b) = claims_of(&slots[1], true);
    let (a1, b1) = claims_of(&slots[0], false);
    match fold2(a0, [a1, bad_a]) {
        Err(_) => {}
        Ok(acc) => assert!(!acc.check_direct(a0), "a corrupted A report survived"),
    }
    assert!(
        fold2(b0, [b1, good_b])
            .expect("B fold verifies")
            .check_direct(b0),
        "the B accumulator is unaffected by the A-side tamper"
    );
}

/// The aggregation driver end to end: succinctly verify real proofs, fold
/// their matrix work into one accumulator, discharge once.
///
/// This is the composition the recursion circuit arithmetises — and the
/// native payoff on its own, since N proofs now cost N succinct replays plus
/// ONE `O(Σ_t nnz_t)` discharge instead of N of them. `verify_aggregate`
/// reads no matrix, which is what makes it circuit-shaped; the only place
/// anything looks at `A₀`/`B₀` is the prover and the final discharge.
#[test]
fn aggregating_real_proofs_defers_all_matrix_work_to_one_discharge() {
    const SEED: u64 = 0xA66_5EED;
    let nu = 4usize;
    // Same seed ⇒ same base matrices; different counts ⇒ different proofs.
    let slots: Vec<SlotData> = [11usize, 7, 14, 5]
        .into_iter()
        .map(|n| build_slot(9, 300, nu, n, Some(0), SEED))
        .collect();
    let registry = Registry::new(vec![table_type(&slots[0])], nu);
    let m = registry.m_total();
    let mats: Vec<(&SparseBinaryMatrix, &SparseBinaryMatrix)> = vec![(&slots[0].a0, &slots[0].b0)];
    // The tuned column-marginal path the fold uses for its k·nnz work.
    let agg_circ =
        SparseMatrixCircuit::new(&slots[0].a0, &slots[0].b0).with_const_pin(slots[0].pin);

    // Prove one instance and return the assertion its succinct verify emits.
    let assert_of = |slot: &SlotData, tamper: bool| -> MatrixAssertion {
        let union = UnionInstance::new(&registry, vec![slot.n]);
        let off = registry.slots()[0].offset;
        let mut z_addr = vec![false; 1 << m];
        let mut a_addr = vec![false; 1 << m];
        let mut b_addr = vec![false; 1 << m];
        place_addr(&mut z_addr, &slot.z_sem, slot.k_log, nu, off);
        place_addr(&mut a_addr, &slot.a_sem, slot.k_log, nu, off);
        place_addr(&mut b_addr, &slot.b_sem, slot.k_log, nu, off);
        let c_addr: Vec<bool> = a_addr.iter().zip(&b_addr).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = (pack_bits(&a_addr), pack_bits(&b_addr), pack_bits(&c_addr));
        let stripe = pack_z_lincheck(&slot.z_sem, nu + slot.k_log, slot.k_log);
        let circ = SparseMatrixCircuit::new(&slot.a0, &slot.b0).with_const_pin(slot.pin);

        let mut ch = FsChallenger::new(DOMAIN);
        let (zc_proof, zc) =
            prove_packed_padded(&a_p, &b_p, &c_p, m, &union.padding_spec(), &mut ch);
        let x_ab = union.x_ab_from_mlv(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        let (mut proof, _c, _g) = prove_union_capture_z_vec(
            &union,
            &[UnionLincheckSlot {
                z_lincheck: &stripe,
                circuit: &circ,
            }],
            &x_ab,
            &mut ch,
        );
        if tamper {
            proof.matrix_evals[0].1 += F128::ONE;
        }
        let circuits: Vec<&dyn LincheckCircuit> = vec![&circ];
        let mut chv = FsChallenger::new(DOMAIN);
        let zcv = verify_zerocheck(m, &zc_proof, &mut chv).expect("zerocheck accepts");
        let xv = union.x_ab_from_mlv(SkipPoint::Phi8(zcv.z), &zcv.mlv_challenges);
        verify_union_deferred(
            &union, &circuits, &xv, zcv.a_eval, zcv.b_eval, &proof, &mut chv,
        )
        .expect("deferred verify accepts")
        .1
    };

    let run = |asserts: &[MatrixAssertion],
               priors: &[&Accumulator]|
     -> Result<Accumulator, AggregateError> {
        let mut chp = FsChallenger::new(b"agg");
        let circs: Vec<&dyn LincheckCircuit> = vec![&agg_circ];
        let (proof, acc_p) = prove_aggregate(&registry, &mats, &circs, asserts, priors, &mut chp)?;
        let mut chv = FsChallenger::new(b"agg");
        let acc_v = verify_aggregate(&registry, asserts, priors, &proof, &mut chv)?;
        assert_eq!(acc_p, acc_v, "prover and verifier accumulators must agree");
        Ok(acc_v)
    };

    // Leaf: two proofs, 2 → 1 per accumulator.
    let a0 = assert_of(&slots[0], false);
    let a1 = assert_of(&slots[1], false);
    let acc = run(&[a0, a1], &[]).expect("leaf aggregation verifies");
    assert!(acc.discharge(&mats), "leaf accumulator must discharge");
    assert_eq!(acc.registry_digest, registry.digest());

    // Merge: two more proofs folded in on top — 3 → 1 per accumulator (one
    // inherited, two fresh). Still one discharge at the end.
    let a2 = assert_of(&slots[2], false);
    let a3 = assert_of(&slots[3], false);
    let acc2 = run(&[a2, a3], &[&acc]).expect("merge aggregation verifies");
    assert!(acc2.discharge(&mats), "merged accumulator must discharge");

    // The true merge-node shape: TWO priors — each child of a merge brings
    // its own accumulator — plus fresh assertions, folding 4 → 1 per
    // matrix (two inherited, two fresh).
    let leaf = |i: usize, j: usize| {
        run(
            &[assert_of(&slots[i], false), assert_of(&slots[j], false)],
            &[],
        )
        .expect("leaf aggregation verifies")
    };
    let (acc_l, acc_r) = (leaf(0, 1), leaf(2, 3));
    let fresh = [assert_of(&slots[1], false), assert_of(&slots[2], false)];
    let acc_m = run(&fresh, &[&acc_l, &acc_r]).expect("two-prior merge verifies");
    assert!(
        acc_m.discharge(&mats),
        "the 4->1 accumulator must discharge"
    );

    // A tampered INHERITED claim must poison the merge: the fold verifier
    // targets the claimed values, so the replay diverges from the honest
    // prover's rounds — rejection or a false accumulator, both sound.
    let mut bad_prior = acc_l.clone();
    bad_prior.per_type[0].0.value += F128::ONE;
    let fresh2 = [assert_of(&slots[1], false), assert_of(&slots[2], false)];
    match run(&fresh2, &[&bad_prior, &acc_r]) {
        Err(_) => {}
        Ok(a) => assert!(
            !a.discharge(&mats),
            "a tampered inherited claim produced a true accumulator"
        ),
    }

    // A tampered report is caught by the equation check inside
    // `verify_aggregate` — a caller cannot forget it.
    let bad = assert_of(&slots[1], true);
    assert!(
        matches!(run(&[bad], &[]), Err(AggregateError::Reported(_))),
        "a tampered matrix report must be rejected"
    );

    // An accumulator from another registry must not be folded in.
    let other_slot = build_slot(8, 120, nu, 9, None, 0xBAD_5EED);
    let other = Registry::new(vec![table_type(&other_slot)], nu);
    let alien = Accumulator {
        registry_digest: other.digest(),
        per_type: acc.per_type.clone(),
        per_element: Vec::new(),
        sigma: Vec::new(),
        jagged: Vec::new(),
    };
    let a0b = assert_of(&slots[0], false);
    assert!(
        matches!(
            run(&[a0b], &[&alien]),
            Err(AggregateError::RegistryMismatch)
        ),
        "an accumulator from a different registry must be rejected"
    );
}
