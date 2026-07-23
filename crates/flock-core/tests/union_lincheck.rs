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

use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::lincheck::{
    self, LincheckCircuit, QuirkyPoint, SparseMatrixCircuit, UnionLincheckSlot, build_eq_table,
    build_quirky_eq_table, pack_z_lincheck,
};
use flock_core::r1cs::SparseBinaryMatrix;
use flock_core::schedule::{Registry, TableType};
use flock_core::union::UnionInstance;
use flock_core::zerocheck::multilinear::lagrange_weights_naive;
use flock_core::zerocheck::univariate_skip::pack_bits;
use flock_core::zerocheck::{self, K_SKIP};

const DOMAIN: &[u8] = b"flock-union-lincheck-test-v0";

/// SplitMix64 PRNG, deterministic.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
    fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
}

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
        let mut cols = std::collections::BTreeSet::new();
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
    let lambda = lagrange_weights_naive(K_SKIP, p.z_skip);
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
        z_skip: rng.f128(),
        x_inner_rest: rng.f128_vec(k_log - K_SKIP),
        x_outer: rng.f128_vec(nu),
    };

    // Today's single-table lincheck.
    let mut ch1 = FsChallenger::new(DOMAIN);
    let (proof1, claim1, zvec1) = lincheck::prove_padded_capture_z_vec(
        &stripe, m, k_log, K_SKIP, useful, &circuit, &x_ab, &mut ch1,
    );

    // The union-column lincheck on the one-slot registry.
    let registry = Registry::new(vec![table_type(&slot)], nu);
    let union = UnionInstance::new(&registry, vec![n]);
    let mut ch2 = FsChallenger::new(DOMAIN);
    let (proof2, claim2, zvec2) = lincheck::prove_union_capture_z_vec(
        &union,
        &[UnionLincheckSlot {
            z_lincheck: &stripe,
            circuit: &circuit,
        }],
        &x_ab,
        &mut ch2,
    );

    assert_eq!(
        bincode::serialize(&proof1).unwrap(),
        bincode::serialize(&proof2).unwrap(),
        "union lincheck proof must be byte-identical to the single-table one"
    );
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
    let vclaim1 = lincheck::verify(
        m, k_log, K_SKIP, &circuit, &x_ab, v_a, v_b, &proof1, &mut chv1,
    )
    .expect("single-table verifier must accept");
    let mut chv2 = FsChallenger::new(DOMAIN);
    let vclaim2 = lincheck::verify_union(&union, &[&circuit], &x_ab, v_a, v_b, &proof2, &mut chv2)
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
    let (zc_proof, zc_claim) =
        zerocheck::prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch_p);
    let x_ab = union.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
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
        lincheck::prove_union_capture_z_vec(&union, &lc_slots, &x_ab, &mut ch_p);

    // ---- Verify (and probe the transcript for α, β_A, and the round
    // challenges — cloned BEFORE verify_union consumes them).
    let mut ch_v = FsChallenger::new(DOMAIN);
    let zc_claim_v = zerocheck::verify(m, &zc_proof, &mut ch_v).expect("zerocheck must accept");
    assert_eq!(zc_claim_v, zc_claim);
    let x_ab_v = union.x_ab_from_mlv(zc_claim_v.z, &zc_claim_v.mlv_challenges);
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
    let lc_claim_v = lincheck::verify_union(
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
        quirky_eval_addr(&z_addr, point.z_skip, &rest),
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
            let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest[..inner], K_SKIP);
            let mut comb = circuit.fold_alpha_batched(alpha, &eq_inner);
            let w_t = lincheck::eq_prefix_weight(&x_ab.x_inner_rest[inner..], layout.prefix);
            for v in &mut comb {
                *v *= w_t;
            }
            comb
        })
        .collect();
    combs_v[0][slot_a.pin.unwrap()] += beta_a;
    let closed = lincheck::union_comb_partial(&registry, &combs_v, &rr, K_SKIP);

    // Dense: per-type ξ recomputed from the raw matrix entries with explicit
    // Lagrange × eq products, w_t-scaled, placed at the aligned column
    // offsets (zero on gaps), pin added, then folded with the full eq tensor
    // over the bound challenges.
    let col_vars = m - nu;
    let lambda = lagrange_weights_naive(K_SKIP, x_ab.z_skip);
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
    let mut running =
        alpha * v_a_bf + v_b_bf + beta_a * lincheck::eq_prefix_sum(&x_ab.x_outer, slot_a.n);
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
                       proof: &lincheck::LincheckProof|
     -> Result<lincheck::LincheckClaim, lincheck::VerifyError> {
        let mut ch = FsChallenger::new(DOMAIN);
        let zc = zerocheck::verify(m, &zc_proof, &mut ch).expect("zerocheck side is untampered");
        let x = union.x_ab_from_mlv(zc.z, &zc.mlv_challenges);
        lincheck::verify_union(union, &circuits, &x, zc.a_eval, zc.b_eval, proof, &mut ch)
    };

    // Corrupted round message.
    let mut bad = lc_proof.clone();
    bad.rounds[1].0.lo ^= 1;
    assert!(
        matches!(
            verify_with(&union, &bad),
            Err(lincheck::VerifyError::ConsistencyFailed { .. })
        ),
        "corrupted round message must be rejected"
    );

    // Corrupted z_partial entry.
    let mut bad = lc_proof.clone();
    bad.z_partial[3].hi ^= 1;
    assert!(
        matches!(
            verify_with(&union, &bad),
            Err(lincheck::VerifyError::ConsistencyFailed { .. })
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
            Err(lincheck::VerifyError::ConsistencyFailed { .. })
        ),
        "corrupted pinned-slot count must be rejected"
    );
}
