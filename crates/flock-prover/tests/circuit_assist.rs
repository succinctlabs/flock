//! MVP-8 groundwork: the multipoint anchor's boundary DP as a gate.
//!
//! The one genuinely new relation the in-circuit multipoint verifier needs
//! (docs/multipoint-twisted-assist.tex; everything else is PrefixGate /
//! MergedRoundGate / FinalDot reuse) is `verify_frobenius_assist`'s
//! per-statement 4-state boundary DP: `m + 1` layers of the sparse
//! transition table, each layer consuming one `(z_row, ρ″)` coordinate pair
//! and one `(σ_c, σ_d)` challenge pair. [`AssistLayerGate`] is one layer;
//! a statement chains `m + 1` rows from the SUCCESS seed down to the
//! INITIAL read-out, exactly as the native verifier iterates.
//!
//! The transition table and state indices are sourced from
//! `flock_core::pcs::jagged` (`assist_sparse_transitions`, `STATE_*`) — not
//! replicated here — so a protocol change cannot silently drift the gate
//! from the verifier it transcribes.

use flock_core::circuit::builder::{GateType, ShapeBuilder, SlotWitness};
use flock_core::field::F128;
use flock_core::pcs::PcsParams;
use flock_core::pcs::jagged::{STATE_INITIAL, STATE_SUCCESS, assist_sparse_transitions};
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::schedule::IoWord;
use flock_core::verifier;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionElementSlotInput};
use flock_prover::union::UnionInstance;

const DOMAIN: &[u8] = b"flock-circuit-assist-v0";

use flock_core::test_rng::Rng;

/// `point_bit`'s convention: coordinate `layer`, zero past the end.
fn pb(z: &[F128], layer: usize) -> F128 {
    z.get(layer).copied().unwrap_or(F128::ZERO)
}

/// One layer of the anchor verifier's boundary DP.
///
/// Inputs: the 4-state vector `g`, the layer's `(za, rb)` = (row-point
/// coordinate, ρ″ coordinate), the layer's `(rc, rd)` σ pair, and the
/// constant `one`. Outputs: the next state vector. Per layer:
///
/// ```text
///   eq4 = eq-table of [za, rb]         (1 mult + 3 linear, t3 reuses the mult)
///   e   = σ quadrants of (rc, rd)      (same shape)
///   p[i][o] = eq4[i]·g[o]              (16 mults — all combinations)
///   t[cd][s] = e[cd]·(p[i0][o0] + p[i1][o1])   (16 mult_lin, sparse table)
///   g'[s] = Σ_cd t[cd][s]              (4 linear)
/// ```
///
/// 53 columns — kappa 6. The sparse table is baked from
/// [`assist_sparse_transitions`] at construction.
struct AssistLayerGate {
    ty: std::sync::Arc<flock_core::element_r1cs::ElementTableType>,
}

const AL_IN: usize = 9; // g0..g3, za, rb, rc, rd, one
const AL_ONE: usize = 8;
const AL_OUT0: usize = 49;
const AL_K: usize = 53;

impl AssistLayerGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let sparse = assist_sparse_transitions();
        let mut b = ElementTableBuilder::new(6);
        for w in 0..AL_IN {
            b.free_wire(w);
        }
        // eq4 of [za, rb]: m1 = za·rb, then linear forms.
        b.mult(9, 4, 5)
            .linear(10, &[(AL_ONE, one), (4, one), (5, one), (9, one)])
            .linear(11, &[(4, one), (9, one)])
            .linear(12, &[(5, one), (9, one)]);
        let eq4 = [10usize, 11, 12, 9];
        // σ quadrants of (rc, rd).
        b.mult(13, 6, 7)
            .linear(14, &[(AL_ONE, one), (6, one), (7, one), (13, one)])
            .linear(15, &[(6, one), (13, one)])
            .linear(16, &[(7, one), (13, one)]);
        let e = [14usize, 15, 16, 13];
        // All 16 products eq4[i]·g[o].
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                b.mult(p(i, o), eq4[i], o);
            }
        }
        // t[cd][s] = e[cd]·(p(i0,o0) + p(i1,o1)), sparse-table indexed.
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                b.mult_lin(
                    33 + 4 * cd + s,
                    &[(p(i0, o0), one), (p(i1, o1), one)],
                    &[(e[cd], one)],
                );
            }
        }
        // g'[s] = Σ_cd t[cd][s].
        for s in 0..4 {
            b.linear(
                AL_OUT0 + s,
                &[(33 + s, one), (37 + s, one), (41 + s, one), (45 + s, one)],
            );
        }
        Self {
            ty: std::sync::Arc::new(b.build().expect("assist layer gate is valid")),
        }
    }
}

impl GateType for AssistLayerGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> flock_prover::schedule::TableType {
        let mut schema: Vec<IoWord> = (0..AL_IN).map(IoWord::input).collect();
        for s in 0..4 {
            schema.push(IoWord::output(AL_OUT0 + s));
        }
        flock_prover::schedule::TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let (o, row) = {
            let sparse = assist_sparse_transitions();
            let mut z = vec![F128::ZERO; AL_K];
            z[..AL_IN].copy_from_slice(&inputs[..AL_IN]);
            let one = F128::ONE;
            z[9] = z[4] * z[5];
            z[10] = one + z[4] + z[5] + z[9];
            z[11] = z[4] + z[9];
            z[12] = z[5] + z[9];
            let eq4 = [10usize, 11, 12, 9];
            z[13] = z[6] * z[7];
            z[14] = one + z[6] + z[7] + z[13];
            z[15] = z[6] + z[13];
            z[16] = z[7] + z[13];
            let e = [14usize, 15, 16, 13];
            let p = |i: usize, o: usize| 17 + 4 * i + o;
            for i in 0..4 {
                for o in 0..4 {
                    z[p(i, o)] = z[eq4[i]] * z[o];
                }
            }
            for (cd, rows) in sparse.iter().enumerate() {
                for (s, row) in rows.iter().enumerate() {
                    let [(i0, o0), (i1, o1)] = *row;
                    z[33 + 4 * cd + s] = z[e[cd]] * (z[p(i0, o0)] + z[p(i1, o1)]);
                }
            }
            for s in 0..4 {
                z[AL_OUT0 + s] = z[33 + s] + z[37 + s] + z[41 + s] + z[45 + s];
            }
            (z[AL_OUT0..AL_OUT0 + 4].to_vec(), z)
        };
        outputs.extend_from_slice(&o);
        row
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                z[(c << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The native replica of `verify_frobenius_assist`'s per-statement DP —
/// the same loop, the same table, the same seed and read-out.
fn native_dp(zr: &[F128], rho: &[F128], sigma: &[F128], m: usize) -> F128 {
    let sparse = assist_sparse_transitions();
    let mut g = [F128::ZERO; 4];
    g[STATE_SUCCESS] = F128::ONE;
    for layer in (0..=m).rev() {
        let (za, rb) = (pb(zr, layer), pb(rho, layer));
        let (rc, rd) = (sigma[2 * layer], sigma[2 * layer + 1]);
        let one = F128::ONE;
        let (m1, m2) = (za * rb, rc * rd);
        let eq4 = [one + za + rb + m1, za + m1, rb + m1, m1];
        let e = [one + rc + rd + m2, rc + m2, rd + m2, m2];
        let mut prev = [F128::ZERO; 4];
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                prev[s] += e[cd] * (eq4[i0] * g[o0] + eq4[i1] * g[o1]);
            }
        }
        g = prev;
    }
    g[STATE_INITIAL]
}

/// Two statements' DP chains through the gate, proven and verified over the
/// circuit path; the published read-outs must equal the native replica, and
/// every intermediate is pinned by the relation (perturbation check).
#[test]
fn assist_layer_gate_chains_match_the_native_dp() {
    let (nu, m, n_row) = (9usize, 15usize, 4usize);
    let mut rng = Rng::new(0xA51_57D0);
    let statements: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..2)
        .map(|_| {
            (
                (0..n_row).map(|_| rng.f128()).collect(),
                (0..m).map(|_| rng.f128()).collect(),
                (0..2 * (m + 1)).map(|_| rng.f128()).collect(),
            )
        })
        .collect();

    let mut sb = ShapeBuilder::new(nu);
    let slot = sb.slot(AssistLayerGate::new());
    let mut vals: Vec<F128> = Vec::new();
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let mut outs = Vec::new();
    for (zr, rho, sigma) in &statements {
        let mut g = [zw, zw, zw, zw];
        g[STATE_SUCCESS] = ow;
        for layer in (0..=m).rev() {
            let mut a_in = g.to_vec();
            for v in [
                pb(zr, layer),
                pb(rho, layer),
                sigma[2 * layer],
                sigma[2 * layer + 1],
            ] {
                vals.push(v);
                a_in.push(sb.public_input());
            }
            a_in.push(ow);
            let o = sb.gate(slot, &a_in);
            g = [o[0], o[1], o[2], o[3]];
        }
        outs.push(g[STATE_INITIAL]);
    }
    // Publish only after ALL public inputs are declared — `built.public`
    // is declaration-ordered (the MVP-7 gotcha).
    for &w in &outs {
        sb.publish(w);
    }
    let shape = sb.finish().expect("valid assist DP circuit");
    let built = shape.run(&vals, &[]);

    // The published read-outs are the native DP values.
    for (i, (zr, rho, sigma)) in statements.iter().enumerate() {
        let want = native_dp(zr, rho, sigma, m);
        let got = built.public[built.public.len() - statements.len() + i];
        assert_eq!(got, want, "statement {i} DP read-out");
    }

    // The relation pins the intermediates: perturbing any single computed
    // column breaks satisfaction (the leaf-eval discipline).
    let gate = AssistLayerGate::new();
    let el = match &built.witnesses[shape.registry_slot(slot)] {
        SlotWitness::Element(z) => z.clone(),
        other => panic!("assist slot produced {other:?}"),
    };
    let rows = 2 * (m + 1);
    assert!(gate.ty.satisfies(&el, nu, rows), "honest DP witness");
    for col in [9usize, 10, 13, 17, 33, AL_OUT0] {
        let mut bad = el.clone();
        bad[col << nu] += F128::ONE;
        assert!(
            !gate.ty.satisfies(&bad, nu, rows),
            "column {col} is not constrained"
        );
    }

    // ---- prove / verify over the element-only circuit path ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut ch,
    );
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the assist DP chain verifies");

    // A wrong published read-out must be rejected.
    let mut bad = built.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &bad,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered DP read-out must be rejected"
    );
}
