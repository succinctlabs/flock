//! Where the verifier's arithmetic actually goes.
//!
//!   cargo bench --bench verifier_mul_count --features mul-count
//!
//! The recursion circuit pays one element-class constraint per multiplication
//! and one per inversion. Natively an inversion is `x^(2^128−2)` — about 255
//! multiplications — so **the native profile and the circuit profile rank
//! routines differently**, and both are reported. Ranking circuit work by
//! native timing is the mistake this exists to prevent.
//!
//! Two measurements:
//!
//! 1. **Whole verify**, as the denominator. Everything else is a fraction of
//!    this.
//! 2. **Per routine**, called standalone at the parameters the verifier
//!    actually uses (`K_SKIP = 6`, so `ell = 64`). Attribution without
//!    threading counters through the verifier.
//!
//! Counters are global and relaxed, so the standalone measurements run
//! single-threaded and one at a time.

use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::field::gf2_128::op_count::{self, Snapshot};
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::PcsParams;
use flock_prover::prover::{self, UnionElementSlotInput};
use flock_prover::schedule::{Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier;
use std::sync::Arc;

const DOMAIN: &[u8] = b"flock-union-element-v0";
const K_SKIP: usize = 6;

use flock_core::test_rng::Rng;

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
    z
}

fn row(label: &str, s: Snapshot, total: Option<u64>) {
    let cc = s.circuit_constraints();
    let share = match total {
        Some(t) if t > 0 => format!("{:>6.1}%", 100.0 * cc as f64 / t as f64),
        _ => "     —".to_string(),
    };
    println!(
        "  {label:<44} {:>10} {:>8} {:>12} {share}",
        s.muls_excluding_inv(),
        s.invs,
        cc,
    );
}

fn header() {
    println!(
        "  {:<44} {:>10} {:>8} {:>12} {:>7}",
        "routine", "muls", "invs", "constraints", "share"
    );
    println!("  {}", "-".repeat(85));
}

/// `(full verify, deferred verify)` for an element-only union proof.
fn measure_whole_verify() -> (Snapshot, Snapshot) {
    let nu = 12usize;
    let kappas = [3usize];
    let counts = [1usize << 12];
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));
    let mut rng = Rng::new(0xC057_0001);

    let tys: Vec<Arc<ElementTableType>> = kappas.iter().map(|&k| gate_block(k, w0, w1)).collect();
    let registry = Registry::new(
        tys.iter().map(|t| TableType::element(t.clone())).collect(),
        nu,
    );
    let slot_tys: Vec<Arc<ElementTableType>> = registry
        .element_types()
        .iter()
        .map(|t| match &t.class {
            flock_core::schedule::TableClass::LargeField(e) => e.clone(),
            _ => unreachable!(),
        })
        .collect();
    let union = UnionInstance::new(&registry, counts.to_vec());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };

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
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_mixed_class(
        &union,
        &pcs_params,
        Vec::new(),
        element_slots,
        &mut ch_p,
    );

    let mut ch_v = FsChallenger::new(DOMAIN);
    let (_, full) = op_count::measure(|| {
        verifier::verify_ligerito_union_mixed_class(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_v,
        )
        .expect("honest proof verifies")
    });

    // The SAME proof through the deferred entry — the one a recursion circuit
    // replays, which emits the matrix claims instead of evaluating them. The
    // difference is exactly the O(nnz) matrix work that folding removes, and
    // it is the reason a whole-verify number overstates what recursion pays.
    let mut ch_d = FsChallenger::new(DOMAIN);
    let (_, deferred) = op_count::measure(|| {
        verifier::verify_ligerito_union_mixed_class_deferred(
            &union,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch_d,
        )
        .expect("honest proof verifies")
    });
    (full, deferred)
}

/// A BLAKE3 single-table proof — the **boolean** class, which is what runs the
/// univariate-skip zerocheck. The element class has no in-word structure to
/// fold, so an element-only proof never touches that interpolation at all;
/// measuring only element-only would attribute the skip's cost to a verify
/// that does not pay it.
fn measure_boolean_verify(n_blocks: usize) -> Snapshot {
    use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
    let setup = Blake3Setup::with_log_inv_rate(n_blocks, 1);
    let mut rng = Rng::new(0xC057_0003);
    let blocks: Vec<Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u64() as u32);
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u64() as u32);
            (cv, m, rng.next_u64(), 64u32, 11u32)
        })
        .collect();

    let mut ch_p = FsChallenger::new(b"flock-mul-count");
    let (proof, commitment, _) = setup.prove_fast(&blocks, &mut ch_p);

    let mut ch_v = FsChallenger::new(b"flock-mul-count");
    let (_, snap) = op_count::measure(|| {
        setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("honest proof verifies")
    });
    snap
}

fn main() {
    use flock_core::zerocheck::multilinear::{
        interpolate_at_z_combined, interpolate_at_z_on_lambda, lagrange_weights_naive,
    };
    let _ = flock_prover::init_perf_thread_pool();

    println!("\n=== whole verify ===");
    header();
    let (elem_full, elem_deferred) = measure_whole_verify();
    row("element-only, full verify", elem_full, None);
    row(
        "element-only, DEFERRED (recursion target)",
        elem_deferred,
        None,
    );
    println!(
        "    -> the matrix work folding removes: {} constraints ({:.0}% of the full verify)",
        elem_full.circuit_constraints() - elem_deferred.circuit_constraints(),
        100.0 * (elem_full.circuit_constraints() - elem_deferred.circuit_constraints()) as f64
            / elem_full.circuit_constraints() as f64
    );
    let boolean = measure_boolean_verify(256);
    row("BLAKE3 boolean (K=256, m=22)", boolean, None);
    println!(
        "\n  The element class has no in-word structure to fold, so an\n  \
         element-only verify never runs the univariate skip. The boolean\n  \
         verify is the one that pays for it — use that denominator."
    );
    let total = boolean;
    let denom = total.circuit_constraints();

    println!(
        "\n=== univariate-skip interpolation (K_SKIP={K_SKIP}, ell={}) ===",
        1 << K_SKIP
    );
    header();
    let mut rng = Rng::new(0xC057_0002);
    let z = rng.f128();
    let ell = 1usize << K_SKIP;
    let values: Vec<F128> = (0..ell).map(|_| rng.f128()).collect();

    let (_, s) = op_count::measure(|| lagrange_weights_naive(K_SKIP, z));
    row("lagrange_weights_naive", s, Some(denom));
    let (_, s) = op_count::measure(|| interpolate_at_z_on_lambda(&values, K_SKIP, z));
    row("interpolate_at_z_on_lambda", s, Some(denom));
    let (_, s) = op_count::measure(|| interpolate_at_z_combined(&values, K_SKIP, z));
    row("interpolate_at_z_combined", s, Some(denom));

    let (_, a) = op_count::measure(|| interpolate_at_z_on_lambda(&values, K_SKIP, z));
    let (_, b) = op_count::measure(|| interpolate_at_z_combined(&values, K_SKIP, z));
    let (_, c) = op_count::measure(|| lagrange_weights_naive(K_SKIP, z));
    let round1 = a.circuit_constraints() + b.circuit_constraints() + c.circuit_constraints();
    println!(
        "\n  zerocheck round 1 + the lincheck's skip weights = {round1} constraints \
         ({:.1}% of a boolean verify).",
        100.0 * round1 as f64 / denom as f64
    );
    println!(
        "  These routines are called from ~a dozen sites (lincheck, pcs.rs,\n  \
         ring_switch.rs per claim), so their total contribution exceeds this."
    );
}
