use super::artin_schreier::ArtinSchreierSolver;
use super::base_evaluator::{base_evaluation_functional, evaluate_base_functional};
use super::constants::PRODUCT_MESSAGE_BITS;
use super::constants::{BASE_X_POWER_COUNT, FOUR_RUSSIANS_BLOCK_BITS};
use super::constants::{GAMMA_GROUP_COUNT, MAX_X_DEGREE, X_POWER_COUNT};
use super::evaluator::{
    EvaluationPoint, eval_poly_mask, evaluate_product_functional, product_evaluation_functional,
    x_powers, y_powers,
};
use super::field::{F128, F128Ext};
use super::messages::{BaseMessage, ProductMessage};
use super::product::{extended_base_product_message, product_code_message};
use super::sage_data::{ARTIN_SCHREIER_RHS, GAMMA_SLOT_MASKS, PRODUCT_DENOMINATOR};
use super::sampling::{eval_base_rational_function, sample_random_evaluation_point};
use super::tables::TABLES;
use super::try_evaluation_point;
use super::{RngCore, Sha256Rng};
use std::collections::HashSet;

#[test]
fn product_message_has_expected_prefix() {
    let mut rng = Sha256Rng::seed_from_u64(1);
    for _ in 0..128 {
        let left = BaseMessage::random(&mut rng);
        let right = BaseMessage::random(&mut rng);
        let product = product_code_message(left, right);
        assert_eq!(product.limbs[0], left.0 & right.0);
    }
}

#[test]
fn sampled_points_are_evaluable() {
    let mut rng = Sha256Rng::seed_from_u64(2);
    for _ in 0..8 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        assert!(sampled_point_satisfies_equations(&point));

        let x_powers = x_powers(point.x);
        assert!(!eval_poly_mask(TABLES.product_denominator, &x_powers).is_zero());

        let functional = product_evaluation_functional(&point).expect("functional");
        assert!(functional.iter().any(|value| !value.is_zero()));
        assert_eq!(
            evaluate_product_functional(&functional, &ProductMessage::default()),
            F128::ZERO
        );
    }
}

#[test]
fn artin_schreier_solver_round_trip() {
    let solver = ArtinSchreierSolver::new();
    let mut rng = Sha256Rng::seed_from_u64(3);
    for _ in 0..128 {
        let z = F128::random(&mut rng);
        let rhs = z.square() + z;
        let recovered = solver.solve(rhs).expect("solvable rhs");
        assert_eq!(recovered.square() + recovered, rhs);
    }
}

#[test]
fn field_helpers_round_trip() {
    let mut rng = Sha256Rng::seed_from_u64(4);
    for _ in 0..512 {
        let left = F128::random(&mut rng);
        assert_eq!(left.square(), left * left);
        if !left.is_zero() {
            assert_eq!(left * left.inverse().unwrap(), F128::ONE);
        }
    }
}

#[test]
fn product_message_is_bilinear_and_symmetric() {
    let mut rng = Sha256Rng::seed_from_u64(5);
    for _ in 0..256 {
        let a = BaseMessage::random(&mut rng);
        let b = BaseMessage::random(&mut rng);
        let c = BaseMessage::random(&mut rng);

        assert_eq!(product_code_message(a, b), product_code_message(b, a));
        assert_eq!(
            product_code_message(BaseMessage(a.0 ^ b.0), c),
            xor_product_messages(product_code_message(a, c), product_code_message(b, c))
        );
        assert_eq!(
            product_code_message(a, BaseMessage(b.0 ^ c.0)),
            xor_product_messages(product_code_message(a, b), product_code_message(a, c))
        );
        assert_eq!(
            product_code_message(BaseMessage::default(), a),
            ProductMessage::default()
        );
    }
}

#[test]
fn functional_evaluation_is_linear() {
    let mut rng = Sha256Rng::seed_from_u64(5);
    for _ in 0..4 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let functional = product_evaluation_functional(&point).expect("functional");
        for _ in 0..64 {
            let left = ProductMessage::random(&mut rng);
            let right = ProductMessage::random(&mut rng);
            let sum = xor_product_messages(left, right);
            assert_eq!(
                evaluate_product_functional(&functional, &sum),
                evaluate_product_functional(&functional, &left)
                    + evaluate_product_functional(&functional, &right)
            );
        }
    }
}

#[test]
fn rust_product_identity_residuals_are_zero_at_sampled_points() {
    let mut rng = Sha256Rng::seed_from_u64(6);
    for _ in 0..4 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let functional = product_evaluation_functional(&point).expect("functional");
        for _ in 0..32 {
            let left = BaseMessage::random(&mut rng);
            let right = BaseMessage::random(&mut rng);
            let left_value =
                evaluate_product_functional(&functional, &extended_base_product_message(left));
            let right_value =
                evaluate_product_functional(&functional, &extended_base_product_message(right));
            let product_value =
                evaluate_product_functional(&functional, &product_code_message(left, right));
            assert_eq!((left_value * right_value) + product_value, F128::ZERO);
        }
    }
}

#[test]
fn random_nonzero_product_messages_evaluate_nonzero_at_sampled_points() {
    let mut rng = Sha256Rng::seed_from_u64(7);
    for _ in 0..4 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let functional = product_evaluation_functional(&point).expect("functional");
        for _ in 0..64 {
            let message = random_nonzero_product_message(&mut rng);
            assert_ne!(
                evaluate_product_functional(&functional, &message),
                F128::ZERO
            );
        }
    }
}

#[test]
fn base_evaluator_matches_product_path_at_sampled_points() {
    // Mirror of the Sage audit's check_base_evaluator_matches_product: the
    // direct base evaluator C(m)(P) must equal C_*(R*m)(P) for every base
    // message m at every point. extended_base_product_message(m) is exactly
    // R*m, so dotting it against the product functional is the C_*(R*m) path.
    let mut rng = Sha256Rng::seed_from_u64(11);
    for _ in 0..32 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let base_functional = base_evaluation_functional(&point).expect("base functional");
        let product_functional = product_evaluation_functional(&point).expect("product functional");
        for _ in 0..32 {
            let message = BaseMessage::random(&mut rng);
            let direct = evaluate_base_functional(&base_functional, &message);
            let via_product = evaluate_product_functional(
                &product_functional,
                &extended_base_product_message(message),
            );
            assert_eq!(direct, via_product);
        }
    }
}

#[test]
fn base_functional_evaluation_is_linear() {
    let mut rng = Sha256Rng::seed_from_u64(12);
    for _ in 0..4 {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let functional = base_evaluation_functional(&point).expect("base functional");
        for _ in 0..64 {
            let left = BaseMessage::random(&mut rng);
            let right = BaseMessage::random(&mut rng);
            let sum = BaseMessage(left.0 ^ right.0);
            assert_eq!(
                evaluate_base_functional(&functional, &sum),
                evaluate_base_functional(&functional, &left)
                    + evaluate_base_functional(&functional, &right)
            );
        }
    }
}

fn sampled_point_satisfies_equations(point: &EvaluationPoint) -> bool {
    let x = point.x;
    let y = point.y;
    let x2 = x.square();
    let x3 = x2 * x;
    let x4 = x2.square();
    let x7 = x4 * x2 * x;
    let y2 = y.square();
    let y4 = y2.square();
    let base = y4 + ((x + F128::ONE) * y2) + ((x3 + x) * y) + x7 + x3;
    if !base.is_zero() {
        return false;
    }

    let tables = &*TABLES;
    let x_powers = x_powers(x);
    let y_powers = y_powers(y);
    for (z, rhs) in [point.z1, point.z2, point.z3]
        .into_iter()
        .zip(tables.artin_schreier_rhs.iter())
    {
        let Some(rhs_value) = eval_base_rational_function(rhs, &x_powers, &y_powers) else {
            return false;
        };
        if (z.square() + z) != rhs_value {
            return false;
        }
    }
    true
}

fn xor_product_messages(left: ProductMessage, right: ProductMessage) -> ProductMessage {
    let mut out = ProductMessage {
        limbs: [
            left.limbs[0] ^ right.limbs[0],
            left.limbs[1] ^ right.limbs[1],
            left.limbs[2] ^ right.limbs[2],
            left.limbs[3] ^ right.limbs[3],
        ],
    };
    out.limbs[3] &= (1u64 << 30) - 1;
    out
}

fn random_nonzero_product_message(rng: &mut impl RngCore) -> ProductMessage {
    loop {
        let message = ProductMessage::random(rng);
        if message.limbs.iter().any(|&limb| limb != 0) {
            return message;
        }
    }
}

/// Structural precondition of the minimum-distance argument's zero-count half
/// (`F2_human_audit/check_F2_product_properties.sage`, half (A)): the 222 gamma
/// functions are F-linearly independent, i.e. `w != 0 => Gamma_w != 0`.
///
/// Equivalently, the 222-column matrix whose rows are the evaluation
/// functionals `phi_P` at distinct points `P` has rank 222 -- so no nonzero
/// codeword `w` is orthogonal to every `phi_P`. We sample more than 222 points
/// and check the rank over GF(2^128) is exactly 222. The empirical sibling
/// `random_nonzero_product_messages_evaluate_nonzero_at_sampled_points` is the
/// one-sided shadow of this; the rank check pins the full structural fact.
#[test]
fn product_evaluation_functionals_have_full_rank_222() {
    const N_POINTS: usize = 256;
    let mut rng = Sha256Rng::seed_from_u64(13);
    let mut rows: Vec<[F128; PRODUCT_MESSAGE_BITS]> = Vec::with_capacity(N_POINTS);
    for _ in 0..N_POINTS {
        let point = sample_random_evaluation_point(&mut rng).expect("sample point");
        let functional = product_evaluation_functional(&point).expect("functional");
        let mut row = [F128::ZERO; PRODUCT_MESSAGE_BITS];
        for (slot, value) in functional.iter().enumerate() {
            row[slot] = *value;
        }
        rows.push(row);
    }
    assert_eq!(gf128_row_rank(rows), PRODUCT_MESSAGE_BITS);
}

/// Degree-bound guard: the explicit constants the minimum-distance bound feeds
/// on (`check_F2_product_properties.sage`, "Minimum-distance argument") are
/// derived back from the baked `sage_data` and checked to still equal the
/// values the proof uses. If a data regression raised a gamma's x-degree or the
/// denominator degree, the reconstructed zero bound `Z` -- and the relative
/// distance inequality -- would change here.
#[test]
fn baked_degrees_match_minimum_distance_constants() {
    // (1) Max x-degree of any gamma numerator, read straight off the per-slot
    // masks: slot `s` carries x-degree `s % X_POWER_COUNT` of gamma group
    // `s / X_POWER_COUNT`. This is the "x-degree at most 49" input to the bound.
    assert_eq!(GAMMA_SLOT_MASKS.len(), GAMMA_GROUP_COUNT * X_POWER_COUNT);
    let mut gamma_max_x_degree = 0usize;
    for (slot, mask) in GAMMA_SLOT_MASKS.iter().enumerate() {
        if mask.iter().any(|&limb| limb != 0) {
            gamma_max_x_degree = gamma_max_x_degree.max(slot % X_POWER_COUNT);
        }
    }
    assert_eq!(gamma_max_x_degree, MAX_X_DEGREE);
    assert_eq!(MAX_X_DEGREE, 49);

    // (2) The common evaluator denominator H has x-degree 49.
    let h_degree = mask_degree(PRODUCT_DENOMINATOR);
    assert_eq!(h_degree, 49);

    // (3) The Artin-Schreier reduction denominators. The two distinct moduli are
    // (x^10+x^4+1) [0x411, degree 10] and (x+1)(x^10+x^4+1) [0xc33, degree 11];
    // the structural excluded locus has degree deg((x+1)(x^10+x^4+1)) = 11.
    let mut as_denom_degree = 0usize;
    for row in ARTIN_SCHREIER_RHS.iter() {
        for &(_numerator, denominator) in row.iter() {
            as_denom_degree = as_denom_degree.max(mask_degree(denominator));
        }
    }
    assert_eq!(as_denom_degree, 11);
    let structural_excluded_degree = as_denom_degree;

    // Reassemble the elementary zero bound Z from these degrees exactly as the
    // Sage audit does: cover degree 32; matrix entry x-degree <= 49 + 31;
    // deg(E_w) <= 32*80; excluded x0 <= H-degree + structural-degree; finally
    // Z <= 32*deg(E_w) + 32*excluded.
    const COVER_DEGREE: usize = 32; // 4 (y) * 2 * 2 * 2 (z1, z2, z3)
    const REDUCTION_DENOM_TOTAL_DEGREE: usize = 31; // powers of (x^10+x^4+1) and (x+1)
    let entry_degree = gamma_max_x_degree + REDUCTION_DENOM_TOTAL_DEGREE;
    assert_eq!(entry_degree, 80);
    let norm_degree = COVER_DEGREE * entry_degree; // deg(E_w) bound
    assert_eq!(norm_degree, 2560);
    let excluded = h_degree + structural_excluded_degree;
    assert_eq!(excluded, 60);
    let z_max_zeros = COVER_DEGREE * norm_degree + COVER_DEGREE * excluded;
    assert_eq!(z_max_zeros, 83840);

    // Length lower bound |S| >= 2^128 - 190*2^64 - 1952 (Lang-Weil, genus 95),
    // written without overflowing u128 (2^128 - X = u128::MAX - (X - 1)).
    let s_lower: u128 = u128::MAX - (190u128 << 64) - 1951;
    // Relative distance: 1 - delta = Z / |S| < 2^-111  <=>  Z * 2^111 < |S|.
    assert!((z_max_zeros as u128) << 111 < s_lower);
}

/// Degree of a GF(2) polynomial stored as a degree-bitmask (highest set bit).
fn mask_degree(mask: u64) -> usize {
    assert!(mask != 0, "zero mask has no degree");
    (63 - mask.leading_zeros()) as usize
}

/// Rank over GF(2^128) of a set of 222-wide rows, by Gauss-Jordan elimination.
fn gf128_row_rank(mut rows: Vec<[F128; PRODUCT_MESSAGE_BITS]>) -> usize {
    let cols = PRODUCT_MESSAGE_BITS;
    let nrows = rows.len();
    let mut rank = 0usize;
    for col in 0..cols {
        let pivot = (rank..nrows).find(|&r| !rows[r][col].is_zero());
        let Some(pivot) = pivot else { continue };
        rows.swap(rank, pivot);
        let inv = rows[rank][col].inverse().expect("nonzero pivot");
        for r in 0..nrows {
            if r != rank && !rows[r][col].is_zero() {
                let factor = rows[r][col] * inv;
                for c in col..cols {
                    let term = factor * rows[rank][c];
                    rows[r][c] += term;
                }
            }
        }
        rank += 1;
        if rank == cols {
            break;
        }
    }
    rank
}

/// The rejection sampler's acceptance rate is the protocol constant
/// 1/(BASE_Y_DEGREE · 2^3) = 1/32 (up to the ~2^-56 Hasse–Weil dust) — the
/// number behind the fused-nonce grinding credit
/// (`zerocheck::ag_skip::AG_SAMPLING_CREDIT_BITS`). 200k attempts give
/// σ ≈ 4·10⁻⁴; the asserted window is ±8σ around 1/32.
#[test]
fn acceptance_rate_is_one_in_32() {
    let mut rng = Sha256Rng::new([0xACu8; 32]);
    let n: u32 = 200_000;
    let mut ok: u32 = 0;
    for _ in 0..n {
        if try_evaluation_point(&mut rng).is_some() {
            ok += 1;
        }
    }
    let p = f64::from(ok) / f64::from(n);
    assert!(
        (p - 1.0 / 32.0).abs() < 0.0032,
        "acceptance rate {p:.5} strayed from the pinned 1/32"
    );
}

/// Census of the BASE functional's in-circuit cost surfaces (phase D's
/// `emit_ag_lows` sizing): the pushed-monomial count, the total XOR terms
/// across the 64 coordinate masks, and the x-power ladder length.
#[test]
fn base_functional_circuit_census() {
    let l = &TABLES.base_layout;
    let mut xor_terms = 0usize;
    for (blk, masks) in l.block_masks.iter().enumerate() {
        let (s, e) = (
            l.block_coordinate_offsets[blk],
            l.block_coordinate_offsets[blk + 1],
        );
        for &coord in &l.block_coordinates[s..e] {
            xor_terms += masks[coord as usize].count_ones() as usize;
        }
    }
    println!(
        "base layout: input_count {} | xor terms {} | x powers {} | blocks {} (block bits {})",
        l.input_count,
        xor_terms,
        BASE_X_POWER_COUNT,
        l.block_masks.len(),
        FOUR_RUSSIANS_BLOCK_BITS,
    );
    // The in-circuit sharing estimate: per block, each DISTINCT nonzero
    // sub-mask sum is built once (popcount-1 adds), then one add folds it
    // into each affected coordinate.
    let (mut pairs, mut distinct_builds, mut distinct_total) = (0usize, 0usize, 0usize);
    for (blk, masks) in l.block_masks.iter().enumerate() {
        let (s, e) = (
            l.block_coordinate_offsets[blk],
            l.block_coordinate_offsets[blk + 1],
        );
        let mut seen = HashSet::new();
        for &coord in &l.block_coordinates[s..e] {
            let m = masks[coord as usize];
            pairs += 1;
            if seen.insert(m) {
                distinct_total += 1;
                distinct_builds += (m.count_ones() as usize).saturating_sub(1);
            }
        }
    }
    println!(
        "sharing: (block,coord) pairs {} | distinct masks {} (build adds {}) | est rows = pushes {} + builds {} + pairs {} = {}",
        pairs,
        distinct_total,
        distinct_builds,
        l.input_count,
        distinct_builds,
        pairs,
        l.input_count + distinct_builds + pairs,
    );
}
