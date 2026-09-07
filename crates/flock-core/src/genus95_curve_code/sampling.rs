use blake3::Hasher;
use flock_hash::HashKind;
use rand_core::RngCore;
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::genus95_curve_code::tables::RationalMask;
use crate::{
    challenger::has_leading_zero_bits,
    genus95_curve_code::{
        artin_schreier::ArtinSchreierSolver,
        constants::{BASE_Y_DEGREE, SAMPLE_X_POWER_COUNT},
        evaluator::{EvaluationPoint, eval_poly_mask, x_powers, y_powers},
        field::{F128, F128Ext},
        rng::FsRng,
        tables::TABLES,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleError {
    AttemptsExceeded,
}

/// Attempt budget for [`sample_random_evaluation_point`], and the AG-skip r₁
/// grinding nonce space: a valid nonce is `< SAMPLE_ATTEMPT_BUDGET`.
pub const SAMPLE_ATTEMPT_BUDGET: u32 = 20_000;

/// Sample a random affine point of the F2 genus-95 cover where the product
/// evaluator denominator is nonzero.
pub fn sample_random_evaluation_point(
    rng: &mut impl RngCore,
) -> Result<EvaluationPoint, SampleError> {
    for _ in 0..SAMPLE_ATTEMPT_BUDGET {
        if let Some(point) = try_evaluation_point(rng) {
            return Ok(point);
        }
    }
    Err(SampleError::AttemptsExceeded)
}

/// The per-nonce DRBG seed `H(seed ‖ LE32(nonce))`, where `H` follows the
/// transcript hash `kind`. Shared by the plain and PoW-fused nonce attempts.
fn nonce_seed(seed: &[u8; 32], nonce: u32, kind: HashKind) -> [u8; 32] {
    let mut nonce_seed = [0u8; 32];
    match kind {
        HashKind::Sha256 => {
            let mut h = Sha256::new();
            h.update(seed);
            h.update(nonce.to_le_bytes());
            nonce_seed.copy_from_slice(&h.finalize());
        }
        HashKind::Blake3 => {
            let mut h = Hasher::new();
            h.update(seed);
            h.update(&nonce.to_le_bytes());
            nonce_seed.copy_from_slice(h.finalize().as_bytes());
        }
    }
    nonce_seed
}

/// One attempt from a 32-byte transcript seed and a grinding nonce: the
/// attempt's DRBG is `FsRng::new(kind, H(seed ‖ LE32(nonce)))` where `H` and
/// the DRBG follow the transcript hash `kind` (SHA-256 counter mode or BLAKE3
/// XOF — see [`super::rng::FsRng`]), so each nonce yields an independent
/// uniform draw and no second primitive enters the soundness argument.
/// `None` = this nonce is rejected.
pub fn evaluation_point_from_nonce(
    seed: &[u8; 32],
    nonce: u32,
    kind: HashKind,
) -> Option<EvaluationPoint> {
    try_evaluation_point(&mut FsRng::new(kind, nonce_seed(seed, nonce, kind)))
}

/// [`evaluation_point_from_nonce`] with a FUSED proof-of-work criterion: the
/// nonce is valid only when its DRBG seed `H(seed ‖ LE32(nonce))` ALSO clears
/// the PoW target — at least `pow_bits` leading zero bits of the seed's
/// SECOND 16-byte word (bytes 16..32, MSB-first within each byte: the same
/// predicate word and bit convention as the transcript PoW and the recursion
/// circuit's `PowMaskTable`, so the eventual in-circuit check is a gadget
/// reuse) — both criteria on the same hash, so a prover iterating nonces
/// pays the PoW lottery on every sampling attempt and vice versa.
///
/// Success per nonce is exactly `p · 2^-pow_bits`, where `p` is the sampler's
/// acceptance probability. `p = (1/32)·(1 ± 2^-56)` PROVABLY: the sampler
/// weights every reachable cover point at exactly `1/(2^128 · 4 · 8)` (the
/// `BASE_Y_DEGREE` slot flattening plus the three all-or-nothing
/// Artin–Schreier choice bits — see [`try_evaluation_point`]), and Hasse–Weil
/// bounds the genus-95 cover's point count within `2·95·2^64` of `2^128`
/// (denominator-pole and infinity exclusions are a few hundred points). The
/// rejection sampling therefore contributes `log2(32) = 5` bits of grinding
/// on top of `pow_bits` — a protocol constant, not an empirical estimate —
/// which is how the AG challenge sites reach their strict 128-bit budgets
/// with small explicit PoW (see `zerocheck::ag_skip::AG_SAMPLING_CREDIT_BITS`
/// and the guard tests tying the constants together).
pub fn evaluation_point_from_nonce_pow(
    seed: &[u8; 32],
    nonce: u32,
    kind: HashKind,
    pow_bits: u32,
) -> Option<EvaluationPoint> {
    debug_assert!(pow_bits <= 64);
    let ns = nonce_seed(seed, nonce, kind);
    if !has_leading_zero_bits(&ns[16..32], pow_bits) {
        return None;
    }
    try_evaluation_point(&mut FsRng::new(kind, ns))
}

/// One rejection-sampling attempt: draw `x` and lift `(y, z1, z2, z3)`,
/// rejecting (`None`) at empty fibers, denominator poles, slot overflow, or a
/// failed z-lift. Consumes exactly one attempt's worth of the rng stream.
pub fn try_evaluation_point(rng: &mut impl RngCore) -> Option<EvaluationPoint> {
    let tables = &*TABLES;
    let mut roots = RootList::empty();

    let x = F128::random(rng);
    base_y_roots_for_x_factored(&tables.as_solver, x, &mut roots);
    if roots.is_empty() {
        return None;
    }

    let (sample_x_powers, product_denominator) = sample_x_powers_and_product_denominator(x);
    debug_assert_eq!(
        product_denominator,
        eval_poly_mask(tables.product_denominator, &x_powers(x))
    );
    if product_denominator.is_zero() {
        return None;
    }

    // Uniformize over the fiber. There are at most BASE_Y_DEGREE roots y
    // over a given x, so draw a slot in [0, BASE_Y_DEGREE) and reject this
    // x whenever the slot lands past the actual fiber. This is rejection
    // sampling against a uniform envelope: it flattens the per-point weight
    // to 1/BASE_Y_DEGREE regardless of how many roots x actually has,
    // instead of the 1/fiber_size weighting a scan-and-take-first gives.
    let slot = (rng.next_u64() as usize) % BASE_Y_DEGREE;
    if slot >= roots.len {
        return None;
    }
    let y = roots.values[slot];
    let y_powers = y_powers(y);
    let z_choice_bits = rng.next_u64();

    let mut inverse_cache = SampleInverseCache::empty();
    let mut rhs_coeff_cache = [[F128::ZERO; BASE_Y_DEGREE]; 3];
    let mut rhs_coeff_cached = [false; 3];

    // The z-fiber over (x, y) is all-or-nothing (8 points or 0): the three
    // z-coordinates are independent Artin-Schreier extensions and we keep a
    // point only if all three lift. On any failure reject and redraw x
    // rather than trying another root, so the chosen slot — and hence every
    // cover point — stays equally likely. The three branch bits already
    // sample the 8 lifts uniformly.
    let mut z = [F128::ZERO; 3];
    for i in 0..3 {
        let rhs_coeffs = sample_artin_schreier_rhs_coeffs_cached(
            i,
            &sample_x_powers,
            &mut inverse_cache,
            &mut rhs_coeff_cache,
            &mut rhs_coeff_cached,
        )?;
        let rhs = eval_base_coefficients(rhs_coeffs, &y_powers);
        let mut root = tables.as_solver.solve(rhs)?;
        if ((z_choice_bits >> i) & 1) != 0 {
            root += F128::ONE;
        }
        z[i] = root;
    }

    Some(EvaluationPoint {
        x,
        y,
        z1: z[0],
        z2: z[1],
        z3: z[2],
    })
}

/// Compute the first `SAMPLE_X_POWER_COUNT` powers of `x` together with the
/// product evaluator's common denominator, using a specialized squaring chain.
/// Used only to reject `x` at poles of the denominator before lifting a point.
#[inline(always)]
fn sample_x_powers_and_product_denominator(x: F128) -> ([F128; SAMPLE_X_POWER_COUNT], F128) {
    let mut powers = [F128::ZERO; SAMPLE_X_POWER_COUNT];
    powers[0] = F128::ONE;
    powers[1] = x;
    powers[2] = x.square();
    powers[3] = powers[2] * x;
    powers[4] = powers[2].square();
    powers[5] = powers[4] * x;
    powers[6] = powers[5] * x;
    powers[7] = powers[5] * powers[2];
    powers[8] = powers[4].square();
    powers[9] = powers[8] * x;
    powers[10] = powers[9] * x;
    powers[11] = powers[10] * x;

    let x16 = powers[8].square();
    let x21 = x16 * powers[5];
    let x23 = x21 * powers[2];
    let x31 = x23 * powers[8];
    let x32 = x16.square();
    let x35 = x32 * powers[3];
    let x37 = x32 * powers[5];
    let x45 = x37 * powers[8];
    let x47 = x45 * powers[2];
    let x49 = x47 * powers[2];
    let product_denominator =
        x49 + x47 + x45 + x37 + x35 + x31 + x23 + x21 + powers[9] + powers[7] + powers[5] + x;

    (powers, product_denominator)
}

#[cfg(test)]
pub(crate) fn eval_base_rational_function<const N: usize>(
    coeffs: &[RationalMask; BASE_Y_DEGREE],
    x_powers: &[F128; N],
    y_powers: &[F128; BASE_Y_DEGREE],
) -> Option<F128> {
    let mut out = F128::ZERO;
    for i in 0..BASE_Y_DEGREE {
        let coeff = coeffs[i].eval(x_powers)?;
        out += coeff * y_powers[i];
    }
    Some(out)
}

fn sample_artin_schreier_rhs_coeffs_cached<'a>(
    rhs_index: usize,
    x_powers: &[F128; SAMPLE_X_POWER_COUNT],
    inverse_cache: &mut SampleInverseCache,
    rhs_coeff_cache: &'a mut [[F128; BASE_Y_DEGREE]; 3],
    rhs_coeff_cached: &mut [bool; 3],
) -> Option<&'a [F128; BASE_Y_DEGREE]> {
    if !rhs_coeff_cached[rhs_index] {
        let d0 = inverse_cache.d0(x_powers)?;
        let coeffs = &mut rhs_coeff_cache[rhs_index];
        match rhs_index {
            0 => {
                coeffs[0] = (x_powers[9]
                    + x_powers[6]
                    + x_powers[5]
                    + x_powers[4]
                    + x_powers[3]
                    + x_powers[2])
                    * d0;
                coeffs[1] =
                    (x_powers[5] + x_powers[4] + x_powers[3] + x_powers[2] + F128::ONE) * d0;
                coeffs[2] = (x_powers[4] + x_powers[3] + x_powers[2]) * d0;
                coeffs[3] = (x_powers[3] + F128::ONE) * d0;
            }
            1 => {
                let d1 = inverse_cache.d1(x_powers)?;
                coeffs[0] = (x_powers[8]
                    + x_powers[5]
                    + x_powers[4]
                    + x_powers[3]
                    + x_powers[2]
                    + x_powers[1])
                    * d0;
                coeffs[1] = (x_powers[6] + x_powers[5] + x_powers[2] + x_powers[1]) * d0;
                coeffs[2] = (x_powers[3] + x_powers[1]) * d0;
                coeffs[3] = (x_powers[4] + x_powers[2] + x_powers[1]) * d1;
            }
            2 => {
                coeffs[0] = (x_powers[6] + x_powers[4] + x_powers[3] + x_powers[2]) * d0;
                coeffs[1] = (x_powers[5] + F128::ONE) * d0;
                coeffs[2] = (x_powers[3] + x_powers[2] + x_powers[1] + F128::ONE) * d0;
                coeffs[3] = (x_powers[3] + x_powers[2]) * d0;
            }
            _ => unreachable!(),
        }
        rhs_coeff_cached[rhs_index] = true;
    }
    Some(&rhs_coeff_cache[rhs_index])
}

fn eval_base_coefficients(
    coeffs: &[F128; BASE_Y_DEGREE],
    y_powers: &[F128; BASE_Y_DEGREE],
) -> F128 {
    let mut out = F128::ZERO;
    for i in 0..BASE_Y_DEGREE {
        out += coeffs[i] * y_powers[i];
    }
    out
}

#[derive(Clone, Copy)]
struct SampleInverseCache {
    d0: Option<Option<F128>>,
    d1: Option<Option<F128>>,
}

impl SampleInverseCache {
    fn empty() -> Self {
        Self { d0: None, d1: None }
    }

    fn d0(&mut self, x_powers: &[F128; SAMPLE_X_POWER_COUNT]) -> Option<F128> {
        if let Some(inverse) = self.d0 {
            return inverse;
        }
        let denominator = x_powers[10] + x_powers[4] + F128::ONE;
        let inverse = denominator.inverse();
        self.d0 = Some(inverse);
        inverse
    }

    fn d1(&mut self, x_powers: &[F128; SAMPLE_X_POWER_COUNT]) -> Option<F128> {
        if let Some(inverse) = self.d1 {
            return inverse;
        }
        let denominator =
            x_powers[11] + x_powers[10] + x_powers[5] + x_powers[4] + x_powers[1] + F128::ONE;
        let inverse = denominator.inverse();
        self.d1 = Some(inverse);
        inverse
    }
}

#[derive(Clone, Copy)]
struct RootList {
    values: [F128; 7],
    len: usize,
}

impl RootList {
    fn empty() -> Self {
        Self {
            values: [F128::ZERO; 7],
            len: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, value: F128) {
        debug_assert!(self.len < self.values.len());
        self.values[self.len] = value;
        self.len += 1;
    }

    fn push_unique(&mut self, value: F128) {
        if self.values[..self.len].contains(&value) {
            return;
        }
        self.push(value);
    }
}

fn base_y_roots_for_x_factored(as_solver: &ArtinSchreierSolver, x: F128, roots: &mut RootList) {
    let u = x + F128::ONE;
    if u.is_zero() {
        roots.push(F128::ZERO);
        return;
    }

    let x3 = x.square() * x;
    let Some(t0) = as_solver.solve(x3 + x) else {
        return;
    };
    let Some(inv_u) = u.inverse() else {
        return;
    };
    let x_over_u = x * inv_u;

    for t in [t0, t0 + F128::ONE] {
        let Some(s0) = as_solver.solve(x_over_u * t) else {
            continue;
        };
        let y = u * s0;
        roots.push_unique(y);
        roots.push_unique(y + u);
    }
}
