// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from binius64's `crates/math/src/tensor_algebra.rs`
// (https://github.com/binius-zk/binius64), specialized to `F = F_2`,
// `FE = F_{2^128}`.

//! Tensor algebra over `F_{2^128} ⊗_{F_2} F_{2^128}`.
//!
//! An element is a length-128 vector of `F128` (the "vertical-subring" elements
//! in DP24 nomenclature). Conceptually it's a 128×128 F_2 matrix, where row `i`
//! is `elems[i]` viewed via its bit-decomposition in the GHASH polynomial
//! basis (`bit_j(elems[i])` = coefficient of `γ^i ⊗ γ^j` in the tensor algebra).
//!
//! Used by the verifier's polylog `eval_rs_eq` (DP24 §1.3, Figure 3).

use crate::field::{F128, F256};
use core::ops::{Add, AddAssign};

/// The degree of `F_{2^128}` over `F_2`.
pub const DEGREE: usize = 128;

/// An element of `F_{2^128} ⊗_{F_2} F_{2^128}`, stored as 128 `F128` elements
/// (the vertical-subring decomposition).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAlgebra {
    /// Length-128 vector. `elems[i]` is the coefficient of `γ^i` in the
    /// vertical basis decomposition.
    pub elems: Vec<F128>,
}

impl TensorAlgebra {
    /// All-zero element.
    pub fn zero() -> Self {
        Self {
            elems: vec![F128::ZERO; DEGREE],
        }
    }

    /// Multiplicative identity: `1 ⊗ 1`.
    pub fn one() -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = F128::ONE;
        Self { elems }
    }

    /// Embed `x ∈ F_{2^128}` into the vertical subring: returns `1 ⊗ x`.
    pub fn from_vertical(x: F128) -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = x;
        Self { elems }
    }

    /// Multiply by an element of the vertical subring: each `elems[i]` is
    /// scaled by `scalar` in `F_{2^128}`.
    pub fn scale_vertical(mut self, scalar: F128) -> Self {
        for e in self.elems.iter_mut() {
            *e *= scalar;
        }
        self
    }

    /// Multiply by an element of the horizontal subring. Implemented as
    /// `transpose ∘ scale_vertical ∘ transpose`.
    pub fn scale_horizontal(self, scalar: F128) -> Self {
        self.transpose().scale_vertical(scalar).transpose()
    }

    /// Transpose the tensor algebra element: swap vertical and horizontal
    /// subring roles. Concretely, after transpose, `bit_j(elems'[i]) =
    /// bit_i(elems[j])` for all `i, j ∈ [0, 128)`.
    pub fn transpose(mut self) -> Self {
        square_transpose(&mut self.elems);
        self
    }

    /// Fold the tensor algebra element to a single `F128` by scaling rows with
    /// `coeffs` (length 128) and summing.
    ///
    /// Computes `Σ_i coeffs[i] · transpose(self).elems[i]`.
    pub fn fold_vertical(self, coeffs: &[F128]) -> F128 {
        assert_eq!(
            coeffs.len(),
            DEGREE,
            "fold_vertical: coeffs.len() must be 128"
        );
        let transposed = self.transpose();
        let mut acc = F128::ZERO;
        for (e, c) in transposed.elems.iter().zip(coeffs.iter()) {
            acc += *e * *c;
        }
        acc
    }
}

impl Add<&TensorAlgebra> for TensorAlgebra {
    type Output = TensorAlgebra;
    fn add(mut self, rhs: &TensorAlgebra) -> TensorAlgebra {
        self += rhs;
        self
    }
}

impl AddAssign<&TensorAlgebra> for TensorAlgebra {
    fn add_assign(&mut self, rhs: &TensorAlgebra) {
        for (a, b) in self.elems.iter_mut().zip(rhs.elems.iter()) {
            *a = *a + *b;
        }
    }
}

/// An element of `F128 tensor F256`, represented as two ordinary tensor
/// values in the canonical quadratic-extension basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAlgebra256 {
    pub c0: TensorAlgebra,
    pub c1: TensorAlgebra,
}

impl TensorAlgebra256 {
    pub fn one() -> Self {
        Self {
            c0: TensorAlgebra::one(),
            c1: TensorAlgebra::zero(),
        }
    }

    pub fn scale_vertical_base(mut self, scalar: F128) -> Self {
        for value in &mut self.c0.elems {
            *value *= scalar;
        }
        for value in &mut self.c1.elems {
            *value *= scalar;
        }
        self
    }

    pub fn scale_horizontal_extension(self, scalar: F256) -> Self {
        let mut sum = self.c0.clone();
        sum += &self.c1;
        let p0 = self.c0.scale_horizontal(scalar.c0);
        let p1 = self.c1.scale_horizontal(scalar.c1);
        let p2 = sum.scale_horizontal(scalar.c0 + scalar.c1);
        let mut c0 = p0.clone();
        c0 += &p1
            .clone()
            .scale_horizontal(crate::field::QUADRATIC_NONRESIDUE);
        let mut c1 = p2;
        c1 += &p0;
        Self { c0, c1 }
    }

    pub fn fold(self, coeffs: &[F128]) -> F256 {
        F256::new(self.c0.fold_vertical(coeffs), self.c1.fold_vertical(coeffs))
    }
}

impl AddAssign<&TensorAlgebra256> for TensorAlgebra256 {
    fn add_assign(&mut self, rhs: &TensorAlgebra256) {
        self.c0 += &rhs.c0;
        self.c1 += &rhs.c1;
    }
}

/// In-place 128×128 F_2 matrix transpose of the F128 coefficient table.
///
/// On input: `elems[i]` viewed as a 128-bit row; bit `j` is the F_2 coefficient
/// at position `(i, j)`.
/// On output: bit `j` of `elems[i]` becomes the old bit `i` of `elems[j]`.
///
/// V1 implementation: naive O(D²) bit-scan. Each of 128² output bits is read
/// from exactly one input bit.
fn square_transpose(elems: &mut [F128]) {
    assert_eq!(
        elems.len(),
        DEGREE,
        "square_transpose: input must be length 128"
    );

    let mut out = [F128::ZERO; DEGREE];
    for j in 0..DEGREE {
        let src_bit = |k: usize| -> u64 {
            if j < 64 {
                (elems[k].lo >> j) & 1
            } else {
                (elems[k].hi >> (j - 64)) & 1
            }
        };
        let mut lo: u64 = 0;
        let mut hi: u64 = 0;
        for i in 0..64 {
            lo |= src_bit(i) << i;
        }
        for i in 64..128 {
            hi |= src_bit(i) << (i - 64);
        }
        out[j] = F128 { lo, hi };
    }
    elems.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_rng::Rng;

    /// Site-specific draws kept verbatim from this file's former local `Rng`.
    trait RngExt {
        fn ta(&mut self) -> TensorAlgebra;
    }
    impl RngExt for Rng {
        fn ta(&mut self) -> TensorAlgebra {
            let elems = (0..DEGREE).map(|_| self.f128()).collect();
            TensorAlgebra { elems }
        }
    }

    #[test]
    fn transpose_involution() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..10 {
            let t = rng.ta();
            assert_eq!(t.clone().transpose().transpose(), t);
        }
    }

    #[test]
    fn transpose_bit_semantics() {
        // bit_j(elems[i]) on input becomes bit_i(elems[j]) on output.
        let mut rng = Rng::new(42);
        let original = rng.ta();
        let transposed = original.clone().transpose();

        fn bit(x: F128, b: usize) -> u64 {
            if b < 64 {
                (x.lo >> b) & 1
            } else {
                (x.hi >> (b - 64)) & 1
            }
        }

        for i in 0..DEGREE {
            for j in 0..DEGREE {
                let orig_ij = bit(original.elems[i], j);
                let trans_ji = bit(transposed.elems[j], i);
                assert_eq!(orig_ij, trans_ji, "transpose mismatch at (i={i}, j={j})");
            }
        }
    }

    #[test]
    fn from_vertical_scale_vertical() {
        // from_vertical(x).scale_vertical(y) should equal from_vertical(x*y).
        let mut rng = Rng::new(123);
        for _ in 0..10 {
            let x = rng.f128();
            let y = rng.f128();
            let lhs = TensorAlgebra::from_vertical(x).scale_vertical(y);
            let rhs = TensorAlgebra::from_vertical(x * y);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn scale_horizontal_via_transpose() {
        // scale_horizontal(s) == transpose.scale_vertical(s).transpose by
        // construction, but verify the API works end-to-end.
        let mut rng = Rng::new(456);
        for _ in 0..10 {
            let t = rng.ta();
            let s = rng.f128();
            let via_api = t.clone().scale_horizontal(s);
            let manual = t.transpose().scale_vertical(s).transpose();
            assert_eq!(via_api, manual);
        }
    }

    #[test]
    fn add_is_xor_pairwise() {
        let mut rng = Rng::new(789);
        let a = rng.ta();
        let b = rng.ta();
        let sum = a.clone() + &b;
        for i in 0..DEGREE {
            let expected = F128 {
                lo: a.elems[i].lo ^ b.elems[i].lo,
                hi: a.elems[i].hi ^ b.elems[i].hi,
            };
            assert_eq!(sum.elems[i], expected);
        }
    }

    #[test]
    fn add_zero_is_identity() {
        let mut rng = Rng::new(1011);
        let t = rng.ta();
        let z = TensorAlgebra::zero();
        assert_eq!(t.clone() + &z, t);
    }

    #[test]
    fn one_from_vertical_one() {
        assert_eq!(
            TensorAlgebra::one(),
            TensorAlgebra::from_vertical(F128::ONE)
        );
    }

    #[test]
    fn scale_vertical_distributes_over_add() {
        // (a + b) * s == a*s + b*s
        let mut rng = Rng::new(1213);
        let a = rng.ta();
        let b = rng.ta();
        let s = rng.f128();
        let lhs = (a.clone() + &b).scale_vertical(s);
        let rhs = a.scale_vertical(s) + &b.scale_vertical(s);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn fold_vertical_with_zero_coeffs_is_zero() {
        let mut rng = Rng::new(1415);
        let t = rng.ta();
        let zeros = vec![F128::ZERO; DEGREE];
        assert_eq!(t.fold_vertical(&zeros), F128::ZERO);
    }
}
