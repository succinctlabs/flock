//! Fast routines for the F2 genus-95 product-code artifacts.
//!
//! Imported from the `AG_codes` crate (`succinctlabs/AG_codes`).  The Sage
//! files in that repo's `F2_human_audit/` remain the source of truth; the
//! matrices and polynomials they define were parsed once and baked into
//! [`sage_data`] as constants, so this module has no parse-time or I/O cost.
//!
//! Public surface:
//!   - [`product_code_message`] — multiply two base-code messages.
//!   - [`sample_random_evaluation_point`] — sample a random affine point of the
//!     cover (uniform over cover points) where the evaluator denominator is
//!     nonzero.
//!   - [`product_evaluation_functional`] — build the point-eval coefficient
//!     vector (222 GF(2^128) coordinates) at a point.
//!   - [`evaluate_product_functional`] — evaluate a product-code message
//!     against a precomputed functional.
//!   - [`base_evaluation_functional`] / [`evaluate_base_functional`] — the
//!     direct base-code evaluator `C(m)(P)`: 64 coordinates instead of 222,
//!     equivalent to `evaluate_product_functional(_, R*m)` but cheaper.

#[cfg(test)]
pub(crate) use constants::BASE_Y_DEGREE;
pub use rand_core::RngCore;

pub use crate::genus95_curve_code::{
    base_evaluator::{BaseFunctional, base_evaluation_functional, evaluate_base_functional},
    constants::{BASE_MESSAGE_BITS, PRODUCT_MESSAGE_BITS},
    evaluator::{
        EvaluationPoint, ProductFunctional, evaluate_product_functional,
        product_evaluation_functional,
    },
    field::F128,
    messages::{BaseMessage, ProductMessage},
    product::product_code_message,
    rng::{Blake3Rng, FsRng, Sha256Rng},
    sampling::{
        SAMPLE_ATTEMPT_BUDGET, SampleError, evaluation_point_from_nonce,
        evaluation_point_from_nonce_pow, sample_random_evaluation_point, try_evaluation_point,
    },
};
mod artin_schreier;
mod base_evaluator;
mod constants;
mod evaluator;
mod field;
mod messages;
mod product;
mod rng;
#[cfg(target_arch = "aarch64")]
pub mod round1;
#[rustfmt::skip]
mod sage_data;
mod sampling;
#[cfg(target_arch = "aarch64")]
#[rustfmt::skip]
mod slp_derived;
mod tables;

#[cfg(test)]
mod tests;
