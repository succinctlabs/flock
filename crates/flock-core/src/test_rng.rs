//! Re-export of the workspace test RNG, which lives in [`flock_field`] so
//! the crates below `flock-core` can use it too. See that module's docs.
pub use flock_field::test_rng::Rng;
