//! Fiat-Shamir challenger interfaces and implementations.

#[cfg(any(test, feature = "unsound-challenger"))]
pub use flock_transcript::challenger::RandomChallenger;
#[cfg(feature = "hash-count")]
pub use flock_transcript::challenger::fs_count;
pub use flock_transcript::challenger::{
    Challenger, FsChallenger, KIND_NONE, KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL,
    OP_OBSERVE, OP_SQUEEZE, POW_SQUEEZE_COUNTER_TAG, grinding_bits_for_degree,
    has_leading_zero_bits, pow_has_leading_zero_bits, pow_squeeze_counter,
};
