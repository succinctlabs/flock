//! A Flock circuit that verifies leanXMSS signatures.
//!
//! leanXMSS is the hash-based signature of leanEthereum's leanVM: two layers of
//! target-sum Winternitz one-time keys (58 chains of 4 positions, digit sum
//! 106) under Merkle trees of height 16. [`reference`](mod@reference) is the
//! native scheme, generic over the hash.

pub mod circuit;
pub mod gadgets;
pub mod machine;
pub mod reference;
