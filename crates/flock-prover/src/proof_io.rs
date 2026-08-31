//! Serialize / deserialize proofs to bytes (and files).
//!
//! Two bundle types: [`R1csProofBundleLigerito`] for the base R1CS proof and
//! [`MixedProofBundleLigerito`] for the multi-table mixed proof. Both pair a
//! proof with its commitment (which the verifier needs); the mixed bundle
//! additionally carries its registry id + counts vector.
//!
//! On-disk format:
//! ```text
//!   bytes 0..5    "FLOCK"                  (5-byte magic)
//!   byte  5       VERSION                  (currently 22)
//!   bytes 6..7    flavor: 2 = R1cs, 4 = Mixed
//!                 (0/1 reserved: legacy BaseFold; 3 was the retired chain)
//!   bytes 7..     bincode-serialized payload
//! ```
//!
//! Versioning is here to make schema changes detectable cleanly: bump
//! `VERSION` whenever a payload field is added/removed/reordered. Forward
//! compatibility is NOT promised — `from_bytes` of a different version is
//! rejected (`UnsupportedVersion`).
//!
//! ## Round-trip example
//! ```ignore
//! let bundle = R1csProofBundleLigerito { commitment, proof };
//! let bytes = bundle.to_bytes();
//! std::fs::write("proof.bin", &bytes)?;
//! ...
//! let bytes = std::fs::read("proof.bin")?;
//! let bundle = R1csProofBundleLigerito::from_bytes(&bytes)?;
//! // Then call e.g. `setup.verify(&bundle.commitment, &bundle.proof, ...)`.
//! ```

use std::io::{self, Read};
use std::path::Path;

use bincode::Options;
use serde::{Deserialize, Serialize};

use flock_core::pcs::Commitment;

/// Magic bytes prepended to every serialized proof. Lets readers reject
/// random binary data early.
const MAGIC: [u8; 5] = *b"FLOCK";

/// Maximum accepted proof-bundle size, including the seven-byte header.
/// Current bundles are well below this ceiling; bounding both file reads and
/// bincode prevents malformed length prefixes or oversized files from turning
/// verification into an unbounded allocation.
const MAX_BUNDLE_BYTES: usize = 64 * 1024 * 1024;

/// Format version. Bumped on incompatible serialization changes.
/// v20 enables the existing non-Ligerito algebraic grinding schedules for
/// the strict Fast and Slim profiles. No payload fields changed, but their
/// nonce vectors and Fiat--Shamir transcript shapes are incompatible with
/// v19 proofs made under those profiles.
/// v19 moves Ligerito sumcheck claims and challenges to F256. Recursive
/// commitments remain over F128 by splitting each extension word into a
/// coordinate bit, so the transcript, recursive dimensions, final `yr`, and
/// sumcheck-message payload are all incompatible with v18.
/// v18 changes every Johnson/Ligerito query schedule to make the consistency
/// term strictly 128-bit after optional query grinding. No struct changes,
/// but v17 paths/caps and transcript shapes cannot replay under the larger
/// public query counts.
/// v17 adds the claim- and consistency-batching PoW nonce vectors from the
/// Flock paper's Appendix C.3 to `LigeritoProof`.
/// v16 changes the Johnson/Ligerito transcript: L0 batches one additional OOD
/// evaluation before the initial sumcheck message, and every deeper level now
/// carries two OOD evaluations. The proof structs are unchanged, but v15
/// proof bytes cannot be replayed under the two-point transcript.
/// v15: remaining non-Ligerito algebraic grinding. Product-GKR, dense and
/// jagged aggregation folds, chain shift, and Merkle-path shift proofs carry
/// transcript-ordered PoW witnesses. Opening batching now protects all
/// ring-switched and packed-direct coefficients with one PoW and samples the
/// coefficients in one vector squeeze; the multipoint gamma schedule is
/// derived from its actual `K - 1` degree. These proof and transcript changes
/// are incompatible with v14.
/// v14: PCS-transport grinding. Ring-switch proofs carry the PoW witness for
/// their seven-coordinate point; opening proofs carry claim-batching and
/// merged-sumcheck witnesses; the multipoint and Frobenius-anchor proofs
/// carry their protected challenge witnesses. Old payloads cannot be
/// interpreted under the Secure opening policy.
/// v13: Element/dense PIOP grinding. Both the element zerocheck and element
/// lincheck subproofs now carry their transcript-ordered nonce vectors; the
/// Secure profile verifies them before tau, alpha, and every protected round
/// challenge. Old mixed payloads therefore cannot be interpreted safely.
/// v12: Boolean lincheck grinding. `LincheckProof` carries the nonce vector
/// for α batching, every constant-wire β, every degree-two sumcheck round,
/// and the final φ8 skip evaluation. Secure profiles select the schedule from
/// the public PCS profile; old payloads cannot be interpreted safely.
/// v11: Boolean zerocheck grinding. `ZerocheckProof` carries the nonce vector
/// for the initial eq-weighted identity challenge, the univariate-skip
/// challenge, and every multilinear sumcheck round. The verifier selects the
/// required schedule from the public PCS profile, so an old payload could not
/// be interpreted safely even when its nonce vector would be empty.
///
/// v9: the two-product multipoint grouping — packed-direct
/// claims collapse by shared row point into merged-column scalar groups
/// carrying ONE untwisted dual value each
/// (`MultipointTwistedProof.group_values`); ring-switched claims keep 128.
/// The sumcheck becomes two products (`ā·g + b̄·eq(ρ,·)`) and the single
/// anchor binds the whole endpoint sum via closed-form-baked coefficients.
/// The multipoint label bumps to v1 and the values' absorb shrinks from
/// `128·K` to `128·R + P` words. Soundness:
/// docs/multipoint-twisted-assist.tex §"The two-product grouping".
///
/// v8: the multipoint-twisted assist — `MergedOpenProof.
/// frobenius` becomes `MultipointTwistedProof` (128K claimed dual values,
/// m product-sumcheck rounds, one untwisted anchor); the transcript gains
/// the values' absorb + gamma squeeze and loses the per-statement assist
/// rounds. Soundness: docs/multipoint-twisted-assist.tex.
///
/// v10: stratified queries (docs/stratified-queries.tex) — every level's
/// query count decomposes into power-of-two summands, one query per
/// depth-c subtree; the absorbed cap moves to the TOP SET BIT of the
/// count (from ⌈log2 q⌉) and openings carry per-summand path lengths.
/// No struct changed — the vectors are self-describing — but v9 proofs
/// can never verify under the stratified statement, so versioning stays
/// strict.
/// v7: Merkle capping — `Commitment.root`, `LigeritoProof.
/// initial_root`, and `recursive_roots` become cap-node VECTORS (the
/// commitment is the cap layer at depth ⌈log2 q⌉; the transcript absorbs
/// the cap itself), and the per-tree octopus multi-proofs become flat
/// per-query capped paths. The symmetric bookend to v3, which introduced
/// the octopus.
/// v6 switched the Mixed flavor's payload to the MERGED
/// jagged/ring-switch transport ([`MixedProofBundleLigerito`] now carries
/// an `R1csProofMergedLigerito` — design doc §"Capacity-free
/// ring-switching"); the R1cs/Chain flavors' payloads are unchanged, but
/// versioning is strict so v5 files are rejected.
/// v5 added the Mixed flavor (registry id + counts vector +
/// jagged-transport proof). v4 added `ood_values` + `fold_grinding_nonces`
/// to `LigeritoProof` and `profile` to `PcsParams` (Johnson+OOD profiles).
/// v3 restructured `BaseFoldProof`: per-query Merkle paths were replaced by
/// shared octopus multi-proofs (one per Merkle tree). v2 added `HashKind`
/// to the (since-deleted) chain-proof bundle.
// v21 (2026-08-14): the R1cs flavor's payload became the MERGED union
// proof — the standalone hash setups prove over the single-slot union
// commit now (dense stack + integer lanes); the padded-commit
// R1csProofLigerito payload is gone from this flavor.
// v22 (2026-08-27): the profile consolidation (bloat ledger §C). The
// grind-free `fast`/`slim` (+1 rate/level ladder) were deleted and the
// `fast128`/`slim128` schedules took their names: aggressive +2/level
// ladder, 16-bit query PoW per level, larger deep-level batch grinding. No
// payload field changed, but a v21 proof made under `fast`/`slim` carries a
// different query/rate/PoW schedule than this build derives for those
// names, so it cannot be interpreted safely. `fast100`/`slim100`/`secure`
// are byte-for-byte unchanged.
const VERSION: u8 = 22;

/// Flavor discriminator (1 byte). Lets a generic reader peek what kind of
/// bundle a file holds without parsing the payload first (see
/// [`peek_flavor`]). Values 0/1 are reserved: they were the legacy BaseFold
/// R1cs/Chain flavors.
const FLAVOR_R1CS_LIGERITO: u8 = 2;
// Flavor byte 3 was the hash-chain (shift-argument) bundle — the product was
// retired 2026-08-14 with `chain.rs`/`chain_common.rs`; the byte stays
// reserved and now parses as UnknownFlavor.
const FLAVOR_MIXED_LIGERITO: u8 = 4;

/// What kind of bundle a byte buffer holds. Returned by [`peek_flavor`] so
/// generic readers (the CLI) can dispatch before parsing the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleFlavor {
    R1cs,
    Mixed,
}

/// Validate the header (magic + version) and return the bundle flavor,
/// without touching the payload.
pub fn peek_flavor(bytes: &[u8]) -> Result<BundleFlavor, DeserializeError> {
    check_bundle_size(bytes.len())?;
    if bytes.len() < HEADER_LEN {
        return Err(DeserializeError::Truncated);
    }
    if bytes[0..5] != MAGIC {
        return Err(DeserializeError::BadMagic);
    }
    if bytes[5] != VERSION {
        return Err(DeserializeError::UnsupportedVersion(bytes[5]));
    }
    match bytes[6] {
        FLAVOR_R1CS_LIGERITO => Ok(BundleFlavor::R1cs),
        FLAVOR_MIXED_LIGERITO => Ok(BundleFlavor::Mixed),
        other => Err(DeserializeError::UnknownFlavor(other)),
    }
}

/// Header size = 5-byte magic + 1-byte version + 1-byte flavor.
const HEADER_LEN: usize = 7;

/// Errors from `from_bytes` / `read_from_file`.
#[derive(Debug)]
pub enum DeserializeError {
    /// The 5-byte magic prefix did not match `FLOCK`.
    BadMagic,
    /// The version byte didn't match this build's `VERSION`. The number is
    /// the version found in the file.
    UnsupportedVersion(u8),
    /// The flavor byte was neither `2` (R1cs Ligerito) nor `4` (Mixed
    /// Ligerito).
    UnknownFlavor(u8),
    /// `from_bytes` was called with a slice shorter than `HEADER_LEN`.
    Truncated,
    /// The encoded bundle exceeds [`MAX_BUNDLE_BYTES`].
    TooLarge { len: usize, max: usize },
    /// The expected flavor and the file's flavor disagree (e.g. trying to
    /// load a `MixedProofBundleLigerito` from an R1CS bundle file).
    FlavorMismatch { expected: u8, found: u8 },
    /// The bincode-deserialization step failed (corrupted payload, etc.).
    Bincode(bincode::Error),
}

impl std::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic: not a FLOCK proof file"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported version {v} (this build expects {VERSION})")
            }
            Self::UnknownFlavor(v) => write!(f, "unknown flavor byte: {v}"),
            Self::Truncated => write!(f, "input shorter than header ({HEADER_LEN} bytes)"),
            Self::TooLarge { len, max } => {
                write!(f, "proof bundle is {len} bytes; maximum is {max}")
            }
            Self::FlavorMismatch { expected, found } => {
                write!(f, "flavor mismatch: expected {expected}, found {found}")
            }
            Self::Bincode(e) => write!(f, "bincode error: {e}"),
        }
    }
}

impl std::error::Error for DeserializeError {}

impl From<bincode::Error> for DeserializeError {
    fn from(e: bincode::Error) -> Self {
        Self::Bincode(e)
    }
}

/// Bundles a base R1CS proof with its commitment for self-contained
/// serialization. Verification still needs the relevant [`flock_core::r1cs::BlockR1cs`]
/// (or a `*Setup`) on the verifier side — that's a public artifact derived
/// from the setup parameters, not part of the proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofBundleLigerito {
    pub commitment: Commitment,
    pub proof: flock_core::proof::R1csProofMergedLigerito,
}

impl R1csProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        bincode::serialize_into(&mut out, self).expect("bincode serialize R1csProofBundleLigerito");
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS_LIGERITO)?;
        deserialize_payload(payload)
    }
}

/// Bundles a multi-table MIXED proof (wire format v6): the built-in
/// registry id — which pins the FULL registry, type list and uniform
/// capacity `nu` included (see [`crate::mixed::MixedRegistryId`]) — the
/// declared counts vector (one `u64` per type, **in slot order**), the
/// commitment to the dense stack, and the MERGED-transport union proof
/// (design doc §"Capacity-free ring-switching").
/// The statement is well-formedness only (design doc §"Statement,
/// transcript, wire format"): the commitment opens to tables with the
/// declared counts, every declared row satisfying its type's hash relation
/// — no per-invocation I/O binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixedProofBundleLigerito {
    pub registry_id: crate::mixed::MixedRegistryId,
    /// Declared invocation counts, in the registry's slot order.
    pub counts: Vec<u64>,
    pub commitment: Commitment,
    pub proof: flock_core::proof::R1csProofMergedLigerito,
}

impl MixedProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_MIXED_LIGERITO);
        bincode::serialize_into(&mut out, self)
            .expect("bincode serialize MixedProofBundleLigerito");
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_MIXED_LIGERITO)?;
        deserialize_payload(payload)
    }
}

/// Write a mixed bundle to `path`.
pub fn write_mixed_bundle_ligerito_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &MixedProofBundleLigerito,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

/// Read a mixed bundle from `path`.
pub fn read_mixed_bundle_ligerito_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<MixedProofBundleLigerito, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    MixedProofBundleLigerito::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

// ---------------------------------------------------------------------------
// Encoding helpers
// ---------------------------------------------------------------------------

fn write_header(out: &mut Vec<u8>, flavor: u8) {
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(flavor);
}

fn parse_header(bytes: &[u8], expected_flavor: u8) -> Result<&[u8], DeserializeError> {
    check_bundle_size(bytes.len())?;
    if bytes.len() < HEADER_LEN {
        return Err(DeserializeError::Truncated);
    }
    if bytes[0..5] != MAGIC {
        return Err(DeserializeError::BadMagic);
    }
    let v = bytes[5];
    if v != VERSION {
        return Err(DeserializeError::UnsupportedVersion(v));
    }
    let flavor = bytes[6];
    if flavor != FLAVOR_R1CS_LIGERITO && flavor != FLAVOR_MIXED_LIGERITO {
        return Err(DeserializeError::UnknownFlavor(flavor));
    }
    if flavor != expected_flavor {
        return Err(DeserializeError::FlavorMismatch {
            expected: expected_flavor,
            found: flavor,
        });
    }
    Ok(&bytes[HEADER_LEN..])
}

fn check_bundle_size(len: usize) -> Result<(), DeserializeError> {
    if len > MAX_BUNDLE_BYTES {
        Err(DeserializeError::TooLarge {
            len,
            max: MAX_BUNDLE_BYTES,
        })
    } else {
        Ok(())
    }
}

fn deserialize_payload<T: serde::de::DeserializeOwned>(
    payload: &[u8],
) -> Result<T, DeserializeError> {
    Ok(bincode::DefaultOptions::new()
        // Preserve the wire encoding used by bincode's top-level helpers.
        .with_fixint_encoding()
        .with_limit((MAX_BUNDLE_BYTES - HEADER_LEN) as u64)
        .reject_trailing_bytes()
        .deserialize(payload)?)
}

// ---------------------------------------------------------------------------
// File-IO conveniences
// ---------------------------------------------------------------------------

/// Atomically write `bytes` to `path` (write-then-rename via the
/// stdlib — best-effort; on error the rename may leave a temp file behind).
pub fn write_bytes_to_file<P: AsRef<Path>>(path: P, bytes: &[u8]) -> io::Result<()> {
    let path = path.as_ref();
    let tmp = match path.parent() {
        Some(dir) => dir.join(format!(
            ".{}.tmp",
            path.file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("flock-proof")
        )),
        None => Path::new(".flock-proof.tmp").to_path_buf(),
    };
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Read proof bytes from a file without ever buffering more than
/// [`MAX_BUNDLE_BYTES`]. The `take` guard also handles a file growing between
/// its metadata check and the read.
pub fn read_bytes_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let declared_len = file.metadata()?.len();
    if declared_len > MAX_BUNDLE_BYTES as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("proof bundle is {declared_len} bytes; maximum is {MAX_BUNDLE_BYTES}"),
        ));
    }
    let mut bytes = Vec::with_capacity(declared_len as usize);
    file.by_ref()
        .take(MAX_BUNDLE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_BUNDLE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("proof bundle exceeded the {MAX_BUNDLE_BYTES}-byte maximum while reading"),
        ));
    }
    Ok(bytes)
}

/// Combined error returned by file-read helpers: either IO failed or the
/// bytes weren't a valid bundle.
#[derive(Debug)]
pub enum BundleReadError {
    Io(io::Error),
    Deserialize(DeserializeError),
}

impl std::fmt::Display for BundleReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Deserialize(e) => write!(f, "deserialize error: {e}"),
        }
    }
}

impl std::error::Error for BundleReadError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::blake3::{Blake3Setup, Compression, blake3_compress};
    use flock_core::challenger::FsChallenger;

    /// SplitMix64.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Build a small honest BLAKE3 chain (n=8) for the bundle tests.
    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = Rng::new(seed);
        let mut cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
        let cv0 = cv;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
            let counter = 0u64;
            let block_len = 64u32;
            let flags = 0u32;
            blocks.push((cv, m, counter, block_len, flags));
            let st = blake3_compress(&cv, &m, counter, block_len, flags);
            cv = st[0..8].try_into().unwrap();
        }
        (blocks, cv0, cv)
    }

    /// Default Ligerito bundle roundtrip, byte-flip rejection, and file
    /// roundtrip. Requires m ≥ 21 — use n_blocks=256 (m=22 with K_LOG=14).
    #[test]
    #[ignore] // Heavier — run with `cargo test r1cs_bundle_roundtrip -- --ignored --nocapture`
    fn r1cs_bundle_roundtrip() {
        // K=256 → n_log=8 → m=22 with BLAKE3 K_LOG=14 (smallest Ligerito target).
        let setup = Blake3Setup::new(256);
        let (blocks, _, _) = honest_chain(256, 0xDEAD_5170);
        let mut ch = FsChallenger::new(b"flock-proofio-lig");
        let (proof, commitment, _claim) = setup.prove_fast(&blocks, &mut ch);

        let bundle = R1csProofBundleLigerito {
            commitment: commitment.clone(),
            proof: proof.clone(),
        };
        let bytes = bundle.to_bytes();
        assert_eq!(&bytes[0..5], &MAGIC);
        assert_eq!(bytes[5], VERSION);
        assert_eq!(bytes[6], FLAVOR_R1CS_LIGERITO);

        let bundle2 = R1csProofBundleLigerito::from_bytes(&bytes).expect("must round-trip");
        assert_eq!(bundle2.commitment.cap, commitment.cap);

        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify round-tripped Ligerito R1cs proof");

        // Byte-flipping inside the payload should make verification reject.
        // The flip can either fail deserialization OR succeed-then-fail-at-
        // verify; either is acceptable evidence the proof was consumed.
        let flip_at = HEADER_LEN + (bytes.len() - HEADER_LEN) / 2;
        let mut mutated = bytes.clone();
        mutated[flip_at] ^= 0xFF;
        match R1csProofBundleLigerito::from_bytes(&mutated) {
            Err(_) => {}
            Ok(bundle3) => {
                let mut chv = FsChallenger::new(b"flock-proofio-lig");
                let res = setup.verify(&bundle3.commitment, &bundle3.proof, &mut chv);
                assert!(res.is_err(), "verify must reject byte-mutated proof");
            }
        }

        // File roundtrip.
        let path = std::env::temp_dir().join("flock-proofio-roundtrip.bin");
        write_bytes_to_file(&path, &bytes).expect("write");
        let read_back = read_bytes_from_file(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        let bundle4 = R1csProofBundleLigerito::from_bytes(&read_back).expect("file round-trip");
        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle4.commitment, &bundle4.proof, &mut chv)
            .expect("verify after file round-trip");

        eprintln!(
            "Ligerito R1csProofBundle: {} bytes ({:.1} KB)",
            bytes.len(),
            bytes.len() as f64 / 1024.0
        );
    }

    /// Mixed bundle (current wire version, merged transport) end-to-end:
    /// prove a small partial-count mixed
    /// instance on the nu7 tier, serialize, roundtrip, verify from the
    /// deserialized bundle (registry rebuilt from the id, counts from the
    /// bundle), and reject count tampering.
    #[test]
    #[ignore] // Heavier — run with `cargo test mixed_bundle_roundtrip -- --ignored`
    fn mixed_bundle_roundtrip_and_verify() {
        use crate::mixed::{MixedCounts, MixedRegistryId, MixedSetup};
        use flock_prover_test_inputs::{random_blake3_inputs, random_sha2_inputs};

        let setup = MixedSetup::new(MixedRegistryId::Blake3Sha2Nu7);
        let mut rng = Rng::new(0x0511_31ED);
        let sha2_inputs = random_sha2_inputs(&mut rng, 100);
        let blake3_inputs = random_blake3_inputs(&mut rng, 37);

        let mut ch = FsChallenger::new(b"flock-proofio-mixed");
        let (proof, commitment, _claim) =
            setup.prove(&sha2_inputs, &blake3_inputs, Default::default(), &mut ch);

        let bundle = MixedProofBundleLigerito {
            registry_id: setup.id,
            counts: vec![100, 37],
            commitment,
            proof,
        };
        let bytes = bundle.to_bytes();
        assert_eq!(&bytes[0..5], &MAGIC);
        assert_eq!(bytes[5], VERSION);
        assert_eq!(bytes[6], FLAVOR_MIXED_LIGERITO);
        assert!(matches!(peek_flavor(&bytes), Ok(BundleFlavor::Mixed)));

        let bundle2 = MixedProofBundleLigerito::from_bytes(&bytes).expect("must round-trip");
        assert_eq!(bundle2.registry_id, bundle.registry_id);
        assert_eq!(bundle2.counts, bundle.counts);
        assert_eq!(bundle2.commitment.cap, bundle.commitment.cap);

        let mut bytes_with_suffix = bytes.clone();
        bytes_with_suffix.push(0);
        assert!(
            MixedProofBundleLigerito::from_bytes(&bytes_with_suffix).is_err(),
            "a valid bundle with trailing bytes must be rejected"
        );

        // Verify from the deserialized bundle alone (+ the rebuilt tier).
        let setup2 = MixedSetup::new(bundle2.registry_id);
        let counts = MixedCounts {
            sha2: bundle2.counts[0] as usize,
            blake3: bundle2.counts[1] as usize,
        };
        let mut chv = FsChallenger::new(b"flock-proofio-mixed");
        setup2
            .verify(
                counts,
                bundle2.commitment.params.profile,
                &bundle2.commitment,
                &bundle2.proof,
                &mut chv,
            )
            .expect("verify round-tripped mixed proof");

        // Verification policy is caller-selected. A valid Fast proof must
        // not be accepted when the caller requests a compatibility profile.
        let mut chv = FsChallenger::new(b"flock-proofio-mixed");
        assert!(
            setup2
                .verify(
                    counts,
                    flock_core::pcs::ligerito::LigeritoProfile::Fast100,
                    &bundle2.commitment,
                    &bundle2.proof,
                    &mut chv,
                )
                .is_err(),
            "proof-carried profile must not override verifier policy"
        );

        // Tampered counts must reject (they bind before any challenge).
        let mut chv = FsChallenger::new(b"flock-proofio-mixed");
        assert!(
            setup2
                .verify(
                    MixedCounts {
                        sha2: 101,
                        blake3: 37
                    },
                    bundle2.commitment.params.profile,
                    &bundle2.commitment,
                    &bundle2.proof,
                    &mut chv,
                )
                .is_err(),
            "tampered counts must reject"
        );

        // File roundtrip.
        let path = std::env::temp_dir().join("flock-proofio-mixed-roundtrip.bin");
        write_mixed_bundle_ligerito_to_file(&path, &bundle).expect("write");
        let bundle3 = read_mixed_bundle_ligerito_from_file(&path).expect("file round-trip");
        let _ = std::fs::remove_file(&path);
        assert_eq!(bundle3.counts, bundle.counts);

        eprintln!(
            "Mixed proof bundle ({}, counts sha2=100 blake3=37): {} bytes ({:.1} KB)",
            bundle.registry_id.as_str(),
            bytes.len(),
            bytes.len() as f64 / 1024.0
        );
    }

    /// Deterministic input generators shared with the mixed bundle test.
    mod flock_prover_test_inputs {
        use super::Rng;

        pub fn random_blake3_inputs(
            rng: &mut Rng,
            n: usize,
        ) -> Vec<crate::r1cs_hashes::blake3::Compression> {
            (0..n)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
                    let m: [u32; 16] = std::array::from_fn(|_| rng.nx() as u32);
                    (cv, m, rng.nx(), 64u32, 11u32)
                })
                .collect()
        }

        pub fn random_sha2_inputs(
            rng: &mut Rng,
            n: usize,
        ) -> Vec<crate::r1cs_hashes::sha2::Compression> {
            (0..n)
                .map(|_| {
                    (
                        std::array::from_fn(|_| rng.nx() as u32),
                        std::array::from_fn(|_| rng.nx() as u32),
                    )
                })
                .collect()
        }
    }

    /// Mixed flavor header mechanics (cheap): peek_flavor on both flavors,
    /// mixed-vs-R1cs flavor mismatch, and version strictness for the mixed
    /// reader.
    #[test]
    fn mixed_flavor_header_checks() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION;
        for (flavor, expect) in [
            (FLAVOR_R1CS_LIGERITO, BundleFlavor::R1cs),
            (FLAVOR_MIXED_LIGERITO, BundleFlavor::Mixed),
        ] {
            bytes[6] = flavor;
            assert!(matches!(peek_flavor(&bytes), Ok(f) if f == expect));
        }

        // R1cs-flavored header read as Mixed: flavor mismatch.
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        assert!(matches!(
            MixedProofBundleLigerito::from_bytes(&bytes),
            Err(DeserializeError::FlavorMismatch {
                expected: FLAVOR_MIXED_LIGERITO,
                found: FLAVOR_R1CS_LIGERITO
            })
        ));

        // Old version (v20) rejected — strict versioning.
        bytes[5] = VERSION - 1;
        bytes[6] = FLAVOR_MIXED_LIGERITO;
        assert!(matches!(
            MixedProofBundleLigerito::from_bytes(&bytes),
            Err(DeserializeError::UnsupportedVersion(v)) if v == VERSION - 1
        ));
        assert!(matches!(
            peek_flavor(&bytes),
            Err(DeserializeError::UnsupportedVersion(v)) if v == VERSION - 1
        ));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(b"NOPE!");
        bytes[5] = VERSION;
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        let res = R1csProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::BadMagic)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION.wrapping_add(1);
        bytes[6] = FLAVOR_R1CS_LIGERITO;
        let res = R1csProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::UnsupportedVersion(_))));
    }

    #[test]
    fn rejects_flavor_mismatch() {
        // Mixed-flavored header — try to read as R1cs. Header validation
        // fails before any payload deserialization, so zero payload is fine.
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION;
        bytes[6] = FLAVOR_MIXED_LIGERITO;
        let res = R1csProofBundleLigerito::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(DeserializeError::FlavorMismatch {
                expected: FLAVOR_R1CS_LIGERITO,
                found: FLAVOR_MIXED_LIGERITO
            })
        ));
    }

    #[test]
    fn rejects_legacy_basefold_flavor() {
        // Flavor bytes 0/1 were the legacy BaseFold bundles — now unknown.
        for legacy in [0u8, 1u8] {
            let mut bytes = vec![0u8; HEADER_LEN + 10];
            bytes[0..5].copy_from_slice(&MAGIC);
            bytes[5] = VERSION;
            bytes[6] = legacy;
            let res = R1csProofBundleLigerito::from_bytes(&bytes);
            assert!(matches!(res, Err(DeserializeError::UnknownFlavor(f)) if f == legacy));
        }
    }

    #[test]
    fn rejects_truncated() {
        let res = R1csProofBundleLigerito::from_bytes(&[0u8; 3]);
        assert!(matches!(res, Err(DeserializeError::Truncated)));
    }

    #[test]
    fn bounded_decoder_rejects_trailing_bytes() {
        let mut payload = bincode::serialize(&0x1280_u64).expect("serialize test value");
        payload.push(0);
        assert!(deserialize_payload::<u64>(&payload).is_err());
    }

    #[test]
    fn bounded_decoder_rejects_impossible_vector_length() {
        // Fixed-width bincode represents a Vec length as a little-endian u64.
        // The byte limit rejects this before attempting the requested
        // allocation.
        let payload = u64::MAX.to_le_bytes();
        assert!(deserialize_payload::<Vec<u8>>(&payload).is_err());
    }

    #[test]
    fn rejects_bundle_above_size_ceiling() {
        assert!(matches!(
            check_bundle_size(MAX_BUNDLE_BYTES + 1),
            Err(DeserializeError::TooLarge {
                len,
                max: MAX_BUNDLE_BYTES
            }) if len == MAX_BUNDLE_BYTES + 1
        ));
    }
}
