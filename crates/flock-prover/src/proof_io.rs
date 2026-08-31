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

use crate::mixed::MixedRegistryId;
use bincode::DefaultOptions;
use bincode::Error as BincodeError;
use bincode::serialize_into;
use flock_core::proof::R1csProofMergedLigerito;
use serde::de::DeserializeOwned;
use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::fs::File;
use std::fs::rename;
use std::fs::write;
use std::io::Read;
use std::io::Result as IoResult;
use std::io::{Error as IoError, ErrorKind as IoErrorKind};
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

/// Format version for the current proof schemas and transcript rules.
/// v22 (2026-08-27): the profile consolidation (bloat ledger §C) — the
/// grind-free `fast`/`slim` were deleted and the `fast128`/`slim128`
/// schedules took their names, so a v21 proof under those names carries a
/// different query/rate/PoW schedule and cannot be interpreted safely; the
/// mixed registry codes 3 and 4 (`merkle26+blake3@nu*`) were retired and
/// are never reused. `fast100`/`slim100`/`secure` payloads are unchanged.
/// (The per-version history this file used to carry lives in git.)
const VERSION: u8 = 22;

/// Flavor discriminator (1 byte). Lets a generic reader peek what kind of
/// bundle a file holds without parsing the payload first (see
/// [`peek_flavor`]). Values 0, 1, and 3 are reserved.
const FLAVOR_R1CS_LIGERITO: u8 = 2;
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
    Bincode(BincodeError),
}

impl Display for DeserializeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
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

impl Error for DeserializeError {}

impl From<BincodeError> for DeserializeError {
    fn from(e: BincodeError) -> Self {
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
    pub proof: R1csProofMergedLigerito,
}

impl R1csProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_R1CS_LIGERITO);
        serialize_into(&mut out, self).expect("bincode serialize R1csProofBundleLigerito");
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
/// commitment to the dense stack, and the merged union proof.
/// The statement proves well-formedness. The commitment opens to tables with the
/// declared counts, every declared row satisfying its type's hash relation
/// — no per-invocation I/O binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixedProofBundleLigerito {
    pub registry_id: MixedRegistryId,
    /// Declared invocation counts, in the registry's slot order.
    pub counts: Vec<u64>,
    pub commitment: Commitment,
    pub proof: R1csProofMergedLigerito,
}

impl MixedProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_MIXED_LIGERITO);
        serialize_into(&mut out, self).expect("bincode serialize MixedProofBundleLigerito");
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
) -> IoResult<()> {
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

fn deserialize_payload<T: DeserializeOwned>(payload: &[u8]) -> Result<T, DeserializeError> {
    Ok(DefaultOptions::new()
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
pub fn write_bytes_to_file<P: AsRef<Path>>(path: P, bytes: &[u8]) -> IoResult<()> {
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
    write(&tmp, bytes)?;
    rename(&tmp, path)
}

/// Read proof bytes from a file without ever buffering more than
/// [`MAX_BUNDLE_BYTES`]. The `take` guard also handles a file growing between
/// its metadata check and the read.
pub fn read_bytes_from_file<P: AsRef<Path>>(path: P) -> IoResult<Vec<u8>> {
    let mut file = File::open(path)?;
    let declared_len = file.metadata()?.len();
    if declared_len > MAX_BUNDLE_BYTES as u64 {
        return Err(IoError::new(
            IoErrorKind::InvalidData,
            format!("proof bundle is {declared_len} bytes; maximum is {MAX_BUNDLE_BYTES}"),
        ));
    }
    let mut bytes = Vec::with_capacity(declared_len as usize);
    file.by_ref()
        .take(MAX_BUNDLE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_BUNDLE_BYTES {
        return Err(IoError::new(
            IoErrorKind::InvalidData,
            format!("proof bundle exceeded the {MAX_BUNDLE_BYTES}-byte maximum while reading"),
        ));
    }
    Ok(bytes)
}

/// Combined error returned by file-read helpers: either IO failed or the
/// bytes weren't a valid bundle.
#[derive(Debug)]
pub enum BundleReadError {
    Io(IoError),
    Deserialize(DeserializeError),
}

impl Display for BundleReadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Deserialize(e) => write!(f, "deserialize error: {e}"),
        }
    }
}

impl Error for BundleReadError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::blake3::{Blake3Setup, Compression};
    use bincode::serialize;
    use flock_hash::blake3_compress;
    use flock_transcript::challenger::FsChallenger;
    use std::array::from_fn;
    use std::env::temp_dir;
    use std::fs::remove_file;

    use crate::mixed::{MixedCounts, MixedRegistryId, MixedSetup};
    use flock_core::pcs::ligerito::LigeritoProfile;
    use flock_core::test_rng::Rng;
    use flock_prover_test_inputs::{random_blake3_inputs, random_sha2_inputs};

    /// Build a small honest BLAKE3 chain (n=8) for the bundle tests.
    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = Rng::new(seed);
        let mut cv: [u32; 8] = from_fn(|_| rng.next_u64() as u32);
        let cv0 = cv;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let m: [u32; 16] = from_fn(|_| rng.next_u64() as u32);
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
        let path = temp_dir().join("flock-proofio-roundtrip.bin");
        write_bytes_to_file(&path, &bytes).expect("write");
        let read_back = read_bytes_from_file(&path).expect("read");
        let _ = remove_file(&path);
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
                    LigeritoProfile::Fast100,
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
        let path = temp_dir().join("flock-proofio-mixed-roundtrip.bin");
        write_mixed_bundle_ligerito_to_file(&path, &bundle).expect("write");
        let bundle3 = read_mixed_bundle_ligerito_from_file(&path).expect("file round-trip");
        let _ = remove_file(&path);
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
        use crate::r1cs_hashes::blake3::Compression as Blake3Compression;
        use crate::r1cs_hashes::sha2::Compression as Sha2Compression;
        use std::array::from_fn;

        pub fn random_blake3_inputs(rng: &mut Rng, n: usize) -> Vec<Blake3Compression> {
            (0..n)
                .map(|_| {
                    let cv: [u32; 8] = from_fn(|_| rng.next_u64() as u32);
                    let m: [u32; 16] = from_fn(|_| rng.next_u64() as u32);
                    (cv, m, rng.next_u64(), 64u32, 11u32)
                })
                .collect()
        }

        pub fn random_sha2_inputs(rng: &mut Rng, n: usize) -> Vec<Sha2Compression> {
            (0..n)
                .map(|_| {
                    (
                        from_fn(|_| rng.next_u64() as u32),
                        from_fn(|_| rng.next_u64() as u32),
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
        let mut payload = serialize(&0x1280_u64).expect("serialize test value");
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
