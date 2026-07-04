//! Serialize / deserialize proofs to bytes (and files).
//!
//! Two bundle types: [`R1csProofBundle`] for the base R1CS proof and
//! [`ChainProofBundle`] for the hash-chain proof. Both pair a proof with its
//! commitment (which the verifier needs); the chain bundle additionally
//! carries the public endpoint bits.
//!
//! On-disk format:
//! ```text
//!   bytes 0..5    "FLOCK"                  (5-byte magic)
//!   byte  5       VERSION                  (currently 1)
//!   bytes 6..7    flavor: 0 = R1cs, 1 = Chain
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
//! let bundle = R1csProofBundle { commitment, proof };
//! let bytes = bundle.to_bytes();
//! std::fs::write("proof.bin", &bytes)?;
//! ...
//! let bytes = std::fs::read("proof.bin")?;
//! let bundle = R1csProofBundle::from_bytes(&bytes)?;
//! // Then call e.g. `setup.verify(&bundle.commitment, &bundle.proof, ...)`.
//! ```

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::r1cs_hashes::chain_common::ChainProof;
use flock_core::pcs::Commitment;
use flock_core::proof::R1csProof;

/// Magic bytes prepended to every serialized proof. Lets readers reject
/// random binary data early.
pub const MAGIC: [u8; 5] = *b"FLOCK";

/// Format version. Bumped on incompatible serialization changes.
/// v4 (current) adds `ood_values` + `fold_grinding_nonces` to
/// `LigeritoProof` and `profile` to `PcsParams` (Johnson+OOD profiles).
/// v3 restructures `BaseFoldProof`: per-query Merkle paths are replaced by
/// shared octopus multi-proofs (one per Merkle tree). v2 added `HashKind`
/// to [`ChainProofBundle`].
pub const VERSION: u8 = 4;

/// Which hash function a chain proof is over. Carried in
/// [`ChainProofBundle`] so the verifier (e.g. the CLI) can pick the right
/// `*_chain` setup without out-of-band info.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashKind {
    Blake3,
    Sha2,
    Keccak,
}

impl HashKind {
    /// Parse a CLI-style name; case-insensitive. Accepts `blake3`, `sha2` /
    /// `sha256`, `keccak` / `keccak_f`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "blake3" => Some(Self::Blake3),
            "sha2" | "sha256" | "sha-2" | "sha-256" => Some(Self::Sha2),
            "keccak" | "keccak_f" | "keccak-f" => Some(Self::Keccak),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Blake3 => "blake3",
            Self::Sha2 => "sha2",
            Self::Keccak => "keccak",
        }
    }
}

/// Flavor discriminator (1 byte). Lets a generic reader peek what kind of
/// bundle a file holds without parsing the payload first.
const FLAVOR_R1CS: u8 = 0;
const FLAVOR_CHAIN: u8 = 1;
const FLAVOR_R1CS_LIGERITO: u8 = 2;
const FLAVOR_CHAIN_LIGERITO: u8 = 3;

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
    /// The flavor byte was neither `0` (R1cs) nor `1` (Chain).
    UnknownFlavor(u8),
    /// `from_bytes` was called with a slice shorter than `HEADER_LEN`.
    Truncated,
    /// The expected flavor and the file's flavor disagree (e.g. trying to
    /// load a `ChainProofBundle` from an R1CS bundle file).
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
pub struct R1csProofBundle {
    pub commitment: Commitment,
    pub proof: R1csProof,
}

/// Bundles a hash-chain proof with its commitment + public endpoint bits
/// (`cv_0_phys` and `cv_last_phys` are the physical within-slot bool layouts
/// returned by per-hash `*_to_phys_bits` helpers — `region_bits` long each)
/// plus the [`HashKind`] discriminator so a verifier can pick the right
/// per-hash setup from the bundle alone.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProofBundle {
    pub hash_kind: HashKind,
    pub commitment: Commitment,
    pub proof: ChainProof,
    pub cv_0_phys: Vec<bool>,
    pub cv_last_phys: Vec<bool>,
}

/// Ligerito-backend mirror of [`R1csProofBundle`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofBundleLigerito {
    pub commitment: Commitment,
    pub proof: flock_core::proof::R1csProofLigerito,
}

/// Ligerito-backend mirror of [`ChainProofBundle`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProofBundleLigerito {
    pub hash_kind: HashKind,
    pub commitment: Commitment,
    pub proof: crate::r1cs_hashes::chain_common::ChainProofLigerito,
    pub cv_0_phys: Vec<bool>,
    pub cv_last_phys: Vec<bool>,
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
        Ok(bincode::deserialize(payload)?)
    }
}

impl ChainProofBundleLigerito {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_CHAIN_LIGERITO);
        bincode::serialize_into(&mut out, self)
            .expect("bincode serialize ChainProofBundleLigerito");
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_CHAIN_LIGERITO)?;
        Ok(bincode::deserialize(payload)?)
    }
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
    if flavor != FLAVOR_R1CS
        && flavor != FLAVOR_CHAIN
        && flavor != FLAVOR_R1CS_LIGERITO
        && flavor != FLAVOR_CHAIN_LIGERITO
    {
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

impl R1csProofBundle {
    /// Serialize to a self-contained byte vector. Format: 7-byte header
    /// (magic + version + flavor=0) followed by bincode payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_R1CS);
        // bincode default config: variable-length integer encoding for Vec
        // lengths etc., little-endian. Deterministic across runs.
        bincode::serialize_into(&mut out, self).expect("bincode serialize R1csProofBundle");
        out
    }

    /// Inverse of [`Self::to_bytes`]. Validates magic/version/flavor before
    /// deserializing.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_R1CS)?;
        Ok(bincode::deserialize(payload)?)
    }
}

impl ChainProofBundle {
    /// Serialize to a self-contained byte vector. Format: 7-byte header
    /// (magic + version + flavor=1) followed by bincode payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 1024);
        write_header(&mut out, FLAVOR_CHAIN);
        bincode::serialize_into(&mut out, self).expect("bincode serialize ChainProofBundle");
        out
    }

    /// Inverse of [`Self::to_bytes`]. Validates magic/version/flavor before
    /// deserializing.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeserializeError> {
        let payload = parse_header(bytes, FLAVOR_CHAIN)?;
        Ok(bincode::deserialize(payload)?)
    }
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

/// Read raw bytes from a file. Thin wrapper over `std::fs::read`.
pub fn read_bytes_from_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Convenience: write the bundle's bytes to `path` in one call.
pub fn write_r1cs_bundle_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &R1csProofBundle,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

/// Convenience: read a `R1csProofBundle` from `path`.
pub fn read_r1cs_bundle_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<R1csProofBundle, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    R1csProofBundle::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

/// Convenience: write the chain bundle's bytes to `path` in one call.
pub fn write_chain_bundle_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &ChainProofBundle,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

/// Convenience: read a `ChainProofBundle` from `path`.
pub fn read_chain_bundle_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<ChainProofBundle, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    ChainProofBundle::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

/// Write a Ligerito chain bundle to `path`.
pub fn write_chain_bundle_ligerito_to_file<P: AsRef<Path>>(
    path: P,
    bundle: &ChainProofBundleLigerito,
) -> io::Result<()> {
    write_bytes_to_file(path, &bundle.to_bytes())
}

/// Read a Ligerito chain bundle from `path`.
pub fn read_chain_bundle_ligerito_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<ChainProofBundleLigerito, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    ChainProofBundleLigerito::from_bytes(&bytes).map_err(BundleReadError::Deserialize)
}

/// Backend-agnostic chain bundle wrapper. Dispatches on the on-disk `FLAVOR`
/// byte at read time, so callers (e.g. the CLI's verify path) don't need to
/// know the backend up front.
#[derive(Clone, Debug)]
pub enum AnyChainBundle {
    BaseFold(ChainProofBundle),
    Ligerito(ChainProofBundleLigerito),
}

/// Read a chain bundle of either backend flavor from `path`. Peeks at the
/// flavor byte to decide; returns an [`AnyChainBundle`] for the caller to
/// dispatch on.
pub fn read_any_chain_bundle_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<AnyChainBundle, BundleReadError> {
    let bytes = read_bytes_from_file(path).map_err(BundleReadError::Io)?;
    if bytes.len() < HEADER_LEN {
        return Err(BundleReadError::Deserialize(DeserializeError::Truncated));
    }
    match bytes[6] {
        FLAVOR_CHAIN => ChainProofBundle::from_bytes(&bytes)
            .map(AnyChainBundle::BaseFold)
            .map_err(BundleReadError::Deserialize),
        FLAVOR_CHAIN_LIGERITO => ChainProofBundleLigerito::from_bytes(&bytes)
            .map(AnyChainBundle::Ligerito)
            .map_err(BundleReadError::Deserialize),
        other => Err(BundleReadError::Deserialize(
            DeserializeError::UnknownFlavor(other),
        )),
    }
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
    use crate::r1cs_hashes::blake3::{Blake3Setup, Compression, blake3_compress, cv_to_phys_bits};
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

    /// Legacy BaseFold bundle roundtrip at small K=8 (m=17).
    #[test]
    fn r1cs_bundle_basefold_roundtrip() {
        let setup = Blake3Setup::new(8);
        let (blocks, _, _) = honest_chain(8, 0xDEAD_BEEF);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment, _claim) = setup.prove_fast_basefold(&blocks, &mut ch);

        let bundle = R1csProofBundle {
            commitment: commitment.clone(),
            proof: proof.clone(),
        };
        let bytes = bundle.to_bytes();

        assert_eq!(&bytes[0..5], &MAGIC);
        assert_eq!(bytes[5], VERSION);
        assert_eq!(bytes[6], FLAVOR_R1CS);

        let bundle2 = R1csProofBundle::from_bytes(&bytes).expect("must round-trip");
        assert_eq!(bundle2.commitment.root, commitment.root);

        let mut chv = FsChallenger::new(b"flock-proofio-test");
        setup
            .verify_basefold(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify round-tripped R1cs proof");
    }

    /// Default Ligerito bundle roundtrip. Requires m ≥ 21 — use n_blocks=128
    /// (m=22 with K_LOG=14).
    #[test]
    #[ignore]
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
        assert_eq!(bundle2.commitment.root, commitment.root);

        let mut chv = FsChallenger::new(b"flock-proofio-lig");
        setup
            .verify(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify round-tripped Ligerito R1cs proof");

        eprintln!(
            "Ligerito R1csProofBundle: {} bytes ({:.1} KB)",
            bytes.len(),
            bytes.len() as f64 / 1024.0
        );
    }

    /// Legacy BaseFold chain bundle roundtrip at small K=8.
    #[test]
    fn chain_bundle_basefold_roundtrip_and_verify() {
        let setup = Blake3Setup::new(8);
        let (blocks, cv_0, cv_last) = honest_chain(8, 0xC0FFEE);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment) = setup.prove_chain_basefold(&blocks, &mut ch);

        let bundle = ChainProofBundle {
            hash_kind: HashKind::Blake3,
            commitment: commitment.clone(),
            proof: proof.clone(),
            cv_0_phys: cv_to_phys_bits(&cv_0),
            cv_last_phys: cv_to_phys_bits(&cv_last),
        };
        let bytes = bundle.to_bytes();
        assert_eq!(bytes[6], FLAVOR_CHAIN);

        let bundle2 = ChainProofBundle::from_bytes(&bytes).expect("chain round-trip");
        assert_eq!(bundle2.cv_0_phys, bundle.cv_0_phys);
        assert_eq!(bundle2.cv_last_phys, bundle.cv_last_phys);

        let mut chv = FsChallenger::new(b"flock-proofio-test");
        setup
            .verify_chain_basefold(
                &bundle2.commitment,
                &bundle2.proof,
                &cv_0,
                &cv_last,
                &mut chv,
            )
            .expect("verify round-tripped chain proof");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(b"NOPE!");
        bytes[5] = VERSION;
        bytes[6] = FLAVOR_R1CS;
        let res = R1csProofBundle::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::BadMagic)));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = vec![0u8; HEADER_LEN + 10];
        bytes[0..5].copy_from_slice(&MAGIC);
        bytes[5] = VERSION.wrapping_add(1);
        bytes[6] = FLAVOR_R1CS;
        let res = R1csProofBundle::from_bytes(&bytes);
        assert!(matches!(res, Err(DeserializeError::UnsupportedVersion(_))));
    }

    #[test]
    fn rejects_flavor_mismatch() {
        // R1CS bundle bytes — try to read as Chain. Uses basefold path since
        // R1csProofBundle is the BaseFold flavor.
        let setup = Blake3Setup::new(8);
        let (blocks, _, _) = honest_chain(8, 0xF00D);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment, _) = setup.prove_fast_basefold(&blocks, &mut ch);
        let bytes = (R1csProofBundle { commitment, proof }).to_bytes();
        let res = ChainProofBundle::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(DeserializeError::FlavorMismatch {
                expected: FLAVOR_CHAIN,
                found: FLAVOR_R1CS
            })
        ));
    }

    #[test]
    fn rejects_truncated() {
        let res = R1csProofBundle::from_bytes(&[0u8; 3]);
        assert!(matches!(res, Err(DeserializeError::Truncated)));
    }

    /// Byte-flipping inside the payload should make verification reject. The
    /// flip can either fail deserialization OR succeed-then-fail-at-verify;
    /// either is acceptable evidence that the proof was actually consumed.
    #[test]
    fn payload_byte_flip_caught() {
        let setup = Blake3Setup::new(8);
        let (blocks, _, _) = honest_chain(8, 0x1234);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment, _) = setup.prove_fast_basefold(&blocks, &mut ch);
        let bundle = R1csProofBundle { commitment, proof };
        let bytes = bundle.to_bytes();

        let flip_at = HEADER_LEN + (bytes.len() - HEADER_LEN) / 2;
        let mut mutated = bytes.clone();
        mutated[flip_at] ^= 0xFF;

        let parsed: Result<R1csProofBundle, _> = R1csProofBundle::from_bytes(&mutated);
        match parsed {
            Err(_) => {}
            Ok(bundle2) => {
                let mut chv = FsChallenger::new(b"flock-proofio-test");
                let res = setup.verify_basefold(&bundle2.commitment, &bundle2.proof, &mut chv);
                assert!(res.is_err(), "verify must reject byte-mutated proof");
            }
        }
    }

    #[test]
    fn file_roundtrip() {
        let setup = Blake3Setup::new(8);
        let (blocks, _, _) = honest_chain(8, 0xAA);
        let mut ch = FsChallenger::new(b"flock-proofio-test");
        let (proof, commitment, _) = setup.prove_fast_basefold(&blocks, &mut ch);
        let bundle = R1csProofBundle { commitment, proof };

        let path = std::env::temp_dir().join("flock-proofio-roundtrip.bin");
        write_r1cs_bundle_to_file(&path, &bundle).expect("write");
        let bundle2 = read_r1cs_bundle_from_file(&path).expect("read");
        let _ = std::fs::remove_file(&path);

        let mut chv = FsChallenger::new(b"flock-proofio-test");
        setup
            .verify_basefold(&bundle2.commitment, &bundle2.proof, &mut chv)
            .expect("verify after file round-trip");
    }
}
