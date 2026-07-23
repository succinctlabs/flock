//! `flock_chain` — CLI for proving and verifying hash-chain proofs, and
//! (since wire v5) multi-table MIXED proofs.
//!
//! ```text
//! Usage:
//!   flock_chain prove   --hash <blake3|sha2|keccak>
//!                       [--steps N]                     (default 8; must be a power-of-2 ≥ 8)
//!                       [--seed HEX]                    (16 hex chars; default 0)
//!                       [--initial-cv HEX]              (64 hex for blake3/sha2, 400 hex for keccak;
//!                                                        default: hash's IV / all-zero state)
//!                       --out FILE
//!   flock_chain prove   --mix blake3=N,sha2=M [--seed HEX] --out FILE
//!   flock_chain verify  --in FILE
//!   flock_chain help
//! ```
//!
//! Build the prover: `cargo build --release --bin flock_chain`.
//! Run via `cargo run --release --bin flock_chain -- <subcommand> [args]`.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::mixed::{MixedCounts, MixedRegistryId, MixedSetup};
use flock_prover::pcs::Commitment;
use flock_prover::proof_io::{
    BundleFlavor, ChainProofBundleLigerito, HashKind, MixedProofBundleLigerito, peek_flavor,
    read_bytes_from_file, write_chain_bundle_ligerito_to_file, write_mixed_bundle_ligerito_to_file,
};
use flock_prover::r1cs_hashes::blake3::{
    self as blake3_chain, BLAKE3_IV, Blake3Setup, blake3_compress, cv_to_phys_bits as bl_cv_phys,
};
use flock_prover::r1cs_hashes::chain_common;
use flock_prover::r1cs_hashes::keccak::{
    self as keccak_chain, KeccakSetup, STATE_BITS, State, keccak_f, state_to_phys_bits,
};
use flock_prover::r1cs_hashes::sha2::{
    self as sha2_chain, SHA256_IV, Sha256HybridSetup, cv_to_phys_bits as sh_cv_phys,
    sha256_compress,
};

// ---------------------------------------------------------------------------
// Argument parsing (tiny, no clap dep)
// ---------------------------------------------------------------------------

/// Prover profile — selects the Ligerito security config. `Fast` = rate 1/2,
/// Johnson+OOD, 100-bit (default). `Slim` = rate 1/4, Johnson+OOD + query
/// grinding, 100-bit (smaller proof, slower prover). `Secure` = rate 1/2,
/// unique-decoding regime, 120-bit (largest proof, most conservative).
type Mode = flock_prover::pcs::ligerito::LigeritoProfile;

#[derive(Default)]
struct Args {
    hash: Option<HashKind>,
    steps: Option<usize>,
    seed: Option<u64>,
    initial_cv_hex: Option<String>,
    out: Option<String>,
    input: Option<String>,
    mode: Option<Mode>,
    mix: Option<MixSpec>,
}

/// Parsed `--mix` argument: per-type invocation counts.
#[derive(Clone, Copy, Debug, Default)]
struct MixSpec {
    blake3: usize,
    sha2: usize,
}

impl MixSpec {
    /// Parse `blake3=N,sha2=M` (either key may be omitted — count 0; at
    /// least one must be positive; keys accept the `HashKind` spellings).
    fn parse(s: &str) -> Result<Self, String> {
        let mut spec = MixSpec::default();
        for part in s.split(',') {
            let (key, val) = part
                .split_once('=')
                .ok_or_else(|| format!("--mix: expected key=COUNT, got '{part}'"))?;
            let count: usize = val
                .parse()
                .map_err(|e| format!("--mix: invalid count '{val}' for '{key}': {e}"))?;
            match HashKind::parse(key) {
                Some(HashKind::Blake3) => spec.blake3 = count,
                Some(HashKind::Sha2) => spec.sha2 = count,
                _ => {
                    return Err(format!(
                        "--mix: unknown type '{key}' (mixed registries cover blake3, sha2)"
                    ));
                }
            }
        }
        if spec.blake3 == 0 && spec.sha2 == 0 {
            return Err("--mix: at least one count must be positive".into());
        }
        Ok(spec)
    }
}

fn parse_args(it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = it.peekable();
    while let Some(flag) = it.next() {
        macro_rules! val {
            () => {
                it.next()
                    .ok_or_else(|| format!("flag {flag} requires a value"))?
            };
        }
        match flag.as_str() {
            "--hash" => {
                let v: String = val!();
                args.hash = Some(HashKind::parse(&v).ok_or_else(|| {
                    format!("--hash: unknown kind '{v}' (expected blake3|sha2|keccak)")
                })?);
            }
            "--steps" => {
                let v: String = val!();
                args.steps = Some(
                    v.parse::<usize>()
                        .map_err(|e| format!("--steps: invalid integer '{v}': {e}"))?,
                );
            }
            "--seed" => {
                let v: String = val!();
                args.seed = Some(
                    u64::from_str_radix(v.trim_start_matches("0x"), 16)
                        .map_err(|e| format!("--seed: invalid hex u64 '{v}': {e}"))?,
                );
            }
            "--initial-cv" => args.initial_cv_hex = Some(val!()),
            "--mix" => {
                let v: String = val!();
                args.mix = Some(MixSpec::parse(&v)?);
            }
            "--out" => args.out = Some(val!()),
            "--in" => args.input = Some(val!()),
            "--mode" => {
                let v: String = val!();
                args.mode = Some(Mode::parse(&v).ok_or_else(|| {
                    format!("--mode: unknown profile '{v}' (expected fast|slim|secure)")
                })?);
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag '{other}'")),
        }
    }
    Ok(args)
}

const USAGE: &str = "\
flock_chain — prove/verify hash-chain proofs and mixed multi-table proofs

Usage:
  flock_chain prove  --hash <blake3|sha2|keccak> [--steps N] [--seed HEX]
                     [--initial-cv HEX] [--mode <fast|slim|secure>] --out FILE
  flock_chain prove  --mix blake3=N,sha2=M [--seed HEX]
                     [--mode <fast|slim|secure>] --out FILE
  flock_chain verify --in FILE
  flock_chain help

Notes:
  --steps N: must be a power of 2 and ≥ 8 (chain protocol requirement). Default 8.
             The Ligerito PCS needs m ≥ ~21, i.e. steps ≥ 256 (blake3),
             ≥ 128 (sha2), or ≥ 64 (keccak).
  --mix blake3=N,sha2=M: ONE mixed proof of N independent BLAKE3 and M
              independent SHA-256 compressions (any counts, chosen at prove
              time; either may be omitted for 0). Statement is
              well-formedness only — no per-invocation I/O binding. Picks
              the smallest built-in registry tier whose capacity fits.
  --seed HEX: 16 hex chars (u64). Drives message generation for blake3/sha2
              (chain and mix). Default 0. Ignored for keccak (no message).
  --initial-cv HEX: hash-specific length:
              blake3, sha2: 64 hex chars = 8 × 32-bit words, big-endian per word
              keccak:       400 hex chars = 1600 bits, LSB-first per byte
              Defaults: BLAKE3_IV, SHA256_IV, or all-zero state for keccak.
  --mode <fast|slim|secure>: prover profile. Default fast.
              fast = rate 1/2 (smaller log_inv_rate, faster prover, larger proof).
              slim = rate 1/4 (larger log_inv_rate, smaller proof, slower prover).
  --out FILE: write proof bundle here.
  --in FILE:  read proof bundle here (flavor auto-detected: chain or mixed).
";

// ---------------------------------------------------------------------------
// Hex helpers
// ---------------------------------------------------------------------------

fn parse_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string has odd length ({})", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|e| format!("invalid hex: {e}"))
}

fn parse_u32_be_words(hex: &str, expected_words: usize) -> Result<Vec<u32>, String> {
    let bytes = parse_hex(hex)?;
    let expected_bytes = expected_words * 4;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "expected {expected_bytes} hex bytes ({} words × 4); got {}",
            expected_words,
            bytes.len()
        ));
    }
    Ok((0..expected_words)
        .map(|w| {
            let b = &bytes[w * 4..w * 4 + 4];
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
        .collect())
}

fn u32_words_to_hex_be(words: &[u32; 8]) -> String {
    let mut out = String::with_capacity(64);
    for w in words {
        out += &format!("{w:08x}");
    }
    out
}

// SplitMix64 — deterministic message generation.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_block(&mut self) -> [u32; 16] {
        std::array::from_fn(|_| self.nx() as u32)
    }
}

// ---------------------------------------------------------------------------
// Prove
// ---------------------------------------------------------------------------

fn cmd_prove(args: Args) -> Result<(), String> {
    if let Some(mix) = args.mix {
        if args.hash.is_some() || args.steps.is_some() || args.initial_cv_hex.is_some() {
            return Err("prove: --mix is exclusive with --hash/--steps/--initial-cv".into());
        }
        return cmd_prove_mix(mix, args);
    }
    let hash = args.hash.ok_or("prove: --hash is required")?;
    let steps = args.steps.unwrap_or(8);
    let seed = args.seed.unwrap_or(0);
    let mode = args.mode.unwrap_or_default();
    let out = args.out.ok_or("prove: --out is required")?;

    if steps < 8 || !steps.is_power_of_two() {
        return Err(format!(
            "--steps must be a power of 2 and ≥ 8; got {steps} \
             (chain shift requires n_compressions == n_block_slots)"
        ));
    }

    eprintln!(
        "flock_chain prove: hash={} steps={} seed=0x{:016x} mode={}",
        hash.as_str(),
        steps,
        seed,
        mode.as_str(),
    );

    let t_total = Instant::now();
    let bundle = match hash {
        HashKind::Blake3 => prove_blake3(steps, seed, args.initial_cv_hex.as_deref(), mode)?,
        HashKind::Sha2 => prove_sha2(steps, seed, args.initial_cv_hex.as_deref(), mode)?,
        HashKind::Keccak => prove_keccak(steps, args.initial_cv_hex.as_deref(), mode)?,
    };
    eprintln!(
        "  total prove (incl. honest-chain build): {:.2}s",
        t_total.elapsed().as_secs_f64()
    );

    let bytes_len = bundle.to_bytes().len();
    write_chain_bundle_ligerito_to_file(&out, &bundle).map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("  wrote {out} ({bytes_len} bytes)");
    Ok(())
}

/// `prove --mix blake3=N,sha2=M`: ONE mixed multi-table proof of N BLAKE3 +
/// M SHA-256 independent compressions (inputs generated from `--seed`),
/// through the union prove entry on the smallest fitting registry tier.
/// Statement = well-formedness only (no per-invocation I/O binding).
fn cmd_prove_mix(mix: MixSpec, args: Args) -> Result<(), String> {
    let seed = args.seed.unwrap_or(0);
    let mode = args.mode.unwrap_or_default();
    let out = args.out.ok_or("prove: --out is required")?;

    let max_count = mix.blake3.max(mix.sha2);
    let id = MixedRegistryId::smallest_fitting(max_count).ok_or_else(|| {
        format!(
            "--mix: count {max_count} exceeds every built-in registry tier \
             (largest capacity: {}); split the workload",
            1usize << MixedRegistryId::ALL.last().unwrap().nu()
        )
    })?;
    eprintln!(
        "flock_chain prove: mix blake3={} sha2={} seed=0x{seed:016x} mode={} tier={}",
        mix.blake3,
        mix.sha2,
        mode.as_str(),
        id.as_str(),
    );

    // Deterministic independent inputs from the seed (no chaining — the
    // invocations are independent by design).
    let mut rng = Rng::new(seed);
    let blake3_inputs: Vec<blake3_chain::Compression> = (0..mix.blake3)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.nx() as u32);
            let m = rng.next_block();
            (cv, m, 0u64, 64u32, 0u32)
        })
        .collect();
    let sha2_inputs: Vec<sha2_chain::Compression> = (0..mix.sha2)
        .map(|_| (std::array::from_fn(|_| rng.nx() as u32), rng.next_block()))
        .collect();

    let t = Instant::now();
    let setup = MixedSetup::new(id);
    eprintln!("  registry setup: {:.2}s", t.elapsed().as_secs_f64());

    let mut ch = FsChallenger::new(b"flock_chain-cli-mixed");
    let t = Instant::now();
    let (proof, commitment, _claim) = setup.prove(&sha2_inputs, &blake3_inputs, mode, &mut ch);
    eprintln!("  prove_mixed: {:.2}s", t.elapsed().as_secs_f64());
    // Height-n_t stacking: the committed dense stack is count-proportional
    // (floored at the m22 Ligerito config); capacity-height stacking would
    // have committed the full 2^(M-7) padded words.
    eprintln!(
        "  committed: 2^{} words (m = {}; capacity-height would be 2^{})",
        commitment.params.m - 7,
        commitment.params.m,
        id.nu() + 16 - 7,
    );

    let bundle = MixedProofBundleLigerito {
        registry_id: id,
        counts: vec![mix.sha2 as u64, mix.blake3 as u64],
        commitment,
        proof,
    };
    let bytes_len = bundle.to_bytes().len();
    write_mixed_bundle_ligerito_to_file(&out, &bundle).map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("  wrote {out} ({bytes_len} bytes)");
    println!(
        "Proved (well-formedness only): {} blake3 and {} sha2 independent \
         compressions in one proof (tier {}, no per-invocation I/O binding).",
        mix.blake3,
        mix.sha2,
        id.as_str()
    );
    Ok(())
}

fn prove_blake3(
    steps: usize,
    seed: u64,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    let initial_cv: [u32; 8] = if let Some(h) = initial_hex {
        let v = parse_u32_be_words(h, 8)?;
        std::array::from_fn(|i| v[i])
    } else {
        BLAKE3_IV
    };
    eprintln!("  initial cv: {}", u32_words_to_hex_be(&initial_cv));

    let mut rng = Rng::new(seed);
    let mut cv = initial_cv;
    let mut blocks = Vec::with_capacity(steps);
    for _ in 0..steps {
        let m = rng.next_block();
        blocks.push((cv, m, 0u64, 64u32, 0u32));
        let st = blake3_compress(&cv, &m, 0, 64, 0);
        cv = st[0..8].try_into().unwrap();
    }
    let cv_last = cv;
    eprintln!("  cv_last:    {}", u32_words_to_hex_be(&cv_last));

    let setup = Blake3Setup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Blake3,
        commitment,
        proof,
        cv_0_phys: bl_cv_phys(&initial_cv),
        cv_last_phys: bl_cv_phys(&cv_last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

fn prove_sha2(
    steps: usize,
    seed: u64,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    let initial_cv: [u32; 8] = if let Some(h) = initial_hex {
        let v = parse_u32_be_words(h, 8)?;
        std::array::from_fn(|i| v[i])
    } else {
        SHA256_IV
    };
    eprintln!("  initial cv: {}", u32_words_to_hex_be(&initial_cv));

    let mut rng = Rng::new(seed);
    let mut cv = initial_cv;
    let mut blocks = Vec::with_capacity(steps);
    for _ in 0..steps {
        let m = rng.next_block();
        blocks.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }
    let cv_last = cv;
    eprintln!("  cv_last:    {}", u32_words_to_hex_be(&cv_last));

    let setup = Sha256HybridSetup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Sha2,
        commitment,
        proof,
        cv_0_phys: sh_cv_phys(&initial_cv),
        cv_last_phys: sh_cv_phys(&cv_last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

fn prove_keccak(
    steps: usize,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    // Keccak state = 1600 bits. Default: all-zero. User may pass 400 hex chars
    // (200 bytes), LSB-first per byte.
    let initial_state: State = if let Some(h) = initial_hex {
        let bytes = parse_hex(h)?;
        if bytes.len() != STATE_BITS / 8 {
            return Err(format!(
                "--initial-cv for keccak: expected {} bytes ({STATE_BITS} bits); got {}",
                STATE_BITS / 8,
                bytes.len()
            ));
        }
        let mut s = [false; STATE_BITS];
        for (i, b) in bytes.iter().enumerate() {
            for bit in 0..8 {
                s[i * 8 + bit] = (b >> bit) & 1 == 1;
            }
        }
        s
    } else {
        [false; STATE_BITS]
    };

    let mut cur = initial_state;
    let mut inputs = Vec::with_capacity(steps);
    for _ in 0..steps {
        inputs.push(cur);
        keccak_f(&mut cur);
    }
    let last = cur;

    let setup = KeccakSetup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&inputs, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Keccak,
        commitment,
        proof,
        cv_0_phys: state_to_phys_bits(&initial_state),
        cv_last_phys: state_to_phys_bits(&last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: Args) -> Result<(), String> {
    let input = args.input.ok_or("verify: --in is required")?;

    // Flavor auto-detect: chain and mixed bundles share the header.
    let bytes = read_bytes_from_file(&input).map_err(|e| format!("read {input}: {e}"))?;
    match peek_flavor(&bytes).map_err(|e| format!("deserialize {input}: {e}"))? {
        BundleFlavor::Chain => {}
        BundleFlavor::Mixed => return cmd_verify_mix(&input, &bytes),
        BundleFlavor::R1cs => {
            return Err(format!(
                "{input}: bare R1cs bundles carry no statement for this CLI \
                 (expected a chain or mixed bundle)"
            ));
        }
    }

    let bundle = ChainProofBundleLigerito::from_bytes(&bytes)
        .map_err(|e| format!("deserialize {input}: {e}"))?;

    let m = bundle.commitment.params.m;
    let hash = bundle.hash_kind;
    let n_log = match hash {
        HashKind::Blake3 => m - blake3_chain::K_LOG,
        HashKind::Sha2 => m - sha2_chain::K_LOG,
        HashKind::Keccak => m - keccak_chain::K_LOG,
    };
    let steps = 1usize << n_log;

    eprintln!(
        "flock_chain verify: hash={} m={m} steps={steps} (n_log={n_log})",
        hash.as_str()
    );

    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    // The profile is recovered from the committed PcsParams in the proof
    // bundle, not assumed — so `verify` works regardless of which `--mode`
    // produced the proof. Reconstruct the setup with that profile so its
    // r1cs/pcs_params match the prover's.
    let result = match hash {
        HashKind::Blake3 => {
            let setup = Blake3Setup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &blake3_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
        HashKind::Sha2 => {
            let setup = Sha256HybridSetup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &sha2_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
        HashKind::Keccak => {
            let setup = KeccakSetup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &keccak_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
    };
    eprintln!("  verify_chain: {:.2}s", t.elapsed().as_secs_f64());

    match result {
        Ok(()) => {
            println!(
                "OK: {} chain of {steps} compressions verified.",
                hash.as_str()
            );
            Ok(())
        }
        Err(e) => Err(format!("verification rejected: {e:?}")),
    }
}

/// Verify a mixed bundle: rebuild the registry from the bundle's id (which
/// pins the full registry, capacity included), take the counts from the
/// bundle, and run the union verifier under the `flock-mixed-v1` binding.
fn cmd_verify_mix(input: &str, bytes: &[u8]) -> Result<(), String> {
    let bundle = MixedProofBundleLigerito::from_bytes(bytes)
        .map_err(|e| format!("deserialize {input}: {e}"))?;
    let id = bundle.registry_id;
    if bundle.counts.len() != 2 {
        return Err(format!(
            "{input}: tier {} declares 2 types, bundle carries {} counts",
            id.as_str(),
            bundle.counts.len()
        ));
    }
    let capacity = 1u64 << id.nu();
    for (t, &n) in bundle.counts.iter().enumerate() {
        if n > capacity {
            return Err(format!(
                "{input}: count n_{t} = {n} exceeds tier capacity {capacity}"
            ));
        }
    }
    // Counts are in slot order: SHA-256 (wider) first, then BLAKE3.
    let counts = MixedCounts {
        sha2: bundle.counts[0] as usize,
        blake3: bundle.counts[1] as usize,
    };
    eprintln!(
        "flock_chain verify: mixed tier={} sha2={} blake3={} (M={})",
        id.as_str(),
        counts.sha2,
        counts.blake3,
        id.nu() + 16,
    );

    let t = Instant::now();
    let setup = MixedSetup::new(id);
    eprintln!("  registry setup: {:.2}s", t.elapsed().as_secs_f64());
    let mut ch = FsChallenger::new(b"flock_chain-cli-mixed");
    let t = Instant::now();
    let result = setup.verify(counts, &bundle.commitment, &bundle.proof, &mut ch);
    eprintln!("  verify_mixed: {:.2}s", t.elapsed().as_secs_f64());

    match result {
        Ok(_claim) => {
            println!(
                "OK: mixed proof verified — {} sha2 and {} blake3 independent \
                 compressions are well-formed under tier {} (statement is \
                 well-formedness only; no per-invocation I/O binding).",
                counts.sha2,
                counts.blake3,
                id.as_str()
            );
            Ok(())
        }
        Err(e) => Err(format!("verification rejected: {e:?}")),
    }
}

fn verify_ligerito_with_layout(
    r1cs: &flock_prover::r1cs::BlockR1cs,
    layout: &chain_common::ChainLayout,
    commitment: &Commitment,
    bundle: &ChainProofBundleLigerito,
    n_log: usize,
    pcs_params: &flock_prover::pcs::PcsParams,
    challenger: &mut FsChallenger,
) -> Result<(), chain_common::ChainVerifyError> {
    let lc_circuit = r1cs.csc_lincheck_circuit();
    chain_common::verify_chain_ligerito_generic(
        r1cs,
        layout,
        commitment,
        &bundle.proof,
        n_log,
        &bundle.cv_0_phys,
        &bundle.cv_last_phys,
        lc_circuit,
        pcs_params,
        challenger,
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let mut argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(1);
    }
    let subcmd = argv.remove(0);
    let result = match subcmd.as_str() {
        "prove" => parse_args(argv.into_iter()).and_then(cmd_prove),
        "verify" => parse_args(argv.into_iter()).and_then(cmd_verify),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown subcommand '{other}'\n\n{USAGE}")),
    };

    // Silence unused-import lint for the type-only re-export.
    let _ = F128::ZERO;

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::from(1)
        }
    }
}
