// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from bolt-rs (https://github.com/bcc-research/bolt-rs,
// `ligerito_recursive.rs`).

//! Ligerito: recursive multilinear PCS.
//!
//! Ported from bolt-rs (`ligerito_recursive.rs`) onto Flock primitives:
//! `F128` (GHASH irreducible), [`AdditiveNttF128`] (LCH novel basis,
//! byte-identical to bolt-rs's FFT), SHA-256 merkle from [`crate::merkle`],
//! and the [`Challenger`] trait for Fiat-Shamir.
//!
//! Soundness regimes (our paper App. C.3): unique decoding (Thm `ca-udr`,
//! BCHKS25 Cor. 1.4, `Secure` profile) and Johnson list decoding with
//! out-of-domain binding (Thm `ca-johnson`, BCHKS25 Thm 4.6 + Johnson
//! interleaved list bound, `Fast`/`Slim` profiles). See [`SoundnessRegime`].
//!
//! ## Protocol
//! 1. Commit f^0: reshape into `num_interleaved × msg_cols`, RS-encode each
//!    lane to `block_len = msg_cols · 2^log_inv_rate`, merkle over codeword
//!    positions (one position across all lanes = one leaf).
//! 2. Partial-eval f^0 with `initial_k` challenges → f^1.
//! 3. Commit f^1.
//! 4. Open `num_queries` rows of f^0; build induced sumcheck basis poly.
//! 5. For each recursive step i:
//!    a. Run k_i sumcheck rounds.
//!    b. Last step: send remaining poly + open f^i.
//!    c. Else: commit f^{i+2}, open f^{i+1}, induce next basis, glue.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use serde::{Deserialize, Serialize};

// ===================================================================
// Config
// ===================================================================

/// Per-level Reed-Solomon inverse rate (log₂). The CORE Ligerito idea is to
/// **decrease the rate at deeper levels**: at level i, lower rate ⟹ Johnson
/// list-decoding per-query error = √ρ ≈ 2^(-log_inv_rate/2) ⟹ fewer queries
/// needed for the same security ⟹ drastically smaller opened-rows cost at
/// deeper levels.
///
/// `log_inv_rates[i]` is the log inverse rate at commit i (so wtns_0 uses
/// `log_inv_rates[0]`, wtns_1 uses `log_inv_rates[1]`, …). Length = R + 1.
/// Named parameter profile for the Ligerito PCS. Decouples "which security
/// config" from the raw code rate: `Fast` and `Secure` share rate 1/2 but
/// differ in regime/target, so the rate alone cannot key the config lookup.
///
/// - `Fast`:   rate 1/2, Johnson list-decoding regime with OOD binding,
///             100-bit overall soundness. Default.
/// - `Slim`:   rate 1/4, Johnson + OOD + 16-bit query grinding, 100-bit
///             overall. Roughly half the proof, ~2x the L0 encoding work.
/// - `Secure`: rate 1/2, unique-decoding regime (list size 1, no OOD),
///             120-bit overall soundness. Largest proof, most conservative
///             analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LigeritoProfile {
    #[default]
    Fast,
    Slim,
    Secure,
}

impl LigeritoProfile {
    /// L0 code rate index for this profile (`rho_0 = 2^-log_inv_rate`).
    pub fn log_inv_rate(self) -> usize {
        match self {
            Self::Fast | Self::Secure => 1,
            Self::Slim => 2,
        }
    }
    /// Round-by-round soundness target (bits) the profile's configs are derived
    /// for: every round must individually clear this level (total security =
    /// min over rounds, per the Fiat-Shamir / `soundcalc` convention).
    pub fn security_bits(self) -> usize {
        match self {
            Self::Fast | Self::Slim => 100,
            Self::Secure => 120,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Slim => "slim",
            Self::Secure => "secure",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "slim" => Some(Self::Slim),
            "secure" => Some(Self::Secure),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProverConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,
    /// Per-level query counts (L0, L1, ..., L_r). Length = recursive_steps + 1.
    /// `default_config` fills these via [`udr_queries`]; for tighter
    /// (or stronger) per-level numbers, load a [`LigeritoSecurityConfig`].
    pub queries: Vec<usize>,
    /// Per-level **query-phase** PoW grinding bits (L0, L1, ..., L_r), ground
    /// post-commit/pre-queries. Length = recursive_steps + 1. Each bit here
    /// substitutes for ~1/log₂(1/(1−γ)) queries at that level.
    pub grinding_bits: Vec<usize>,
    /// Per-level **fold-challenge** PoW grinding bits (L0, ..., L_r), ground
    /// immediately before EACH of the level's fold challenges (so a level
    /// with `k` folds does `k` grinds of this many bits). Boosts the
    /// proximity-gap term, which lives on the fold challenges. Length =
    /// recursive_steps + 1.
    pub fold_grinding_bits: Vec<usize>,
    /// Per-commit-level out-of-domain samples (L0, ..., L_r), taken right
    /// after the level's Merkle root enters the transcript. `[0]` must be 0:
    /// L0 is bound by the opening's own (post-commit, random-point)
    /// evaluation claim. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,
    /// Per-level query counts. Length = recursive_steps + 1.
    pub queries: Vec<usize>,
    /// Per-level query-phase PoW grinding bits. Length = recursive_steps + 1.
    pub grinding_bits: Vec<usize>,
    /// Per-level fold-challenge PoW grinding bits (one grind per fold
    /// challenge of the level). Length = recursive_steps + 1.
    pub fold_grinding_bits: Vec<usize>,
    /// Per-commit-level OOD samples. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
}

/// Proximity loss `ε*` for the UDR (unique-decoding regime) analysis. It
/// would back the proximity radius off to `γ = δ/2 − ε*` (δ = 1 − ρ the
/// code's relative distance); set to `0`, so we decode to the full
/// unique-decoding radius `γ = δ/2` with no backoff. Per our paper's Appendix
/// C.3 (Theorem `ca-udr`, BCHKS25 Cor. 1.4) the proximity-gap exceptional set
/// is then `a = γ·n + 1` — length-dependent (see [`paper_thm_1_4_log_a`]), so
/// `eps_pg = 128 − log₂ a` shrinks ~1 bit per witness doubling and is
/// recovered by `fold_grinding_bits`.
pub const UDR_PROXIMITY_LOSS: f64 = 0.0;

/// Soundness (in bits) the query phase must close on its own at every level
/// (the "100 bits from queries always" policy).
const UDR_TARGET_BITS: f64 = 100.0;

/// Number of queries for 100-bit soundness in the **unique-decoding regime**
/// at rate `2^(-log_inv_rate)`: `γ = δ/2 = (1−ρ)/2`, per-query soundness
/// `log₂(1/(1−γ))` (see [`udr_per_query_bits`]). Within the unique decoding
/// radius the prover is pinned to a single codeword, so there is no list and
/// no union-bound term — queries close the full target by themselves.
/// Per-query soundness saturates below 1 bit (`γ < 1/2`), so slimmer codes
/// bottom out near `UDR_TARGET_BITS` queries: 243 at rate 1/2, 148 at 1/4,
/// 121 at 1/8, 110 at 1/16, 105 at 1/32.
pub fn udr_queries(log_inv_rate: usize) -> usize {
    assert!(log_inv_rate > 0, "log_inv_rate=0 (rate 1) has no soundness");
    let per_q = udr_per_query_bits_asymptotic(log_inv_rate);
    (UDR_TARGET_BITS / per_q).ceil() as usize
}

/// Build a sensible default Ligerito config from the upstream PCS shape.
/// `log_n` is the packed-witness log size (= `m - LOG_PACKING`); `log_batch_size`
/// and `log_inv_rate` come from `PcsParams` (Ligerito's `initial_k` matches
/// `log_batch_size` for L0 reuse; the first rate matches `log_inv_rate`).
///
/// Strategy: 3-bit recursive folds (`k_i = 3`) with **decreasing rate**
/// (one rate step per recursive level) until the residual is small (`≤ 5` bits).
/// Asserts that the chosen rate keeps `block_len ≥ udr_queries(rate)` at
/// every level; if not, bumps the rate further.
///
/// Returns `Err` when no feasible config exists (e.g. `log_n` is too small).
pub fn default_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<ProverConfig, &'static str> {
    let initial_k = log_batch_size;
    if log_n <= initial_k {
        return Err("log_n must be > initial_k");
    }

    let mut log_inv_rates = vec![log_inv_rate];
    let mut recursive_ks = Vec::new();
    let mut recursive_log_msg_cols = Vec::new();

    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;

    // L0 feasibility check.
    {
        let block_len_log = n_running + rate_running;
        let qs = udr_queries(rate_running);
        if (1usize << block_len_log) < qs {
            return Err("L0 block_len < udr_queries — log_n too small for chosen rate");
        }
    }

    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;
        // Pick the smallest rate ≥ rate_running+1 such that block_len ≥ queries.
        let mut next_rate = rate_running + 1;
        loop {
            let bl = 1usize << (log_msg_cols_next + next_rate);
            let qs = udr_queries(next_rate);
            if bl >= qs {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)");
            }
        }
        recursive_log_msg_cols.push(log_msg_cols_next);
        recursive_ks.push(k);
        log_inv_rates.push(next_rate);
        n_running -= k;
        rate_running = next_rate;
    }

    if recursive_ks.is_empty() {
        return Err("log_n too small — no recursive levels needed (use BaseFold directly)");
    }

    let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
    let n_levels = log_inv_rates.len();
    let grinding_bits = vec![0usize; n_levels];

    Ok(ProverConfig {
        log_inv_rates: log_inv_rates.clone(),
        recursive_steps: recursive_ks.len(),
        initial_log_msg_cols: log_n - initial_k,
        initial_log_num_interleaved: initial_k,
        initial_k,
        recursive_log_msg_cols,
        recursive_ks,
        queries,
        grinding_bits,
        fold_grinding_bits: vec![0usize; n_levels],
        ood_samples: vec![0usize; n_levels],
    })
}

/// Recursion-ladder shape: per-level dims (index 0 = L0) plus the residual.
struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    log_num_interleaved: Vec<usize>,
    k_recursive: Vec<usize>,
    yr_log_n: usize,
}

/// Shared shape derivation behind [`default_config`] and
/// [`LigeritoSecurityConfig::derive_profile`]: 3-bit recursive folds with the
/// rate index increasing by ≥ 1 per level, bumped further whenever the block
/// length couldn't accommodate `queries_at_rate(rate)` distinct queries.
fn derive_ladder_shape(
    log_n: usize,
    initial_k: usize,
    log_inv_rate: usize,
    queries_at_rate: &dyn Fn(usize) -> usize,
) -> Result<LadderShape, String> {
    if log_n <= initial_k {
        return Err("log_n must be > initial_k".into());
    }
    let mut shape = LadderShape {
        log_inv_rates: vec![log_inv_rate],
        log_msg_cols: vec![log_n - initial_k],
        log_num_interleaved: vec![initial_k],
        k_recursive: vec![initial_k],
        yr_log_n: 0,
    };
    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;
    if (1usize << (n_running + rate_running)) < queries_at_rate(rate_running) {
        return Err("L0 block_len < queries — log_n too small for chosen rate".into());
    }
    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;
        let mut next_rate = rate_running + 1;
        loop {
            if (1usize << (log_msg_cols_next + next_rate)) >= queries_at_rate(next_rate) {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)".into());
            }
        }
        shape.log_inv_rates.push(next_rate);
        shape.log_msg_cols.push(log_msg_cols_next);
        shape.log_num_interleaved.push(k);
        shape.k_recursive.push(k);
        n_running -= k;
        rate_running = next_rate;
    }
    if shape.k_recursive.len() < 2 {
        return Err("log_n too small — no recursive levels needed (use BaseFold directly)".into());
    }
    shape.yr_log_n = n_running;
    Ok(shape)
}

/// Embedded security-spec TOML files. The lookup table maps `(m, profile)`
/// to a TOML payload that's hash-independent (Ligerito's shape only depends
/// on `log_n = m − LOG_PACKING`). Regenerate with
/// `cargo run --release --example gen_ligerito_configs`.
macro_rules! profile_configs {
    ($($m:literal),+ $(,)?) => {
        &[
            $(
                (($m, LigeritoProfile::Fast),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_fast.toml"))),
                (($m, LigeritoProfile::Slim),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_slim.toml"))),
                (($m, LigeritoProfile::Secure),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_secure.toml"))),
            )+
        ]
    };
}
const EMBEDDED_CONFIGS: &[((usize, LigeritoProfile), &str)] =
    profile_configs!(22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

/// Look up the embedded security config TOML for `(m, profile)`.
/// Returns `None` if no config has been derived for this combination yet.
pub fn embedded_security_config(m: usize, profile: LigeritoProfile) -> Option<&'static str> {
    EMBEDDED_CONFIGS.iter().find_map(|&(key, toml)| {
        if key == (m, profile) {
            Some(toml)
        } else {
            None
        }
    })
}

/// Build a `ProverConfig` for `(log_n, log_batch_size, log_inv_rate)` from
/// the embedded security TOML. **Strict**: returns `Err` if no security
/// config has been derived for `(m, log_inv_rate)`. Use this as the
/// production entry point; never silently falls back to default parameters
/// with weaker (or unverified) soundness.
///
/// For ad-hoc / testing shapes where a security spec hasn't been derived,
/// callers can use [`default_config`] explicitly — but that's
/// `#[deprecated]` outside of test code because the per-level parameters
/// haven't been audited.
pub fn prover_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<ProverConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let toml = embedded_security_config(m, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={}). \
             Add a TOML at configs/ligerito/m{m}_{}.toml and register it in \
             EMBEDDED_CONFIGS, or call default_config explicitly for ad-hoc shapes.",
            profile.as_str(),
            profile.as_str(),
        )
    })?;
    let sec = LigeritoSecurityConfig::from_toml_str(toml)?;
    if sec.initial_k != log_batch_size {
        return Err(format!(
            "embedded config for (m={m}, profile={}) has \
             initial_k={} but caller requested log_batch_size={log_batch_size}",
            profile.as_str(),
            sec.initial_k
        ));
    }
    let (pv, _) = sec.to_prover_verifier_configs()?;
    Ok(pv)
}

/// Verifier-side counterpart to [`prover_config_for`]. Same strict lookup.
pub fn verifier_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<VerifierConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let toml = embedded_security_config(m, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={})",
            profile.as_str()
        )
    })?;
    let sec = LigeritoSecurityConfig::from_toml_str(toml)?;
    if sec.initial_k != log_batch_size {
        return Err(format!(
            "embedded config for (m={m}, profile={}) has \
             initial_k={} but caller requested log_batch_size={log_batch_size}",
            profile.as_str(),
            sec.initial_k
        ));
    }
    let (_, vc) = sec.to_prover_verifier_configs()?;
    Ok(vc)
}

/// Verifier-side counterpart to [`default_config`].
pub fn default_verifier_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<VerifierConfig, &'static str> {
    let p = default_config(log_n, log_batch_size, log_inv_rate)?;
    Ok(VerifierConfig {
        log_inv_rates: p.log_inv_rates,
        recursive_steps: p.recursive_steps,
        initial_log_msg_cols: p.initial_log_msg_cols,
        initial_log_num_interleaved: p.initial_log_num_interleaved,
        initial_k: p.initial_k,
        recursive_log_msg_cols: p.recursive_log_msg_cols,
        recursive_ks: p.recursive_ks,
        queries: p.queries,
        grinding_bits: p.grinding_bits,
        fold_grinding_bits: p.fold_grinding_bits,
        ood_samples: p.ood_samples,
    })
}

// ===================================================================
// Security configuration schema
// ===================================================================
//
// Auditable, per-level spec for a Ligerito instance: query count, grinding
// bits, slack-from-Johnson, and the proximity-gap analysis the parameters
// were derived under. Designed to be (de)serializable so it can live in a
// TOML/JSON file alongside the prover/verifier code.

/// Which proximity-gap analysis a level's parameters were derived under.
/// Determines which formulas the implementation should verify against the
/// declared (η, queries, grinding) tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoundnessRegime {
    /// Unique decoding radius: γ = δ/2 (δ = 1 − ρ the code's relative
    /// distance; no proximity-loss backoff). Theorem `ca-udr` of our paper's
    /// Appendix C.3 (adapted from Ben-Sasson–Carmon–Haböck–Kopparty–Saraf
    /// "On Proximity Gaps for Reed–Solomon Codes", 2025, Corollary 1.4): the
    /// exceptional set is `a = γ·n + 1`, growing with the codeword length `n`,
    /// so the proximity-gap term is recovered per level by `fold_grinding_bits`
    /// rather than coming out 0. `eta` is `None` for this regime.
    Udr,
    /// Johnson radius with explicit slack `η` (γ = (1 − √ρ) − η) **with
    /// out-of-domain binding**. Theorem 1.5 of the same paper gives the
    /// proximity-gap exceptional set `a = O_ρ(n / η^5)`; the level's
    /// `fold_grinding_bits` should be ≥ (target_bits − log₂(q/a)).
    /// Binding to a single codeword of the (Johnson-bounded) interleaved list
    /// is via `ood_samples` explicit multilinear OOD evaluations — except at
    /// L0, where the opening's own post-commit random evaluation claim plays
    /// the OOD role (union over the list, `L·μ/q`), so `ood_samples = 0`.
    ///
    /// Note there is deliberately no plain `Johnson` variant: without OOD
    /// binding the query phase pays a union bound over the interleaved list
    /// (≈ 19–52 bits here), which our query counts do not include. A config
    /// claiming Johnson soundness without OOD accounting would be unsound.
    JohnsonOod,
}

/// Where in a level's Fiat-Shamir transcript the grinding step lands.
/// Currently only one choice; reserved for future protocol variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrindingStep {
    /// Grind happens after the level's Merkle root is observed but before
    /// query positions are sampled. Standard FRI/STARK pattern.
    PostCommitPreQueries,
}

/// Parameters for a single level in the recursive Ligerito ladder.
/// L0 = the upstream `pcs::commit` output (reused, not re-committed);
/// L1 .. L_{r−1} are the recursive commits; the final residual `yr` block
/// is described separately in [`FinalBlockConfig`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoLevelConfig {
    /// PCS rate at this level: codeword expansion factor = 2^log_inv_rate.
    pub log_inv_rate: usize,
    /// Message dimension at this level (log of number of F128 columns in
    /// the codeword). `log_msg_cols + log_inv_rate = log_2(block_len)`.
    pub log_msg_cols: usize,
    /// Log of lane width per Merkle leaf at this level. For L0 = `initial_k`;
    /// for L_i (i ≥ 1) = the previous level's k_recursive.
    pub log_num_interleaved: usize,
    /// Number of sumcheck folds taken at this level. For L0 = `initial_k`
    /// (the lane fold); for L_i (i ≥ 1) = the recursive fold k_{i−1}.
    pub k_recursive: usize,
    /// Which proximity-gap analysis the (eta, queries, grinding_bits)
    /// tuple was derived under. Determines the formulas the implementation
    /// validates against.
    pub regime: SoundnessRegime,
    /// Slack from the Johnson radius. Required for the `JohnsonOod` regime;
    /// must be `None` for `Udr`.
    pub eta: Option<f64>,
    /// Proximity loss `ε*` for the UDR radius `γ = δ/2 − ε*` (our paper
    /// App. C.3 / BCHKS25 Cor. 1.4); `0` in the shipped configs (full
    /// unique-decoding radius δ/2, no backoff). Required for `Udr`; must be
    /// `None` for `JohnsonOod`. The exceptional set is `a = γ·n + 1`,
    /// length-dependent (see [`paper_thm_1_4_log_a`]).
    #[serde(default)]
    pub proximity_loss: Option<f64>,
    /// Number of codeword position queries opened at this level (the FRI
    /// query phase). Bounds the per-query soundness term `(1−γ)^Q`.
    pub queries: usize,
    /// **Query-phase** PoW grinding bits, ground post-commit/pre-queries
    /// (see [`GrindingStep`]). Each bit substitutes for
    /// ~1/log₂(1/(1−γ)) queries at this level.
    pub grinding_bits: usize,
    /// **Fold-challenge** PoW grinding bits, ground immediately before EACH
    /// of this level's `k_recursive` fold challenges. Boosts the
    /// proximity-gap term (which lives on the fold challenges):
    /// `eps_pg + fold_grinding_bits ≥ target`.
    #[serde(default)]
    pub fold_grinding_bits: usize,
    /// Out-of-domain samples taken right after this level's commit enters
    /// the transcript (`JohnsonOod` only). Each binds the prover to a single
    /// codeword of the interleaved list via a multilinear evaluation claim.
    /// Must be 0 at L0 (bound by the opening's own post-commit evaluation
    /// claim) and ≥ 1 at deeper `JohnsonOod` levels.
    #[serde(default)]
    pub ood_samples: usize,
    /// Security target this level guarantees, post-grinding.
    pub target_security_bits: usize,
    /// Diagnostic — `log₂(q/a)` under the chosen regime. The implementation
    /// should assert this matches the formula at startup, modulo rounding.
    pub expected_eps_pg_bits: f64,
    /// Diagnostic — `Q · log₂(1/(1−γ))`. Should be ≥
    /// `target_security_bits − grinding_bits`.
    pub expected_eps_query_bits: f64,
    /// Diagnostic — OOD binding bits (`JohnsonOod` only):
    /// `s·(128 − log₂μ) − (2·log₂L − 1)` for explicit samples, or
    /// `128 − log₂L − log₂μ` for the implicit L0 binding, where `L` is the
    /// Johnson interleaved list size and `μ` the level's variable count.
    #[serde(default)]
    pub expected_eps_ood_bits: Option<f64>,
}

/// Descriptor for the final-residual block (`yr`) sent in the clear at the
/// end of the last recursive level. It has no commit and no queries, so the
/// only meaningful parameter is its dimension.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalBlockConfig {
    /// `log_2(|yr|)` — number of F128 values sent in the clear. The last
    /// recursive level's sumcheck stops at this dim instead of folding to 1.
    pub yr_log_n: usize,
}

/// Complete security spec for one Ligerito instance, covering a single
/// `(hash, m)` pair. Designed to round-trip cleanly via serde (TOML/JSON).
///
/// **Validation invariants** (checked by [`Self::validate`]):
/// 1. `initial_k + Σ levels[1..].k_recursive + final_block.yr_log_n == log_n`.
/// 2. Each level's `expected_eps_pg_bits` is consistent with the declared
///    regime and `eta` (within tolerance).
/// 3. Each level's `expected_eps_query_bits ≥ target_security_bits −
///    grinding_bits` (queries cover what grinding doesn't).
/// 4. `eta` is `Some` iff regime ∈ {Johnson, JohnsonOod}; `None` for Udr.
/// 5. `log_msg_cols`, `log_num_interleaved`, `k_recursive` match the
///    recursive-shape constraint (each level's input dim equals the
///    previous level's `log_msg_cols`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoSecurityConfig {
    /// Block-encoder log size: m = log₂(witness bit count).
    pub m: usize,
    /// Packed-witness log dim (`= m − LOG_PACKING = m − 7`).
    pub log_n: usize,
    /// L0 lane fold. Must equal the upstream `PcsParams::log_batch_size` so
    /// the L0 commit can be reused without re-committing.
    pub initial_k: usize,
    /// Round-by-round security target (bits): validate() asserts every error
    /// term at every round (round-by-round soundness) clears at least this
    /// much. Total security is the *minimum* over rounds — the notion that
    /// governs Fiat-Shamir security (cf. Ethereum's `soundcalc`) — so there is
    /// deliberately no whole-protocol union bound over terms.
    pub target_security_bits: usize,
    /// Identifier of the proximity-gap analysis used. Self-documents which
    /// theorem the per-level parameters were derived from. Example:
    /// `"ben_sasson_2025_thm_4_6"`.
    pub analysis_version: String,
    /// Field of the protocol. Example: `"f128"`.
    pub field: String,
    /// Hash function used by Merkle + FS challenger. Example: `"sha256"`.
    pub hash: String,
    /// Where in the per-level FS transcript grinding is placed.
    pub grinding_step: GrindingStep,
    /// Per-level parameters, in order L0, L1, L2, ....
    pub levels: Vec<LigeritoLevelConfig>,
    /// Final residual block descriptor.
    pub final_block: FinalBlockConfig,
}

/// Default field size used for soundness analysis: `q = 2^128` (our F128).
const ANALYSIS_LOG_Q: f64 = 128.0;

/// Round a float to one decimal place. Used to round paper-predicted
/// soundness diagnostics so the generated TOMLs stay readable.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Bit-level tolerance when comparing declared diagnostics
/// (`expected_eps_pg_bits` / `expected_eps_query_bits`) against the value
/// computed from the regime's formulas. Set generously enough that rounding
/// in the TOML doesn't cause spurious failures, but tightly enough that an
/// incorrect declaration of η, Q, or grinding can't slip through.
const PAPER_COMPAT_TOL_BITS: f64 = 0.6;

/// Proximity-gap exceptional set for the list-decoding (Johnson) regime, per
/// our paper's Appendix C.3 (Theorem `ca-johnson`, adapted from BCHKS25
/// Theorem 4.6). For a Reed–Solomon code of rate `ρ`, codeword length `n`,
/// and Johnson slack `η` (proximity radius `γ = 1 − √ρ − η`), the MCA error is
/// `a/|F|` with
///
///   `a = [2(m+½)^5 + 3(m+½)·γ·ρ] / (3·ρ^{3/2}) · n + (m+½)/√ρ`,
///
/// where `η = 1 − √ρ − γ` and `m = max(⌈√ρ/(2η)⌉, 3)`. Returns `log₂ a`.
///
/// This is the per-fold-step MCA error, stated for a two-row interleaved word
/// (`C ∈ F^{2×n}`). The ℓ-round lane fold of a `2^ℓ`-interleaved word adds a
/// row-union factor via App. C.3's Lemma `mca-commutes`; see
/// [`paper_johnson_log_a`].
fn paper_thm_ca_johnson_log_a(log_inv_rate: usize, eta: f64, log_msg_cols: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let gamma = 1.0 - sqrt_rho - eta;
    // m = ⌈√ρ/(2η)⌉ where η = 1−√ρ−γ, floored at 3.
    let m_param = ((sqrt_rho / (2.0 * eta)).ceil() as usize).max(3) as f64;
    let half = m_param + 0.5;
    let half5 = half.powi(5);
    let numerator = 2.0 * half5 + 3.0 * half * gamma * rho;
    let denominator = 3.0 * rho.powf(1.5);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    let a = (numerator / denominator) * n + half / sqrt_rho;
    a.log2()
}

/// Johnson-regime proximity-gap `log₂ a` for a level, including the row-union
/// factor from our paper's Appendix C.3 (Lemma `mca-commutes`, "MCA commutes
/// with list decoding").
///
/// The base MCA error `ε = a_RLC/|F|` from [`paper_thm_ca_johnson_log_a`] is
/// stated for a two-row interleaved word (one fold step). Folding a
/// `2^ℓ`-interleaved word (ℓ = `log_num_interleaved`) over its ℓ lane-fold
/// rounds pays a row union: by the lemma, round `i` incurs `2^{ℓ-i}·ε`, so the
/// worst round (`i = 1`) pays the factor `2^{ℓ-1}` = (interleaving factor)/2.
/// We bind the per-level grinding to that worst round, returning
/// `log₂(2^{ℓ-1}·a_RLC) = log₂ a_RLC + (ℓ-1)`.
///
/// `ℓ ≤ 1` (`L ≤ 2`) means no row union; the `(ℓ-1)` penalty clamps to 0.
fn paper_johnson_log_a(
    log_inv_rate: usize,
    eta: f64,
    log_msg_cols: usize,
    log_num_interleaved: usize,
) -> f64 {
    let base = paper_thm_ca_johnson_log_a(log_inv_rate, eta, log_msg_cols);
    // Row-union factor 2^{ℓ-1} (worst round i=1 of the ℓ-round lane fold),
    // ℓ = log_num_interleaved. In bits: (ℓ-1), clamped ≥ 0.
    let row_union_penalty = (log_num_interleaved as f64 - 1.0).max(0.0);
    base + row_union_penalty
}

/// Per-query log₂(1/(1−γ)) under the Johnson regime: each query closes
/// `log_2(1/(1-γ))` bits of soundness against a γ-far adversary.
fn paper_per_query_bits(log_inv_rate: usize, eta: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = 1.0 - rho.sqrt() - eta;
    (1.0 / (1.0 - gamma)).log2()
}

/// UDR proximity radius: the **maximum** allowed by our paper's App. C.3
/// (Theorem `ca-udr`, BCHKS25 Cor. 1.4), whose valid range is
/// `[δ/3, δ/2 − 3/(δ·n)]`. We take the top of the range,
///
///   `γ = δ/2 − 3/(δ·n) − ε*`,
///
/// where `δ = 1 − ρ` is the code's relative minimum distance,
/// `n = 2^(log_msg_cols + log_inv_rate)` the codeword length, and `ε*`
/// (`proximity_loss`) optional extra slack below the maximum (`0` in shipped
/// configs → exactly the maximal radius). The `3/(δ·n)` backoff is the
/// theorem-mandated minimum and shrinks with the codeword length.
fn udr_gamma(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let delta = 1.0 - rho;
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    delta / 2.0 - 3.0 / (delta * n) - proximity_loss
}

/// Per-query log₂(1/(1−γ)) under the UDR regime at the maximal radius
/// `γ = δ/2 − 3/(δ·n) − ε*` (see [`udr_gamma`]).
fn udr_per_query_bits(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    (1.0 / (1.0 - gamma)).log2()
}

/// Asymptotic (n → ∞) UDR per-query soundness at `γ = δ/2`, dropping the
/// finite-length `3/(δ·n)` backoff. Length-agnostic — used for ladder-shape
/// feasibility and [`udr_queries`]; the shipped per-level configs use the
/// n-aware [`udr_per_query_bits`]. The dropped backoff slightly *under*-counts
/// queries, but the per-level block-length check in `derive_profile` (and the
/// `+5` feasibility padding) catch any shape that wouldn't hold the real,
/// n-aware query count.
fn udr_per_query_bits_asymptotic(log_inv_rate: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = (1.0 - rho) / 2.0;
    (1.0 / (1.0 - gamma)).log2()
}

/// UDR proximity-gap exceptional set, per our paper's Appendix C.3
/// (Theorem `ca-udr`, adapted from BCHKS25 Corollary 1.4): at proximity
/// radius `γ` (here the maximal `γ = δ/2 − 3/(δ·n)`; see [`udr_gamma`]) the
/// exceptional set is
///
///   `a = γ·n + 1`,
///
/// where `n = 2^(log_msg_cols + log_inv_rate)` is the codeword length at this
/// level. The `log₂ a ≈ log₂(γ·n)` term therefore **grows with the codeword
/// length**, so larger witnesses give a smaller `eps_pg = 128 − log₂ a` and
/// need proportionally more `fold_grinding_bits` to hold a fixed target.
/// Callers add **no** row-union penalty in this regime: the unique-decoding
/// list has size 1, so (per Diamond and Gruen) MCA-commutes holds with error
/// ε directly, unlike the Johnson regime's `2^{ℓ-1}` factor. This replaced an
/// earlier length-independent `a ≤ 2/ε*` form, which did not match the paper's
/// stated bound.
fn paper_thm_1_4_log_a(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    (gamma * n + 1.0).log2()
}

/// Johnson-bound list size of the *interleaved* RS code at radius
/// `θ = 1 − √ρ − η`, in log₂. Independent of the interleaving factor.
///
/// Interleaving preserves relative distance — `V^{⊙m}` has the base code's
/// distance `δ = 1 − ρ` — and only enlarges the alphabet (to `q^m`). The
/// Johnson bound depends solely on (distance, radius, alphabet size), so the
/// interleaved list size at any radius *below* the Johnson radius `1 − √ρ`
/// is bounded by the very same single-code Johnson list size
///
///   `L_int ≤ L_base ≤ 1/(2·η·√ρ)`,
///
/// with no dependence on `m` and, crucially, no `L_base^r` blow-up.
///
/// The general GGR (Gopalan–Guruswami–Raghavendra, Thm 2.5) interleaved bound
/// `L_int ≤ C(b+r, r)·L_base^r` is only needed to push the list-decoding
/// radius *past* the Johnson bound toward `δ`. Ligerito deliberately sits at
/// `θ = 1 − √ρ − η`, strictly below the Johnson radius by slack `η > 0`, so
/// that regime never applies and the plain Johnson bound is both correct and
/// far tighter (it dominates GGR throughout the regime RS can reach).
fn johnson_interleaved_list_log2(log_inv_rate: usize, eta: f64) -> f64 {
    debug_assert!(
        eta > 0.0,
        "η must be > 0 to stay strictly below the Johnson radius"
    );
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let l_base = 1.0 / (2.0 * eta * sqrt_rho);
    l_base.log2()
}

/// OOD binding bits for a `JohnsonOod` level. `mu_vars` is the level's
/// multilinear variable count (`log_msg_cols + log_num_interleaved`).
///
/// - `ood_samples ≥ 1` (explicit samples): the bad event is two distinct
///   list elements agreeing on all `s` random points of `F^μ`
///   (Schwartz–Zippel, total degree ≤ μ), union over pairs:
///       bits = s·(128 − log₂ μ) − (2·log₂ L_int − 1).
/// - `ood_samples = 0` (L0's implicit binding): the opening's own evaluation
///   claim at a post-commit random point pins the prover to one claimed
///   value, so the union is over the list (not pairs):
///       bits = 128 − log₂ L_int − log₂ μ.
fn paper_ood_bits(log_inv_rate: usize, eta: f64, mu_vars: usize, ood_samples: usize) -> f64 {
    let log2_l = johnson_interleaved_list_log2(log_inv_rate, eta);
    let log2_mu = (mu_vars as f64).log2();
    if ood_samples == 0 {
        ANALYSIS_LOG_Q - log2_l - log2_mu
    } else {
        ood_samples as f64 * (ANALYSIS_LOG_Q - log2_mu) - (2.0 * log2_l - 1.0)
    }
}

impl LigeritoLevelConfig {
    /// Compute the proximity-gap and per-query soundness bits this level is
    /// expected to deliver under its declared regime. Returns
    /// `(eps_pg_bits, eps_query_bits)` where:
    ///   eps_pg_bits   = log₂(q/a) under the regime's threshold-a formula
    ///   eps_query_bits = Q · log₂(1/(1−γ))
    ///
    /// Used by [`LigeritoSecurityConfig::validate`] to assert the declared
    /// `expected_*_bits` diagnostics are consistent with the regime's
    /// canonical formulas (i.e., the config is compatible with the paper).
    pub fn paper_predicted_bits(&self) -> (f64, f64) {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                // App. C.3 Lemma `mca-commutes`: the ℓ-round lane fold of a
                // 2^ℓ-interleaved word (ℓ = log_num_interleaved) pays a
                // row-union factor 2^{ℓ-i} at round i; the worst round (i=1)
                // gives 2^{ℓ-1}, on top of the base ca-johnson MCA error.
                let log_a = paper_johnson_log_a(
                    self.log_inv_rate,
                    eta,
                    self.log_msg_cols,
                    self.log_num_interleaved,
                );
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                // Per-query soundness WITHOUT a list union bound — the OOD
                // binding (see `paper_ood_bits`) pins the prover to a single
                // codeword of the interleaved list before queries are drawn.
                let per_q = paper_per_query_bits(self.log_inv_rate, eta);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
            SoundnessRegime::Udr => {
                // App. C.3 Thm `ca-udr` (BCHKS25 Cor. 1.4): a = γ·n + 1 for
                // radius γ = δ/2 (ε* = 0, no backoff).
                let proximity_loss = self
                    .proximity_loss
                    .expect("Udr regime must carry proximity_loss");
                // No row-union penalty in the unique-decoding regime: the list
                // has size 1, so (per Diamond and Gruen) the MCA-commutes step
                // holds with error ε directly — the Johnson regime's 2^{ℓ-1}
                // row union is unnecessary. So eps_pg = 128 − log₂ a.
                let log_a =
                    paper_thm_1_4_log_a(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                let per_q =
                    udr_per_query_bits(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
        }
    }

    /// OOD binding bits this level is expected to deliver (`JohnsonOod`
    /// only; `None` for `Udr`, where the unique-decoding list has size 1 and
    /// no binding step exists). See [`paper_ood_bits`].
    pub fn paper_predicted_ood_bits(&self) -> Option<f64> {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                let mu = self.log_msg_cols + self.log_num_interleaved;
                Some(paper_ood_bits(self.log_inv_rate, eta, mu, self.ood_samples))
            }
            SoundnessRegime::Udr => None,
        }
    }
}

impl LigeritoSecurityConfig {
    /// Validate that the config is internally consistent and matches the
    /// declared analysis. Returns the first violation found, if any.
    pub fn validate(&self) -> Result<(), String> {
        if self.log_n + 7 != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING (7) != m ({})",
                self.log_n, self.m
            ));
        }

        // Recursion shape: initial_k + Σ k_recursive (L1+) + yr_log_n = log_n.
        let levels_recursive_sum: usize = self.levels.iter().skip(1).map(|lv| lv.k_recursive).sum();
        let yr_log_n = self.final_block.yr_log_n;
        if self.initial_k + levels_recursive_sum + yr_log_n != self.log_n {
            return Err(format!(
                "shape mismatch: initial_k ({}) + Σ k_recursive ({}) + yr_log_n ({}) = {} ≠ log_n ({})",
                self.initial_k,
                levels_recursive_sum,
                yr_log_n,
                self.initial_k + levels_recursive_sum + yr_log_n,
                self.log_n,
            ));
        }

        // L0 must have k_recursive = initial_k and log_num_interleaved = initial_k.
        let l0 = self
            .levels
            .first()
            .ok_or_else(|| "empty levels".to_string())?;
        if l0.k_recursive != self.initial_k {
            return Err(format!(
                "L0.k_recursive ({}) must equal initial_k ({})",
                l0.k_recursive, self.initial_k
            ));
        }
        if l0.log_num_interleaved != self.initial_k {
            return Err(format!(
                "L0.log_num_interleaved ({}) must equal initial_k ({})",
                l0.log_num_interleaved, self.initial_k
            ));
        }

        // Per-level checks.
        let mut dim_in = self.log_n;
        for (i, lv) in self.levels.iter().enumerate() {
            // Shape: log_msg_cols + log_num_interleaved = dim_in.
            if lv.log_msg_cols + lv.log_num_interleaved != dim_in {
                return Err(format!(
                    "L{i}: log_msg_cols ({}) + log_num_interleaved ({}) ≠ input dim ({dim_in})",
                    lv.log_msg_cols, lv.log_num_interleaved
                ));
            }

            // eta presence matches regime.
            match (lv.regime, lv.eta) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but eta is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!("L{i}: regime requires eta but eta is None"));
                }
                _ => {}
            }

            // proximity_loss presence matches regime (UDR-only).
            match (lv.regime, lv.proximity_loss) {
                (SoundnessRegime::Udr, None) => {
                    return Err(format!("L{i}: regime=udr but proximity_loss is missing"));
                }
                (SoundnessRegime::Udr, Some(eps)) if eps < 0.0 => {
                    return Err(format!("L{i}: proximity_loss must be ≥ 0, got {eps}"));
                }
                (SoundnessRegime::JohnsonOod, Some(_)) => {
                    return Err(format!("L{i}: proximity_loss is only valid for regime=udr"));
                }
                _ => {}
            }

            // OOD samples match regime: UDR has no list, so no OOD; under
            // JohnsonOod every level past L0 needs explicit samples, while
            // L0 is bound by the opening's own post-commit evaluation claim.
            match lv.regime {
                SoundnessRegime::Udr if lv.ood_samples != 0 => {
                    return Err(format!(
                        "L{i}: regime=udr but ood_samples={} (unique decoding \
                         has list size 1 — no OOD binding step exists)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i == 0 && lv.ood_samples != 0 => {
                    return Err(format!(
                        "L0: ood_samples={} but L0 is bound by the opening's \
                         own evaluation claim (must be 0)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i > 0 && lv.ood_samples == 0 => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires ood_samples ≥ 1 \
                         past L0 (the query counts assume single-codeword \
                         binding)"
                    ));
                }
                _ => {}
            }

            // OOD diagnostic matches regime + formula.
            match (lv.regime, lv.expected_eps_ood_bits) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but expected_eps_ood_bits is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires expected_eps_ood_bits"
                    ));
                }
                (SoundnessRegime::JohnsonOod, Some(declared)) => {
                    let pred = lv
                        .paper_predicted_ood_bits()
                        .expect("JohnsonOod has an OOD prediction");
                    if (declared - pred).abs() > PAPER_COMPAT_TOL_BITS {
                        return Err(format!(
                            "L{i}: expected_eps_ood_bits ({declared:.2}) doesn't \
                             match prediction ({pred:.2}); tolerance ±{:.2} bits.",
                            PAPER_COMPAT_TOL_BITS
                        ));
                    }
                }
                _ => {}
            }

            // Paper-compatibility: the declared expected_*_bits must agree
            // with what the regime's formula predicts (within tolerance).
            // Asserts the config was actually derived from the paper, not
            // hand-waved into compliance.
            let (pg_pred, q_pred) = lv.paper_predicted_bits();
            if (lv.expected_eps_pg_bits - pg_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits. \
                     Re-derive Q, eta, or grinding so the declared diagnostic \
                     matches the formula.",
                    lv.expected_eps_pg_bits,
                    pg_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }
            if (lv.expected_eps_query_bits - q_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits.",
                    lv.expected_eps_query_bits,
                    q_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }

            // Security: queries cover the gap left by grinding.
            if lv.target_security_bits > lv.grinding_bits
                && lv.expected_eps_query_bits + 1e-3
                    < (lv.target_security_bits - lv.grinding_bits) as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) < target ({}) - grinding ({}) = {}",
                    lv.expected_eps_query_bits,
                    lv.target_security_bits,
                    lv.grinding_bits,
                    lv.target_security_bits - lv.grinding_bits
                ));
            }

            // Per-application proximity gap + fold-challenge grinding must
            // reach target. (The pg bad event lives on the fold challenges,
            // so only the fold grind — done before each fold challenge —
            // boosts it; the query-phase grind does not.)
            if lv.expected_eps_pg_bits + lv.fold_grinding_bits as f64 + 1e-3
                < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) + fold_grinding ({}) < target ({})",
                    lv.expected_eps_pg_bits, lv.fold_grinding_bits, lv.target_security_bits
                ));
            }

            // OOD binding must reach target on its own (no grind covers it;
            // escalate ood_samples instead).
            if let Some(ood) = lv.expected_eps_ood_bits
                && ood + 1e-3 < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_ood_bits ({ood:.2}) < target ({}); \
                         increase ood_samples",
                    lv.target_security_bits
                ));
            }

            if lv.target_security_bits < self.target_security_bits {
                return Err(format!(
                    "L{i}: target_security_bits ({}) < global target ({})",
                    lv.target_security_bits, self.target_security_bits
                ));
            }

            // Advance dim_in for next level: subtract k_recursive (the folds at this level).
            dim_in -= lv.k_recursive;
        }

        if dim_in != yr_log_n {
            return Err(format!(
                "after consuming all levels, dim_in ({dim_in}) ≠ yr_log_n ({yr_log_n})"
            ));
        }

        // Round-by-round soundness: each error term at each round is checked
        // against `target_security_bits` in the per-level loop above. Total
        // security is the minimum over rounds (the Fiat-Shamir-relevant notion;
        // cf. Ethereum's `soundcalc`), so there is intentionally no
        // whole-protocol union bound summed across terms.
        Ok(())
    }

    /// Mechanically derive a paper-compatible `LigeritoSecurityConfig` for
    /// `(m, log_inv_rate)` targeting `target_security_bits`, in the
    /// **unique-decoding regime** (BCHKS25 Theorem 1.4). Uses the same
    /// recursion shape as [`default_config`] and picks per-level
    /// `(proximity_loss, queries)` so that each level satisfies:
    ///
    ///   * `expected_eps_query_bits ≥ target_security_bits` (queries alone
    ///     close the target; per the "100 bits from queries always" policy).
    ///   * `expected_eps_pg_bits + fold_grinding_bits ≥ target_security_bits`.
    ///     Under Thm `ca-udr` the exceptional set is `a = γ·n + 1`
    ///     (length-dependent), so `eps_pg = 128 − log₂(γ·n+1) − log₂(log L)`
    ///     decreases with witness size; any shortfall below target is made up
    ///     by `fold_grinding_bits` (query-phase `grinding_bits` stays 0).
    ///
    /// All diagnostic fields are populated from the paper formulas so the
    /// resulting config validates strictly against [`Self::validate`].
    pub fn derive_paper_compatible(
        m: usize,
        log_inv_rate: usize,
        target_security_bits: usize,
    ) -> Result<Self, String> {
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;
        let prover = default_config(log_n, initial_k, log_inv_rate).map_err(|e| e.to_string())?;
        let r = prover.recursive_steps;
        let mut levels = Vec::with_capacity(r + 1);
        // Build per-level (log_msg_cols, log_num_interleaved, k_recursive).
        let mut log_msg_cols_per_level = Vec::with_capacity(r + 1);
        let mut log_num_interleaved_per_level = Vec::with_capacity(r + 1);
        let mut k_recursive_per_level = Vec::with_capacity(r + 1);
        // L0
        log_msg_cols_per_level.push(log_n - initial_k);
        log_num_interleaved_per_level.push(initial_k);
        k_recursive_per_level.push(initial_k);
        for i in 0..r {
            log_msg_cols_per_level.push(prover.recursive_log_msg_cols[i]);
            log_num_interleaved_per_level.push(prover.recursive_ks[i]);
            k_recursive_per_level.push(prover.recursive_ks[i]);
        }
        for i in 0..=r {
            let rate = prover.log_inv_rates[i];
            // UDR: γ = δ/2 = (1−ρ)/2 (ε* = UDR_PROXIMITY_LOSS = 0, no backoff).
            // Thm `ca-udr`'s exceptional set a = γ·n + 1 grows with the
            // codeword length, so eps_pg falls ~1 bit per witness doubling and
            // is recovered by fold_grinding_bits below.
            let proximity_loss = UDR_PROXIMITY_LOSS;
            let per_q = udr_per_query_bits(rate, log_msg_cols_per_level[i], proximity_loss);
            let queries = ((target_security_bits as f64) / per_q).ceil() as usize;
            // No row-union penalty in the unique-decoding regime (list size 1):
            // per Diamond and Gruen, MCA-commutes holds with error ε directly,
            // unlike the Johnson regime's 2^{ℓ-1} row union.
            let log_a = paper_thm_1_4_log_a(rate, log_msg_cols_per_level[i], proximity_loss);
            let eps_pg = ANALYSIS_LOG_Q - log_a;
            // Any pg shortfall is ground on the fold challenges (where the
            // pg bad event lives); 0 at the 100-bit target.
            let fold_grinding_bits =
                ((target_security_bits as f64) - eps_pg).ceil().max(0.0) as usize;
            let eps_query = queries as f64 * per_q;
            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: log_msg_cols_per_level[i],
                log_num_interleaved: log_num_interleaved_per_level[i],
                k_recursive: k_recursive_per_level[i],
                regime: SoundnessRegime::Udr,
                eta: None,
                proximity_loss: Some(proximity_loss),
                queries,
                grinding_bits: 0,
                fold_grinding_bits,
                ood_samples: 0,
                target_security_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: None,
            });
        }
        // Final residual: yr_log_n = log_n − initial_k − Σ k_recursive
        let total_recursive: usize = prover.recursive_ks.iter().sum();
        let yr_log_n = log_n - initial_k - total_recursive;
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits,
            analysis_version: "no_row_union_over_ben_sasson_2025_cor_1_4".into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig { yr_log_n },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Derive the security config for a named [`LigeritoProfile`] at witness
    /// size `m`. Each profile targets its bit level under **round-by-round
    /// soundness**: every error term (pg + fold grinding, query + query
    /// grinding, OOD) clears the target individually, and the protocol's
    /// security is the *minimum* over rounds — the notion that governs
    /// Fiat-Shamir security (cf. Ethereum's `soundcalc`), not a whole-protocol
    /// union bound over terms. The three shipped profiles:
    ///
    /// - `Fast`:   JohnsonOod, rate 1/2, η = 0.02, 100 bits per round.
    /// - `Slim`:   JohnsonOod, rate 1/4, η = 0.02, 16-bit query grinding at
    ///             every level, 100 bits per round.
    /// - `Secure`: Udr, rate 1/2, ε* = 1e-3, 120 bits per round.
    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        /// Johnson slack below the Johnson radius, flat across levels.
        const JOHNSON_ETA: f64 = 0.02;
        let target_bits = profile.security_bits();
        let log_inv_rate = profile.log_inv_rate();
        let query_grind: usize = match profile {
            LigeritoProfile::Slim => 16,
            LigeritoProfile::Fast | LigeritoProfile::Secure => 0,
        };
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;

        // Length-agnostic per-query estimate for ladder-shape feasibility
        // (the per-level codeword length `n` is not known until the shape is
        // fixed). UDR uses the asymptotic γ = δ/2; the actual per-level config
        // below uses the n-aware `udr_per_query_bits`.
        let per_query_bits_feas = |rate: usize| -> f64 {
            match profile {
                LigeritoProfile::Secure => udr_per_query_bits_asymptotic(rate),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, JOHNSON_ETA)
                }
            }
        };

        // Shape derivation needs per-level query counts for block-length
        // feasibility before the level count (and hence the exact per-term
        // target) is known. Use a conservative target of target_bits + 5
        // (≥ log₂(3 terms · 10 levels)); the final counts are ≤ this.
        let t_feas = target_bits as f64 + 5.0;
        let queries_feas = |rate: usize| -> usize {
            ((t_feas - query_grind as f64).max(1.0) / per_query_bits_feas(rate)).ceil() as usize
        };
        let shape = derive_ladder_shape(log_n, initial_k, log_inv_rate, &queries_feas)?;
        let n_levels = shape.log_inv_rates.len();

        // Round-by-round target: every error term (pg, query, ood) at every
        // round must individually clear `target_bits`. Round-by-round soundness
        // — the notion that governs the Fiat-Shamir security of the IOP — is the
        // *minimum* security level over rounds, not the sum, so there is
        // deliberately NO `log₂(#terms)` union-bound headroom. This matches the
        // convention Ethereum's `soundcalc` uses for hash-based zkEVM IOPs
        // (total security = min over rounds). It also keeps the proximity-gap
        // fold grinding (especially L0's, the dominant prover cost) at the
        // round-by-round minimum rather than paying ~4 bits of union slack that
        // buys nothing.
        let t = target_bits as f64;

        let mut levels = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let rate = shape.log_inv_rates[i];
            let cols = shape.log_msg_cols[i];
            let ilv = shape.log_num_interleaved[i];
            // Actual per-level per-query bits: n-aware (maximal radius) for
            // UDR, length-agnostic Johnson otherwise.
            let per_q = match profile {
                LigeritoProfile::Secure => udr_per_query_bits(rate, cols, UDR_PROXIMITY_LOSS),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, JOHNSON_ETA)
                }
            };
            let queries = ((t - query_grind as f64).max(1.0) / per_q).ceil() as usize;
            if queries > (1usize << (cols + rate)) {
                return Err(format!(
                    "L{i}: {queries} queries exceed block length 2^{}",
                    cols + rate
                ));
            }
            let eps_query = queries as f64 * per_q;

            let (regime, eta, proximity_loss, eps_pg, ood_samples, eps_ood) = match profile {
                LigeritoProfile::Secure => {
                    // No row-union penalty in the unique-decoding regime (list
                    // size 1): per Diamond and Gruen, MCA-commutes holds with
                    // error ε directly (vs the Johnson regime's 2^{ℓ-1} factor).
                    let eps_pg =
                        ANALYSIS_LOG_Q - paper_thm_1_4_log_a(rate, cols, UDR_PROXIMITY_LOSS);
                    (
                        SoundnessRegime::Udr,
                        None,
                        Some(UDR_PROXIMITY_LOSS),
                        eps_pg,
                        0usize,
                        None,
                    )
                }
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    let eps_pg = ANALYSIS_LOG_Q - paper_johnson_log_a(rate, JOHNSON_ETA, cols, ilv);
                    let mu = cols + ilv;
                    let ood_samples = if i == 0 {
                        0 // bound by the opening's own evaluation claim
                    } else {
                        (1..=8usize)
                            .find(|&s| paper_ood_bits(rate, JOHNSON_ETA, mu, s) >= t)
                            .ok_or_else(|| {
                                format!("L{i}: no OOD sample count reaches {t:.1} bits")
                            })?
                    };
                    let eps_ood = paper_ood_bits(rate, JOHNSON_ETA, mu, ood_samples);
                    (
                        SoundnessRegime::JohnsonOod,
                        Some(JOHNSON_ETA),
                        None,
                        eps_pg,
                        ood_samples,
                        Some(round1(eps_ood)),
                    )
                }
            };
            let fold_grinding_bits = (t - eps_pg).ceil().max(0.0) as usize;

            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: cols,
                log_num_interleaved: ilv,
                k_recursive: shape.k_recursive[i],
                regime,
                eta,
                proximity_loss,
                queries,
                grinding_bits: query_grind,
                fold_grinding_bits,
                ood_samples,
                target_security_bits: target_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: eps_ood,
            });
        }

        let analysis_version = match profile {
            LigeritoProfile::Secure => "no_row_union_over_ben_sasson_2025_cor_1_4",
            LigeritoProfile::Fast | LigeritoProfile::Slim => {
                "johnson_ood_row_union_over_bchks25_thm_4_6"
            }
        };
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits: target_bits,
            analysis_version: analysis_version.into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig {
                yr_log_n: shape.yr_log_n,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a [`LigeritoSecurityConfig`] from a TOML string and validate it.
    /// The caller is expected to embed the file contents via
    /// `include_str!("../../configs/ligerito/m29_fast.toml")` (for compile-time
    /// configs) or read it via `std::fs` (for runtime configs).
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(s).map_err(|e| format!("toml parse: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Serialize the config back out to TOML. Round-trip-stable with
    /// [`from_toml_str`].
    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("toml serialize: {e}"))
    }

    /// Build a `(ProverConfig, VerifierConfig)` pair from this security config.
    /// Drops the security-only fields (eta, queries, grinding, expected_*) but
    /// preserves the recursion shape so the existing prover/verifier code path
    /// works unchanged.
    pub fn to_prover_verifier_configs(&self) -> Result<(ProverConfig, VerifierConfig), String> {
        self.validate()?;
        let log_inv_rates: Vec<usize> = self.levels.iter().map(|lv| lv.log_inv_rate).collect();
        let recursive_ks: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.k_recursive)
            .collect();
        let recursive_log_msg_cols: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.log_msg_cols)
            .collect();
        let queries: Vec<usize> = self.levels.iter().map(|lv| lv.queries).collect();
        let grinding_bits: Vec<usize> = self.levels.iter().map(|lv| lv.grinding_bits).collect();
        let fold_grinding_bits: Vec<usize> =
            self.levels.iter().map(|lv| lv.fold_grinding_bits).collect();
        let ood_samples: Vec<usize> = self.levels.iter().map(|lv| lv.ood_samples).collect();
        let prover = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
        };
        let verifier = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols,
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
        };
        Ok((prover, verifier))
    }
}

// ===================================================================
// Proof
// ===================================================================

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursiveProof {
    /// One row per query, each of `num_interleaved` F128 entries. Rows are
    /// emitted in **sorted** query-position order so they align with the
    /// merkle multi-proof.
    pub opened_rows: Vec<Vec<F128>>,
    /// Single octopus multi-proof shared across all queries at this level.
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalProof {
    /// Remaining polynomial sent in clear at the last recursive step.
    pub yr: Vec<F128>,
    /// Same sorted-by-position convention as [`RecursiveProof`].
    pub opened_rows: Vec<Vec<F128>>,
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LigeritoProof {
    pub initial_root: Hash,
    pub initial_proof: RecursiveProof,
    pub recursive_roots: Vec<Hash>,
    pub recursive_proofs: Vec<RecursiveProof>,
    pub final_proof: FinalProof,
    pub sumcheck_transcript: Vec<SumcheckMessage>,
    /// Per-level PoW nonces (one entry per query phase). When all
    /// `grinding_bits` are 0 (the default config), each entry is just 0
    /// and the verifier's PoW check is a no-op. `#[serde(default)]` keeps
    /// older serialized proofs that pre-date this field readable.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
    /// Claimed multilinear OOD evaluations, flattened in transcript order
    /// (level 1's `ood_samples[1]` values, then level 2's, ...). Empty when
    /// the config takes no OOD samples (UDR profiles, legacy paths).
    #[serde(default)]
    pub ood_values: Vec<F128>,
    /// Fold-challenge PoW nonces, flattened in transcript order — one per
    /// fold challenge at every level with `fold_grinding_bits > 0`. Empty
    /// when no level fold-grinds.
    #[serde(default)]
    pub fold_grinding_nonces: Vec<u64>,
}

impl LigeritoProof {
    pub fn size_bytes(&self) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        let level_bytes = |p: &RecursiveProof| -> usize {
            p.opened_rows.iter().map(|r| r.len() * ELEM).sum::<usize>() + p.merkle_proof.len() * 32
        };
        let mut total = 32;
        total += self.recursive_roots.len() * 32;
        total += level_bytes(&self.initial_proof);
        for p in &self.recursive_proofs {
            total += level_bytes(p);
        }
        total += self.final_proof.yr.len() * ELEM
            + self
                .final_proof
                .opened_rows
                .iter()
                .map(|r| r.len() * ELEM)
                .sum::<usize>()
            + self.final_proof.merkle_proof.len() * 32;
        total += self.sumcheck_transcript.len() * 2 * ELEM;
        total += self.ood_values.len() * ELEM;
        total += (self.grinding_nonces.len() + self.fold_grinding_nonces.len()) * 8;
        total
    }

    /// Print a per-component breakdown of the proof size to stderr.
    pub fn print_size_breakdown(&self) {
        const ELEM: usize = core::mem::size_of::<F128>();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        let roots_b = 32 * (1 + self.recursive_roots.len());
        let init_opened: usize = self
            .initial_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let init_merkle: usize = self.initial_proof.merkle_proof.len() * 32;
        eprintln!(
            "  L0 (initial): opened={} ({}q × {}lanes × {}B)  merkle={}",
            kb(init_opened),
            self.initial_proof.opened_rows.len(),
            self.initial_proof
                .opened_rows
                .first()
                .map_or(0, |r| r.len()),
            ELEM,
            kb(init_merkle),
        );
        let mut total_opened = init_opened;
        let mut total_merkle = init_merkle;
        for (i, rp) in self.recursive_proofs.iter().enumerate() {
            let opened: usize = rp.opened_rows.iter().map(|r| r.len() * ELEM).sum();
            let merkle: usize = rp.merkle_proof.len() * 32;
            eprintln!(
                "  L{} (recursive): opened={} ({}q × {}lanes × {}B)  merkle={}",
                i + 1,
                kb(opened),
                rp.opened_rows.len(),
                rp.opened_rows.first().map_or(0, |r| r.len()),
                ELEM,
                kb(merkle),
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let final_opened: usize = self
            .final_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let final_merkle: usize = self.final_proof.merkle_proof.len() * 32;
        let yr_b = self.final_proof.yr.len() * ELEM;
        eprintln!(
            "  L{} (final):  opened={} ({}q × {}lanes × {}B)  merkle={}  yr={} ({}×{}B)",
            self.recursive_proofs.len() + 1,
            kb(final_opened),
            self.final_proof.opened_rows.len(),
            self.final_proof.opened_rows.first().map_or(0, |r| r.len()),
            ELEM,
            kb(final_merkle),
            kb(yr_b),
            self.final_proof.yr.len(),
            ELEM,
        );
        total_opened += final_opened;
        total_merkle += final_merkle;
        let tx_b = self.sumcheck_transcript.len() * 2 * ELEM;
        eprintln!(
            "  TOTALS: roots={}  opened={}  merkle={}  yr={}  transcript={} ({}×2×{}B)  GRAND={}",
            kb(roots_b),
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(tx_b),
            self.sumcheck_transcript.len(),
            ELEM,
            kb(self.size_bytes()),
        );
    }
}

// ===================================================================
// Multilinear helpers
// ===================================================================

/// Multilinear extension of `evals` at the boolean cube of dimension `n`,
/// LSB-first indexing: `eval(b_0, …, b_{n-1}) = evals[b_0 + 2·b_1 + …]`.
///
/// Partially evaluate at the first `k` variables (the LSB end): given
/// challenges `rs ∈ F^k`, returns the length-`2^{n-k}` table
/// `f(rs[0], …, rs[k-1], x_k, …, x_{n-1})`.
///
/// Matches Flock's [`build_eq_table`] LSB-first convention (and bolt-rs's
/// `partial_eval` Julia convention).
pub(crate) fn partial_eval_lsb(evals: &[F128], rs: &[F128]) -> Vec<F128> {
    let mut cur = evals.to_vec();
    for &r in rs {
        let one_plus_r = F128::ONE + r;
        let half = cur.len() / 2;
        // Pair (cur[2i], cur[2i+1]) collapses to cur[2i]·(1+r) + cur[2i+1]·r.
        // LSB-first ⇒ adjacent pairs are bit_0 = 0 vs 1.
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            next.push(cur[2 * i] * one_plus_r + cur[2 * i + 1] * r);
        }
        cur = next;
    }
    cur
}

/// Evaluate the multilinear extension of `evals` at `point` (LSB-first).
/// `point.len()` must equal `log2(evals.len())`. Test oracle for
/// `partial_eval_lsb` composition; not used in production paths.
#[cfg(test)]
pub(crate) fn eval_mle_lsb(evals: &[F128], point: &[F128]) -> F128 {
    let folded = partial_eval_lsb(evals, point);
    debug_assert_eq!(folded.len(), 1);
    folded[0]
}

// ===================================================================
// LCH novel-basis evaluations (ported from bolt-rs `fft.rs`)
// ===================================================================
//
// Same subspace-polynomial recurrence `s_{i+1}(x) = s_i(x)² + s_i(v_i)·s_i(x)`
// as Flock's `AdditiveNttF128`, but we expose the evaluation at an arbitrary
// point — which the NTT doesn't currently surface publicly. Standard basis only
// (v_i = 2^i, embedded as `F128::new(1 << i, 0)`).

#[inline]
fn next_s(s: F128, s_at_root: F128) -> F128 {
    s * s + s_at_root * s
}

/// `sks_vks[k] = s_k(v_k)` for `k = 0..=log_n`. Length `log_n + 1`.
/// Only depends on `log_n`, so callers cache.
pub(crate) fn eval_sk_at_vks(log_n: usize) -> Vec<F128> {
    let mut sks_vks = vec![F128::ZERO; log_n + 1];
    sks_vks[0] = F128::ONE;
    if log_n == 0 {
        return sks_vks;
    }
    let mut layer: Vec<F128> = (1..=log_n).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut cur_len = log_n;
    for i in 0..log_n {
        for j in 0..cur_len {
            let sk_at_vk = next_s(layer[j], sks_vks[i]);
            if j == 0 {
                sks_vks[i + 1] = sk_at_vk;
            } else {
                layer[j - 1] = sk_at_vk;
            }
        }
        cur_len -= 1;
    }
    sks_vks
}

/// Write into `basis` the **normalized** LCH novel-basis polynomials
/// `X̂_j(x) = Π_{k: bit_k(j)=1} Ŵ_k(x)` for `j ∈ [0, 2^log_n)`, each scaled by
/// `alpha`. `Ŵ_k = s_k / s_k(v_k)` is normalized to match Flock's NTT twiddles.
///
/// `sks_at_x` is a scratch buffer of length `≥ log_n`. `sks_vks` is from
/// [`eval_sk_at_vks`]; `inv_sks_vks[k] = sks_vks[k].inv()` precomputed once
/// across many queries.
fn evaluate_scaled_basis_inplace(
    sks_at_x: &mut [F128],
    basis: &mut [F128],
    sks_vks: &[F128],
    inv_sks_vks: &[F128],
    x: F128,
    alpha: F128,
) {
    let log_n = basis.len().trailing_zeros() as usize;
    debug_assert_eq!(basis.len(), 1 << log_n);
    debug_assert!(sks_at_x.len() >= log_n);
    debug_assert!(inv_sks_vks.len() > log_n);

    if log_n > 0 {
        sks_at_x[0] = x;
        for i in 1..log_n {
            sks_at_x[i] = next_s(sks_at_x[i - 1], sks_vks[i - 1]);
        }
        // Normalize: Ŵ_i(x) = s_i(x) / s_i(v_i)
        for i in 0..log_n {
            sks_at_x[i] *= inv_sks_vks[i];
        }
    }

    basis[0] = alpha;
    for k in 0..log_n {
        let s_at_x = sks_at_x[k];
        let current_len = 1 << k;
        for i in 0..current_len {
            basis[i + current_len] = s_at_x * basis[i];
        }
    }
}

// ===================================================================
// induce_sumcheck_poly — the per-level basis-poly builder.
// ===================================================================
//
// Given Q opened rows of the previous commitment at query positions and the
// post-partial-eval challenges `v_challenges`, builds:
//   basis_poly[j] = Σ_i  α^i · Ŵ_j(q_i_field)
//   enforced_sum  = Σ_i  α^i · ⟨row_i, eq(v_challenges, ·)⟩
//
// The verifier reconstructs both independently from public inputs and checks
// the sumcheck claim Σ_j f(j) · basis_poly[j] = enforced_sum at the residual.

/// Compute just the `enforced_sum` half of [`induce_sumcheck_poly`]:
///   `enforced_sum = Σ_i eq(α, i_bin) · ⟨opened_rows[i], eq(v_challenges, ·)⟩`
/// Cheap: O(num_queries × num_interleaved). Verifier needs this at level
/// intro time (before residual challenges are known).
pub(crate) fn induce_sumcheck_enforced_sum(
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> F128 {
    assert_eq!(opened_rows.len(), queries.len());
    let eq = build_eq_table(v_challenges);
    let n_queries = queries.len();
    let alpha_weights: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        build_eq_table(alpha).into_iter().take(n_queries).collect()
    };
    let mut sum = F128::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
        debug_assert_eq!(row.len(), eq.len());
        let dot: F128 = row
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        sum += alpha_weights[i] * dot;
    }
    sum
}

/// **Succinct** evaluator for the induced basis poly's MLE at residual points.
/// Replaces `induce_sumcheck_poly` + `partial_eval_lsb` in the verifier:
/// instead of materializing the dense `2^log_msg_cols` basis_poly, evaluates
/// its MLE directly using the closed-form identity:
///   `MLE(basis_poly)(p) = Σ_i α^i · Π_k (1 + p[k] · (1 + Ŵ_k(q_i)))`
/// where each `q_i` is the field embedding of `queries[i]`.
///
/// `ris_for_basis` is the fixed prefix of the residual point (the ris range
/// that would have been passed to `partial_eval_lsb(basis_poly, ris_for_basis)`).
/// Length must be `log_msg_cols - yr_log_n`. The function returns evaluations
/// at `2^yr_log_n` points: `ris_for_basis ++ y_bits` for `y ∈ [0, 2^yr_log_n)`.
///
/// Cost: O(num_queries × yr_log_n × 2^yr_log_n + num_queries × log_msg_cols),
/// vs the dense path's O(num_queries × log_msg_cols × 2^log_msg_cols). At m=30
/// L0 with 221 queries, log_msg_cols=17, yr_log_n=4: ~18k ops vs ~500M ops.
/// `⌈log₂ n⌉`. Number of bits needed to index `n` items. Used to size the
/// per-level `alpha` slice for the eq-tensor basis-induction combination.
#[inline]
pub(crate) fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

pub(crate) fn induce_sumcheck_evaluate_at_residual(
    log_msg_cols: usize,
    sks_vks: &[F128],
    queries: &[usize],
    alpha: &[F128],
    ris_for_basis: &[F128],
    yr_log_n: usize,
) -> Vec<F128> {
    use crate::lincheck::build_eq_table;
    use rayon::prelude::*;
    assert_eq!(ris_for_basis.len() + yr_log_n, log_msg_cols);
    let n_queries = queries.len();
    let yr_len = 1usize << yr_log_n;

    // Per-query weights are the eq-tensor coefficients `eq(α, i_bin)` for
    // `i ∈ {0,1}^{⌈log₂ n_queries⌉}` (LSB-first), padded with zeros for
    // indices ≥ n_queries. Replaces the legacy α^i Vandermonde scheme;
    // soundness bound goes from `Q/q` (univariate S-Z) to `⌈log₂ Q⌉/q`
    // (multilinear S-Z), matching the rest of the multilinear protocol.
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    let prefix_len = ris_for_basis.len();

    // Per-query precomputation: Ŵ_k(q) for all k, then split into prefix
    // product (fixed scalar) and suffix Ŵ values (varied per y).
    struct PerQuery {
        prefix_prod: F128,
        suffix_w: Vec<F128>, // length = yr_log_n
    }
    let compute_query = |&q: &usize| -> PerQuery {
        let q_field = F128::new(q as u64, 0);
        // Compute s_k(q_field) recursively, then normalize by 1/s_k(v_k).
        let mut sks_at_x = Vec::with_capacity(log_msg_cols.max(1));
        if log_msg_cols > 0 {
            sks_at_x.push(q_field);
            for k in 1..log_msg_cols {
                sks_at_x.push(next_s(sks_at_x[k - 1], sks_vks[k - 1]));
            }
            for k in 0..log_msg_cols {
                sks_at_x[k] *= inv_sks_vks[k];
            }
        }
        // Prefix product: Π_{k<prefix_len} (1 + ris[k] · (1 + Ŵ_k(q)))
        let mut prefix_prod = F128::ONE;
        for k in 0..prefix_len {
            prefix_prod *= F128::ONE + ris_for_basis[k] * (F128::ONE + sks_at_x[k]);
        }
        let suffix_w = if log_msg_cols > prefix_len {
            sks_at_x[prefix_len..].to_vec()
        } else {
            Vec::new()
        };
        PerQuery {
            prefix_prod,
            suffix_w,
        }
    };
    // This runs once per recursion level over tiny verify-sized inputs
    // (`queries` ≈ tens; `yr_len` ≤ 2^5 since the residual folds to ≤5 bits), so
    // a rayon dispatch per level costs more than the field work itself (measured
    // ~0.47 ms serial vs ~0.75 ms parallel for the whole residual eval at m=30).
    // Stay serial below the crossover — mirror of merkle.rs's `SERIAL_LEVEL_NODES`.
    const PAR_FLOOR: usize = 1024;
    let per_query: Vec<PerQuery> = if n_queries > PAR_FLOOR {
        queries.par_iter().map(compute_query).collect()
    } else {
        queries.iter().map(compute_query).collect()
    };

    // For each residual position y, accumulate the suffix product per query.
    let compute_y = |y: usize| -> F128 {
        let mut sum = F128::ZERO;
        for i in 0..n_queries {
            let pq = &per_query[i];
            let mut suffix_prod = F128::ONE;
            for j in 0..yr_log_n {
                let p_j = if (y >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                suffix_prod *= F128::ONE + p_j * (F128::ONE + pq.suffix_w[j]);
            }
            sum += alpha_pows[i] * pq.prefix_prod * suffix_prod;
        }
        sum
    };
    if yr_len > PAR_FLOOR {
        (0..yr_len).into_par_iter().map(compute_y).collect()
    } else {
        (0..yr_len).map(compute_y).collect()
    }
}

/// `queries` are **0-indexed** codeword positions. `q_field = F128::new(q, 0)`.
///
/// Parallel: each thread takes a chunk of queries, builds a partial basis_poly
/// accumulator + partial enforced_sum, then we reduce. The per-query work
/// (eq-dot + LCH novel-basis expansion) is independent of other queries.
pub(crate) fn induce_sumcheck_poly(
    log_msg_cols: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    use rayon::prelude::*;
    let n = 1usize << log_msg_cols;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);
    debug_assert_eq!(
        v_challenges.len(),
        opened_rows
            .first()
            .map(|r| r.len().trailing_zeros() as usize)
            .unwrap_or(0)
    );

    let eq = build_eq_table(v_challenges); // length 2^v_challenges.len() = num_interleaved

    // Per-query weights are the eq-tensor coefficients `eq(α, i_bin)` for
    // `i ∈ {0,1}^{⌈log₂ n_queries⌉}` (LSB-first), truncated to the first
    // `n_queries` indices. Replaces the legacy α^i Vandermonde scheme;
    // matches the multilinear S-Z structure used by the lane fold.
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    // Precompute inv_sks_vks once across all queries and threads.
    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    // Per-thread chunked accumulation: each thread accumulates a partial
    // basis_poly (length n) and a partial enforced_sum, then we reduce.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_queries + n_threads - 1) / n_threads.max(1);

    let partials: Vec<(Vec<F128>, F128)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_queries);
            if start >= end {
                return (vec![F128::ZERO; n], F128::ZERO);
            }
            let mut accum_basis = vec![F128::ZERO; n];
            // Per-thread scratch reused across this chunk's queries.
            let mut local_basis = vec![F128::ZERO; n];
            let mut sks_at_x = vec![F128::ZERO; log_msg_cols.max(1)];
            let mut local_sum = F128::ZERO;

            for i in start..end {
                let row = &opened_rows[i];
                let q = queries[i];
                let ap = alpha_pows[i];

                let dot: F128 = row
                    .iter()
                    .zip(eq.iter())
                    .map(|(&r, &e)| r * e)
                    .fold(F128::ZERO, |a, v| a + v);
                local_sum += dot * ap;

                let q_field = F128::new(q as u64, 0);
                evaluate_scaled_basis_inplace(
                    &mut sks_at_x,
                    &mut local_basis,
                    sks_vks,
                    &inv_sks_vks,
                    q_field,
                    ap,
                );
                for (acc, &v) in accum_basis.iter_mut().zip(local_basis.iter()) {
                    *acc += v;
                }
            }
            (accum_basis, local_sum)
        })
        .collect();

    // Reduce across threads.
    let mut basis_poly = vec![F128::ZERO; n];
    let mut enforced_sum = F128::ZERO;
    for (lb, ls) in partials {
        for (acc, &v) in basis_poly.iter_mut().zip(lb.iter()) {
            *acc += v;
        }
        enforced_sum += ls;
    }

    (basis_poly, enforced_sum)
}

/// Transposed forward additive NTT, `Fᵀ`, in place over `2^log_d` coefficients.
/// Forward butterfly is `M=[[1,t],[1,t+1]]`; transpose `Mᵀ=[[1,1],[t,t+1]]` is
/// `s=a+b; top=s; bot=t·s+b`, applied in **reverse** layer order. (Baseline:
/// one parallel sweep per layer.)
fn transpose_forward_ntt(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(data.len(), 1usize << log_d);
    debug_assert!(log_d <= ntt.log_domain_size());
    let n_threads = rayon::current_num_threads().max(1);
    for layer in (0..log_d).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let bsh = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let t = ntt.twiddle(layer, block);
                    let (top, bot) = chunk.split_at_mut(bsh);
                    for (a_ref, b_ref) in top.iter_mut().zip(bot.iter_mut()) {
                        let a = *a_ref;
                        let b = *b_ref;
                        let s = a + b;
                        *a_ref = s;
                        *b_ref = t * s + b;
                    }
                });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top, bot) = chunk.split_at_mut(bsh);
                top.par_iter_mut()
                    .zip(bot.par_iter_mut())
                    .for_each(|(a_ref, b_ref)| {
                        let a = *a_ref;
                        let b = *b_ref;
                        let s = a + b;
                        *a_ref = s;
                        *b_ref = t * s + b;
                    });
            }
        }
    }
}

/// `Fᵀ`-based fast path for [`induce_sumcheck_poly`]: scatter per-query weights
/// into the codeword domain, apply `Fᵀ`, keep the low `2^log_msg_cols` outputs.
/// Byte-identical output to [`induce_sumcheck_poly`].
pub(crate) fn induce_sumcheck_poly_via_ntt(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let n = 1usize << log_msg_cols;
    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);

    let eq = build_eq_table(v_challenges);
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let mut enforced_sum = F128::ZERO;
    for i in 0..n_queries {
        let dot: F128 = opened_rows[i]
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        enforced_sum += dot * alpha_pows[i];
    }

    let mut coeffs = if log_block == 0 {
        let mut c = vec![F128::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += alpha_pows[i];
        }
        c
    } else {
        let ntt = AdditiveNttF128::standard(log_block);
        transpose_forward_ntt_sparse(&ntt, queries, &alpha_pows, log_block)
    };
    coeffs.truncate(n);
    (coeffs, enforced_sum)
}

/// Cost-based dispatch between the dense [`induce_sumcheck_poly`] and the
/// sparse-NTT [`induce_sumcheck_poly_via_ntt`].
///
/// The dense path costs `O(n_queries · 2^log_msg_cols)`; the NTT path costs one
/// pass over the `2^(log_msg_cols+log_inv_rate)` codeword domain, `O(2^log_block
/// · log_block)`. The `2^log_msg_cols` factor cancels, so the NTT wins exactly
/// when there are enough queries to amortize the codeword pass against the rate
/// blow-up and depth:
///   `n_queries  >  C · 2^log_inv_rate · log_block`   (C≈4: the NTT is ~2×
/// costlier per op — memory-bound, multi-pass — plus margin so we only switch
/// when clearly ahead). In the recursive PCS this fires only at the top level
/// (large message domain, many queries); deeper levels stay dense.
///
/// Both paths are byte-identical (see `induce_sumcheck_poly_via_ntt_matches_dense`),
/// so a mis-dispatch only costs time. Tuned/validated at blake m=30.
pub(crate) fn induce_sumcheck_poly_auto(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let log_block = log_msg_cols + log_inv_rate;
    let use_ntt =
        log_msg_cols >= 12 && queries.len() > 4 * (1usize << log_inv_rate) * log_block.max(1);
    if use_ntt {
        induce_sumcheck_poly_via_ntt(
            log_msg_cols,
            log_inv_rate,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    } else {
        induce_sumcheck_poly(
            log_msg_cols,
            sks_vks,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    }
}

/// Sparse-prefix variant of [`transpose_forward_ntt`]: exploits that the input
/// has only `positions.len()` nonzeros and that the first `k` transpose steps
/// (forward layers `log_d-1 .. log_d-k`, pairing distances `1 .. 2^(k-1)`) mix
/// only **within** `2^k`-aligned windows. We process just the windows that
/// contain a nonzero (a dense `2^k` transpose each), densify, then run the
/// remaining steps as full dense sweeps. Output is identical to
/// `transpose_forward_ntt` applied to the scattered input.
fn transpose_forward_ntt_sparse(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
) -> Vec<F128> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let n = 1usize << log_d;
    // No prefix for small domains — just scatter + full dense transpose.
    let k = if log_d >= 12 { 8usize.min(log_d) } else { 0 };

    if k == 0 {
        let mut data = vec![F128::ZERO; n];
        for (&p, &v) in positions.iter().zip(values) {
            data[p] += v;
        }
        if log_d > 0 {
            transpose_forward_ntt(ntt, &mut data, log_d);
        }
        return data;
    }

    let wmask = (1usize << k) - 1;
    // Group nonzeros into 2^k windows.
    let mut windows: HashMap<usize, Vec<F128>> = HashMap::new();
    for (&p, &v) in positions.iter().zip(values) {
        let buf = windows
            .entry(p >> k)
            .or_insert_with(|| vec![F128::ZERO; 1 << k]);
        buf[p & wmask] += v;
    }

    // Steps s = 0..k-1 within each active window, in parallel (windows disjoint).
    let win_vec: Vec<(usize, Vec<F128>)> = windows.into_iter().collect();
    let processed: Vec<(usize, Vec<F128>)> = win_vec
        .into_par_iter()
        .map(|(w, mut buf)| {
            for s in 0..k {
                let layer = log_d - 1 - s;
                let bsh = 1usize << s; // pairing distance
                let block_size = bsh << 1;
                let nblocks = (1usize << k) / block_size;
                for jb in 0..nblocks {
                    // global block index = ((w<<k) + jb*block_size) >> (s+1).
                    let t = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
                    let base = jb * block_size;
                    for r in 0..bsh {
                        let a = buf[base + r];
                        let b = buf[base + r + bsh];
                        let sab = a + b;
                        buf[base + r] = sab;
                        buf[base + r + bsh] = t * sab + b;
                    }
                }
            }
            (w, buf)
        })
        .collect();

    // Densify (active windows only; the rest stay zero, which is the correct
    // post-step-(k-1) state for an all-zero window).
    let mut data = vec![F128::ZERO; n];
    for (w, buf) in processed {
        data[(w << k)..((w + 1) << k)].copy_from_slice(&buf);
    }

    // Remaining steps s = k..log_d-1 = forward layers (log_d-1-k) .. 0, dense.
    let n_threads = rayon::current_num_threads().max(1);
    for layer in (0..(log_d - k)).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let bsh = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let t = ntt.twiddle(layer, block);
                    let (top, bot) = chunk.split_at_mut(bsh);
                    for (a_ref, b_ref) in top.iter_mut().zip(bot.iter_mut()) {
                        let a = *a_ref;
                        let b = *b_ref;
                        let sab = a + b;
                        *a_ref = sab;
                        *b_ref = t * sab + b;
                    }
                });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top, bot) = chunk.split_at_mut(bsh);
                top.par_iter_mut()
                    .zip(bot.par_iter_mut())
                    .for_each(|(a_ref, b_ref)| {
                        let a = *a_ref;
                        let b = *b_ref;
                        let sab = a + b;
                        *a_ref = sab;
                        *b_ref = t * sab + b;
                    });
            }
        }
    }
    data
}

// ===================================================================
// ligero_commit
// ===================================================================

/// Codeword + Merkle tree for one Ligerito commitment level.
///
/// `mat` is row-major: `mat[pos * num_interleaved + lane]` for
/// `pos ∈ [0, block_len)`, `lane ∈ [0, num_interleaved)`. Each row
/// (one `pos` across all lanes) is one Merkle leaf.
pub(crate) struct LigeroWitness {
    pub mat: Vec<F128>,
    pub tree: Vec<Hash>,
    pub block_len: usize,
    pub num_interleaved: usize,
}

// Recycle the codeword matrix (128 MB for L1 at m=29) through the scratch
// pool when a level's witness is replaced/dropped.
impl Drop for LigeroWitness {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.mat));
    }
}

// SumcheckProver owns the two witness-sized polynomials of the open (the
// packed witness `f` and the γ-combined basis) — recycle both on drop.
impl Drop for SumcheckProver {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.f));
        crate::scratch::give_f128(std::mem::take(&mut self.combined_basis));
    }
}

impl LigeroWitness {
    #[inline]
    pub fn row(&self, pos: usize) -> &[F128] {
        let start = pos * self.num_interleaved;
        &self.mat[start..start + self.num_interleaved]
    }

    #[inline]
    pub fn root(&self) -> Hash {
        self.tree[self.tree.len() - 1]
    }
}

/// Reshape `poly` (length `num_interleaved · msg_cols`) into a
/// `block_len × num_interleaved` SoA matrix, RS-encode each lane via the
/// LCH additive NTT (non-systematic: pad message with zeros to `block_len`,
/// then forward-transform), and Merkle-commit the rows.
///
/// `poly` layout: **LSB-first lane index** — `poly[col * num_interleaved + lane]`.
/// The first `log_num_interleaved` LSB variables of the multilinear poly are the
/// lane indices, so `partial_eval_lsb(poly, lane_challenges)` produces the
/// next-level poly directly. This composes cleanly with sumcheck folds.
pub(crate) fn ligero_commit(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
) -> LigeroWitness {
    let msg_cols = 1usize << log_msg_cols;
    let num_interleaved = 1usize << log_num_interleaved;
    let block_len = msg_cols << log_inv_rate;
    let log_block_len = log_msg_cols + log_inv_rate;
    assert_eq!(poly.len(), num_interleaved * msg_cols);
    assert!(log_block_len <= ntt.log_domain_size());

    // LSB-lane layout: input matches the SoA layout `data[pos * num_interleaved + lane]`
    // directly. The first `log_inv_rate` NTT layers on the zero-padded
    // coefficients are pure copies, so fill the matrix with 2^log_inv_rate
    // replicas of `poly` (same write cost as copy + zero-fill) and start the
    // transform past those layers — see `pcs::commit::replicate_message_fill`.
    let codeword_len = block_len * num_interleaved;
    let mut mat = crate::scratch::take_f128(codeword_len);
    super::commit::replicate_message_fill(&mut mat, poly);

    // RS-encode every lane in one call (each lane is one independent NTT).
    ntt.forward_transform_interleaved_from_layer(&mut mat, num_interleaved, log_inv_rate);

    // Merkle over rows. One leaf = `num_interleaved` consecutive F128 = 16·num_interleaved bytes.
    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    let tree = merkle::merkle_tree(data_bytes, block_len);

    LigeroWitness {
        mat,
        tree,
        block_len,
        num_interleaved,
    }
}

// ===================================================================
// Stateful sumcheck — Flock (u_0, u_2) convention
// ===================================================================
//
// Per-round quadratic q(X) = u_0 + u_1·X + u_2·X² with the sumcheck constraint
//   q(0) + q(1) = T_r          (T_r = running sum-claim entering this round)
// Verifier derives u_1 = T_r + u_2 (char 2). Round eval at challenge r:
//   q(r) = u_0 + r·(T_r + u_2) + r²·u_2 = u_0 + r·T_r + (r + r²)·u_2
//
// Ligerito extends plain sumcheck with two ops at recursive-level boundaries:
//
//   introduce_new(b_new, h):
//     Prover commits to a new basis poly b_new with its own claimed sum h
//     (verifier-computable from the open-rows induce step). Sends (u_0, u_2)
//     for the inner product f·b_new at the current (already-folded) dim.
//
//   glue(α):
//     Combine the running round-quadratic with the introduced one as
//     running := running + α·to_glue. New sum-claim becomes T_r + α·h.

/// (u_0, u_2) per round — what the prover sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckMessage {
    pub u_0: F128,
    pub u_2: F128,
}

/// Round-quadratic in coefficient form `c + b·X + a·X²`. Used by the verifier
/// to track the running quadratic across fold / introduce_new / glue.
#[derive(Clone, Copy, Debug)]
struct RoundQuad {
    c: F128, // u_0
    b: F128, // u_1 (X coeff) — derived from T_r and u_2
    a: F128, // u_2 (X² coeff)
}

impl RoundQuad {
    #[inline]
    fn from_msg(msg: SumcheckMessage, t_r: F128) -> Self {
        Self {
            c: msg.u_0,
            b: t_r + msg.u_2,
            a: msg.u_2,
        }
    }
    #[inline]
    fn eval(&self, r: F128) -> F128 {
        self.c + r * self.b + r * r * self.a
    }
    #[inline]
    fn fold(p1: &Self, p2: &Self, alpha: F128) -> Self {
        Self {
            c: p1.c + alpha * p2.c,
            b: p1.b + alpha * p2.b,
            a: p1.a + alpha * p2.a,
        }
    }
}

/// Compute `(u_0, u_2)` for `u(X) = Σ_x f(X, x) · b(X, x)` where `X` is the
/// LSB variable. Parallel reduction across pair indices.
///
/// Uses a SINGLE combined basis poly. (Previously took `&[Vec<F128>]` and
/// summed at every pair index; collapsing to one basis happens at glue time.)
fn round_msg_lsb(f: &[F128], b: &[F128]) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 4096;
    let half = n / 2;
    if half < PAR_THRESHOLD {
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            u_0 += f0 * b0;
            u_2 += (f0 + f1) * (b0 + b1);
        }
        return SumcheckMessage { u_0, u_2 };
    }

    let (u_0, u_2) = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            (f0 * b0, (f0 + f1) * (b0 + b1))
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    SumcheckMessage { u_0, u_2 }
}

/// Fused round message + full inner product: returns `round_msg_lsb(f, b)`
/// alongside `y = Σ_x f(x)·b(x)`, computed in a single pass over `(f, b)`.
///
/// Used by OOD binding, where `b = eq_table(z)` and `y` is the claimed MLE
/// eval `f̂(z)`. Folding `f` against `z` separately (`mle_eval_inline`) then
/// re-reading `f` against `b` in `round_msg_lsb` costs two passes over the
/// 2^n witness; this collapses them into one (the phase is memory-bandwidth
/// bound, so a saved pass is a near-proportional win). The `u_0` term `f0·b0`
/// is shared between the message and the eval, so `y` costs one extra mul per
/// pair. Bit-identical to the unfused path: F128 sums are exact and order-
/// independent, so `y == mle_eval_inline(f, z)`.
fn round_msg_and_eval_lsb(f: &[F128], b: &[F128]) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    const PAR_THRESHOLD: usize = 4096;
    let half = n / 2;
    let term = |j: usize| -> (F128, F128, F128) {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let b0 = b[2 * j];
        let b1 = b[2 * j + 1];
        let e0 = f0 * b0;
        // (u_0 term, u_2 term, y term = f0·b0 + f1·b1).
        (e0, (f0 + f1) * (b0 + b1), e0 + f1 * b1)
    };
    if half < PAR_THRESHOLD {
        let (mut u_0, mut u_2, mut y) = (F128::ZERO, F128::ZERO, F128::ZERO);
        for j in 0..half {
            let (a0, a2, ay) = term(j);
            u_0 += a0;
            u_2 += a2;
            y += ay;
        }
        return (SumcheckMessage { u_0, u_2 }, y);
    }

    let (u_0, u_2, y) = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(term)
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ay), (b0, b2, by)| (a0 + b0, a2 + b2, ay + by),
        );
    (SumcheckMessage { u_0, u_2 }, y)
}

/// Partially evaluate `evals` at LSB variable = `r`, in place. Halves length.
/// Parallel for large arrays. Test oracle for the fused fold below; the
/// production path uses `fold_and_msg_lsb` instead.
#[cfg(test)]
fn partial_eval_lsb_one(evals: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let n = evals.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let one_plus_r = F128::ONE + r;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        for j in 0..half {
            let v0 = evals[2 * j];
            let v1 = evals[2 * j + 1];
            evals[j] = v0 * one_plus_r + v1 * r;
        }
        evals.truncate(half);
        return;
    }

    // Parallel: produce a fresh halved Vec then swap in. Doing it in-place with
    // par_iter on overlapping indices is dicey; allocate the halved output and
    // swap (cheap vs the fold itself).
    let folded: Vec<F128> = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| evals[2 * j] * one_plus_r + evals[2 * j + 1] * r)
        .collect();
    *evals = folded;
}

/// Fused fold + next-round message in a SINGLE parallel pass.
///
/// Replaces the three separate passes a sumcheck fold otherwise needs
/// (`partial_eval_lsb_one(f)` + `partial_eval_lsb_one(b)` + `round_msg_lsb`):
/// each chunk folds its slice of `f` and `b` at `r` (LSB variable) AND
/// accumulates that slice's `(u_0, u_2)` contribution to the message for the
/// *next* round — over the freshly-folded values, computed while they are
/// still in registers. One fork-join instead of three, and ~⅓ less memory
/// traffic (the folded arrays are not re-read to build the message).
///
/// Returns `(folded_f, folded_b, next_msg)` where `next_msg = round_msg_lsb
/// (folded_f, folded_b)`. Bit-identical to the unfused sequence.
fn fold_and_msg_lsb(f: &[F128], b: &[F128], r: F128) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    let one_plus_r = F128::ONE + r;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        let mut nf = Vec::with_capacity(half);
        let mut nb = Vec::with_capacity(half);
        for j in 0..half {
            nf.push(f[2 * j] * one_plus_r + f[2 * j + 1] * r);
            nb.push(b[2 * j] * one_plus_r + b[2 * j + 1] * r);
        }
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        let mut k = 0;
        while k + 1 < half {
            let f0 = nf[k];
            let f1 = nf[k + 1];
            let b0 = nb[k];
            let b1 = nb[k + 1];
            u_0 += f0 * b0;
            u_2 += (f0 + f1) * (b0 + b1);
            k += 2;
        }
        return (nf, nb, SumcheckMessage { u_0, u_2 });
    }

    // Parallel path: `half` is a power of two ≥ PAR_THRESHOLD and CHUNK is a
    // power of two, so every chunk has even length and starts at an even
    // global index — message pairs (2k, 2k+1) never straddle a chunk boundary.
    const CHUNK: usize = 2048;
    let mut nf = crate::alloc_uninit_f128_vec(half);
    let mut nb = crate::alloc_uninit_f128_vec(half);
    let (u_0, u_2) = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            let len = fc.len();
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            // Fold this slice, then pair up the just-folded values for the msg.
            for t in 0..len {
                let j = base + t;
                fc[t] = f[2 * j] * one_plus_r + f[2 * j + 1] * r;
                bc[t] = b[2 * j] * one_plus_r + b[2 * j + 1] * r;
            }
            let mut k = 0;
            while k + 1 < len {
                let f0 = fc[k];
                let f1 = fc[k + 1];
                let b0 = bc[k];
                let b1 = bc[k + 1];
                u0 += f0 * b0;
                u2 += (f0 + f1) * (b0 + b1);
                k += 2;
            }
            (u0, u2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    (nf, nb, SumcheckMessage { u_0, u_2 })
}

pub struct SumcheckProver {
    f: Vec<F128>,
    /// Single combined basis poly. After every `glue(β)`, the introduced
    /// `b_new` is folded into here as `combined_basis += β · b_new`. This
    /// keeps fold cost O(1 + 1) = (f + combined_basis) regardless of how
    /// many recursive intro/glue pairs have happened.
    combined_basis: Vec<F128>,
    t_r: F128,
    transcript: Vec<SumcheckMessage>,
    pending_glue: Option<(Vec<F128>, F128)>,
}

impl SumcheckProver {
    pub fn new(f: Vec<F128>, b1: Vec<F128>, h1: F128) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
        };
        let msg = round_msg_lsb(&inst.f, &inst.combined_basis);
        inst.transcript.push(msg);
        (inst, msg)
    }

    /// Like [`Self::new`] but skips the initial `round_msg_lsb` pass over
    /// `(f, b1)` because the caller already computed `(u_0, u_2)` while
    /// building `b1` (saves a 256 MB read pass at m=30 BLAKE3). Used by
    /// `recursive_prover_with_basis` to consume the round0 prime that
    /// `compute_combined_basis_and_target` produces for free.
    pub fn new_with_first_msg(
        f: Vec<F128>,
        b1: Vec<F128>,
        h1: F128,
        first_msg: SumcheckMessage,
    ) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f,
            combined_basis: b1,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        // Fused: fold f and combined_basis at r AND build the next-round
        // message in one parallel pass (was three passes). See
        // [`fold_and_msg_lsb`].
        let (nf, nb, msg) = fold_and_msg_lsb(&self.f, &self.combined_basis, r);
        self.f = nf;
        self.combined_basis = nb;
        self.transcript.push(msg);
        msg
    }

    /// Introduce a fresh basis poly with claimed sum `h_new`. Sends the
    /// (u_0, u_2) for `Σ_x f(x) · b_new(x)` at the current dim.
    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        assert_eq!(b_new.len(), self.f.len());
        let msg = round_msg_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        msg
    }

    /// Like [`Self::introduce_new`] but also returns the claimed sum
    /// `h_new = Σ_x f(x)·b_new(x)`, computed in the same pass as the round
    /// message. For OOD binding `b_new = eq_table(z)`, so `h_new` is the MLE
    /// eval `f̂(z)` — fusing it here removes the separate `mle_eval_inline`
    /// fold over `f`. Transcript-identical: the caller observes the returned
    /// `h_new` then `(u_0, u_2)`, exactly as the unfused path does.
    pub fn introduce_new_with_eval(&mut self, b_new: Vec<F128>) -> (SumcheckMessage, F128) {
        assert_eq!(b_new.len(), self.f.len());
        let (msg, h_new) = round_msg_and_eval_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        (msg, h_new)
    }

    /// Combine the introduced basis into `combined_basis` with separation α.
    /// `combined_basis[j] += α · b_new[j]` (pointwise), `T_r += α · h_new`.
    pub fn glue(&mut self, alpha: F128) {
        use rayon::prelude::*;
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        const PAR_THRESHOLD: usize = 4096;
        if self.combined_basis.len() < PAR_THRESHOLD {
            for (acc, &v) in self.combined_basis.iter_mut().zip(b_new.iter()) {
                *acc += alpha * v;
            }
        } else {
            self.combined_basis
                .par_iter_mut()
                .zip(b_new.par_iter())
                .with_min_len(PAR_THRESHOLD / 4)
                .for_each(|(acc, &v)| *acc += alpha * v);
        }
        self.t_r += alpha * h_new;
    }

    pub fn f(&self) -> &[F128] {
        &self.f
    }

    pub fn transcript(&self) -> &[SumcheckMessage] {
        &self.transcript
    }
}

// ===================================================================
// Prover / Verifier — stubs
// ===================================================================

/// Sample `count` distinct positions in `[0, block_len)` via the challenger.
/// Asserts `count <= block_len` — otherwise no number of samples could satisfy
/// the distinctness requirement (would infinite-loop).
fn sample_distinct_queries<Ch: Challenger>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
) -> Vec<usize> {
    assert!(
        count <= block_len,
        "sample_distinct_queries: count ({count}) > block_len ({block_len}) — config is too thin for this query count"
    );
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let v = challenger.sample_f128();
        let q = (v.lo as usize) % block_len;
        if seen.insert(q) {
            out.push(q);
        }
    }
    out.sort_unstable();
    out
}

/// Build a single octopus multi-proof for all `queries` against `tree`.
fn merkle_multi_proof_for(tree: &[Hash], block_len: usize, queries: &[usize]) -> Vec<Hash> {
    merkle::merkle_multi_proof(tree, block_len, queries)
}

/// Drive the recursive Ligerito prover to prove `poly(eval_point) = claimed_value`.
///
/// Protocol structure (unique-decoding regime, no OOD samples yet):
/// 1. Commit f⁰ = `poly`.
/// 2. Partial-eval at `eval_point[0..initial_k]` (LSB-first), commit f¹.
/// 3. Open f⁰ at random query positions, induce a basis poly from the openings.
/// 4. Start sumcheck on `Σ_x f¹(x) · eq(eval_point[initial_k..], x) = claimed_value`,
///    introduce the induced basis (α-batched), glue with a separation challenge.
/// 5. For each recursive level: do k_i sumcheck folds; if last, send the residual
///    yr in clear and open the previous commitment; else commit the folded f,
///    open the previous commitment, induce a fresh basis from these opens,
///    introduce + glue.
pub fn recursive_prover<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = std::env::var("LIGERITO_TRACE").is_ok();
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let mut t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(
        config.log_inv_rates.len(),
        r + 1,
        "log_inv_rates must have R+1 entries"
    );
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    // ---- Initial commit (wtns_0) ----
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
    let t = std::time::Instant::now();
    let wtns_0 = ligero_commit(poly, log_msg_cols_0, initial_k, log_inv_rate_0, &ntt_0);
    let t_l0 = t.elapsed();
    t_commits += t_l0;
    tlog!("  [ligerito]   L0 commit: {:.2?}", t_l0);
    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

/// Variant of [`recursive_prover`] that reuses an **externally-built L0 commit**
/// (the codeword + merkle tree). This is what Flock's `pcs::open_batch` will
/// call after `pcs::commit` has already built the same shape. Skips the
/// L0 commit cost (~17 ms at m=29 MT).
///
/// Caller responsibility: the external L0 data must match what `ligero_commit`
/// would produce at the same `(log_msg_cols_0 = log_n - initial_k, initial_k,
/// log_inv_rates[0])`. In practice this means using `PcsParams` with
/// `log_batch_size = config.initial_k` and `log_inv_rate = config.log_inv_rates[0]`.
pub fn recursive_prover_with_l0<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    l0_codeword: Vec<F128>,
    l0_tree: Vec<Hash>,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = std::env::var("LIGERITO_TRACE").is_ok();
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;

    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    let block_len = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved = 1usize << initial_k;
    let _ = r; // used implicitly via config in inner
    assert_eq!(
        l0_codeword.len(),
        block_len * num_interleaved,
        "external L0 codeword wrong size"
    );
    assert_eq!(
        l0_tree.len(),
        2 * block_len - 1,
        "external L0 tree wrong size"
    );

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    let wtns_0 = LigeroWitness {
        mat: l0_codeword,
        tree: l0_tree,
        block_len,
        num_interleaved,
    };
    tlog!("  [ligerito]   L0 commit: REUSED (skipped)");

    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

/// Drop-in replacement for `basefold::prove`: takes a generic basis poly +
/// target (typically the combined `Σ γ_k · eq(z_k, ·)` and target produced by
/// `ring_switch::prove_batched` for batched claims), plus an externally-built
/// L0 commitment (the existing `pcs::commit` output).
///
/// Differs from [`recursive_prover`] in the initial step: instead of partial-
/// evaluating at `z[0..initial_k]` (which doesn't make sense for a combined
/// basis with no single `z`), runs `initial_k` real sumcheck rounds folding
/// both `f` and `b` together with FS challenges. The folded f becomes wtns_1
/// and the rest of the protocol proceeds identically.
pub fn recursive_prover_with_basis<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        None,
        challenger,
    )
}

/// Variant of [`recursive_prover_with_basis`] that accepts the round-0 sumcheck
/// `(u_0, u_2)` pre-computed by the caller. Useful from
/// `pcs::compute_combined_basis_and_target` which produces these values as a
/// side effect while building `b_initial` — passing them in here lets
/// `SumcheckProver::new` skip the redundant 256 MB read pass over (f, b1).
#[allow(clippy::too_many_arguments)]
pub fn recursive_prover_with_basis_precomputed_round0<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn recursive_prover_with_basis_impl<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    first_msg: Option<SumcheckMessage>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(packed_witness.len(), 1usize << log_n);
    assert_eq!(b_initial.len(), 1usize << log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    assert_eq!(l0_codeword.len(), block_len_0 * num_interleaved_0);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);

    let trace = std::env::var("LIG_PROVE_TRACE").is_ok();
    let mut t_init_sumcheck = std::time::Duration::ZERO;
    let mut t_commits = std::time::Duration::ZERO;
    let mut t_opens = std::time::Duration::ZERO;
    let mut t_induce = std::time::Duration::ZERO;
    let mut t_sumcheck_folds = std::time::Duration::ZERO;
    let mut t_intro_glue = std::time::Duration::ZERO;
    let mut t_ood = std::time::Duration::ZERO;

    let t_total = std::time::Instant::now();

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);

    // L0 codeword + tree are borrowed (reused from upstream `pcs::commit`).
    // wtns_0 access reduces to: root (last tree node), row(q), block_len.
    let initial_root: Hash = l0_tree[l0_tree.len() - 1];
    let l0_block_len = block_len_0;
    let l0_num_interleaved = num_interleaved_0;
    let l0_row = |q: usize| -> &[F128] {
        let start = q * l0_num_interleaved;
        &l0_codeword[start..start + l0_num_interleaved]
    };
    challenger.observe_bytes(&initial_root);

    // L0 takes no explicit OOD samples: it is bound by the opening's own
    // evaluation claim (`target` at the post-commit random point behind
    // `b_initial`), which plays the OOD role with a union over the list
    // instead of over pairs. See `paper_ood_bits`.
    assert_eq!(
        config.ood_samples.first().copied().unwrap_or(0),
        0,
        "L0 must not take explicit OOD samples"
    );
    let mut ood_values: Vec<F128> = Vec::new();
    let mut fold_grinding_nonces: Vec<u64> = Vec::new();
    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };

    let _t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = match first_msg {
        Some(msg) => SumcheckProver::new_with_first_msg(packed_witness, b_initial, target, msg),
        None => SumcheckProver::new(packed_witness, b_initial, target),
    };
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge grinding: the L0 proximity-gap bad event lives on
        // each of these lane-fold challenges, so each one is individually
        // PoW-guarded (a cheating prover re-rolls a fold challenge by
        // varying the preceding sumcheck message; the grind prices every
        // such attempt). Tapered per round: round j folds a 2^{ℓ-j}-row word
        // whose MCA error carries the factor 2^{ℓ-1-j} (App. C.3 Lemma
        // `mca-commutes`), so it needs (fold_bits − j) bits — one fewer per
        // round than the worst (j=0) round `fold_grinding_bits` is sized for.
        // Derived from fold_grinding_bits + round index; not stored.
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            fold_grinding_nonces.push(challenger.grind_pow(bits));
        }
        let r = challenger.sample_f128();
        let msg = sc_prover.fold(r);
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        r_lane_fold.push(r);
    }
    if trace {
        t_init_sumcheck += _t.elapsed();
    }

    // Commit f^1 = folded packed witness as wtns_1.
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let _t = std::time::Instant::now();
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let f1 = sc_prover.f().to_vec();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
    );
    if trace {
        t_commits += _t.elapsed();
    }
    challenger.observe_bytes(&wtns_1.root());

    // OOD binding for the L1 commit: each sample evaluates f1's multilinear
    // extension at a random transcript point z ∈ F^{n1}, sends the claimed
    // value, and folds the claim `Σ_x f1(x)·eq(z,x) = y` into the running
    // sumcheck (introduce + glue). Binds the prover to a single codeword of
    // the interleaved list before any of L0's queries are drawn.
    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);
            // Build eq(z, ·) once and fuse the MLE eval `y = f̂1(z)` into the
            // introduce round message (single pass over f1 + eq_z), instead of
            // a separate `mle_eval_inline` fold.
            let eq_z = build_eq_table(&z);
            let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            sc_prover.glue(beta);
        }
        if trace {
            t_ood += _t.elapsed();
        }
    }

    // Query-phase PoW grinding for L0: each ground bit substitutes for
    // ~1/log₂(1/(1−γ)) queries at this level (the Slim profile grinds 16
    // bits here). Verifier mirror checks the nonce; both then proceed to
    // sample query positions. (The proximity-gap shortfall is covered
    // separately by the fold-challenge grinds above.)
    let pow_nonce_0 = challenger.grind_pow(config.grinding_bits[0] as u32);
    let mut grinding_nonces: Vec<u64> = vec![pow_nonce_0];

    // Open L0; lane-fold weights = r_lane_fold.
    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, l0_block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| l0_row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(l0_tree, l0_block_len, &queries_0);
    if trace {
        t_opens += _t.elapsed();
    }
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    // Induce basis_0 from wtns_0 opens. L0 dominates the induce phase, where the
    // sparse-prefix Fᵀ-NTT path wins; the dispatcher auto-selects it (deeper
    // levels stay dense).
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let _t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &opened_rows_0,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_induce += _t.elapsed();
    }

    // Introduce + glue basis_0.
    let _t = std::time::Instant::now();
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);
    if trace {
        t_intro_glue += _t.elapsed();
    }

    // Recursive levels — same as recursive_prover_inner from here.
    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let _t = std::time::Instant::now();
        for j in 0..k_i {
            // These folds fold level i+1's commitment — fold-challenge
            // grinding guards its proximity-gap term. Tapered per round:
            // round j needs (fold_bits − j) bits (see L0 loop).
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        if trace {
            t_sumcheck_folds += _t.elapsed();
        }

        if i == r - 1 {
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            // PoW grinding for the last level before sampling its queries.
            let nonce_last = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
            grinding_nonces.push(nonce_last);
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let _t = std::time::Instant::now();
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            if trace {
                t_opens += _t.elapsed();
            }
            if trace {
                let total = t_total.elapsed();
                eprintln!("[lig-prove] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  initial sumcheck (initial_k folds + SC build): {:.2} ms",
                    t_init_sumcheck.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  recursive commits (NTT + merkle):              {:.2} ms",
                    t_commits.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  opens (rows + multi-proof):                    {:.2} ms",
                    t_opens.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  induce_sumcheck_poly:                          {:.2} ms",
                    t_induce.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sumcheck recursive folds:                      {:.2} ms",
                    t_sumcheck_folds.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  introduce_new + glue:                          {:.2} ms",
                    t_intro_glue.as_secs_f64() * 1e3
                );
                if !ood_values.is_empty() {
                    eprintln!(
                        "  OOD samples ({}): MLE evals + glue:            {:.2} ms",
                        ood_values.len(),
                        t_ood.as_secs_f64() * 1e3
                    );
                }
            }
            return LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces,
            };
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(n_next >= log_num_interleaved_next);
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let _t = std::time::Instant::now();
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let f_evals = sc_prover.f().to_vec();
        let wtns_next = ligero_commit(
            &f_evals,
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
        );
        if trace {
            t_commits += _t.elapsed();
        }
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        // OOD binding for the L_{i+2} commit (same as the L1 block above).
        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                let eq_z = build_eq_table(&z);
                let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
                challenger.observe_f128(y);
                ood_values.push(y);
                challenger.observe_f128(intro.u_0);
                challenger.observe_f128(intro.u_2);
                let beta = challenger.sample_f128();
                sc_prover.glue(beta);
            }
            if trace {
                t_ood += _t.elapsed();
            }
        }

        // PoW grinding for this iteration's query phase.
        let nonce_i = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
        grinding_nonces.push(nonce_i);
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let _t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        if trace {
            t_opens += _t.elapsed();
        }
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i.clone(),
            merkle_proof: merkle_proof_i,
        });

        let sks_vks_i = eval_sk_at_vks(n_next);
        let _t = std::time::Instant::now();
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );
        if trace {
            t_induce += _t.elapsed();
        }

        let _t = std::time::Instant::now();
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);
        if trace {
            t_intro_glue += _t.elapsed();
        }

        wtns_prev = wtns_next;
    }

    unreachable!()
}

/// Succinct verifier for [`recursive_prover_with_basis`]: instead of accepting
/// a dense `b_initial: &[F128]` (which would be ~16 MB at m=29), accepts a
/// **closure** `eval_b` that evaluates `b_initial(point)` at any multilinear
/// point. The verifier calls `eval_b` only `yr.len()` times (at the residual)
/// — typically a few dozen times, not 2^L. Use this from
/// `pcs::verify_opening_batch_ligerito_mixed` where the closure is built from
/// `ring_switch::verify_succinct` outputs + PD claim points.
///
/// `log_n` is the original packed-witness log size (= b_initial's logical dim).
#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis_succinct<Ch, F>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    log_n: usize,
    target: F128,
    expected_initial_root: &Hash,
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    // Called ONCE at the residual check with the full ris and yr_log_n.
    // Returns 2^yr_log_n values: eval_b(ris ++ y_bits) for y ∈ [0, 2^yr_log_n).
    // This API allows callers to amortize prefix work across yr positions
    // (e.g. ring_switch::eval_rs_eq_prefix + finish_from_prefix).
    F: Fn(&[F128], usize) -> Vec<F128>,
{
    let trace = std::env::var("LIG_VERIFY_TRACE").is_ok();
    let mut t_merkle = std::time::Duration::ZERO;
    let mut t_sample_q = std::time::Duration::ZERO;
    let mut t_enforced = std::time::Duration::ZERO;
    let mut t_residual = std::time::Duration::ZERO;
    let mut t_evalb = std::time::Duration::ZERO;
    let t_start = std::time::Instant::now();

    let initial_k = config.initial_k;
    let r = config.recursive_steps;
    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    // OOD claims glued into the running sumcheck: each contributes
    // `beta · Π_b eq(z_b, r_b) · eq(z_tail, ·)` at the residual.
    struct OodCtx {
        z: Vec<F128>,
        ris_start: usize,
        beta: F128,
    }
    let mut ood_ctxs: Vec<OodCtx> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge PoW mirror (L0's lane folds), tapered per round to
        // (fold_bits − j) — see the prover's L0 loop.
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    // OOD binding mirror for the L1 commit: sample z, read the claimed
    // evaluation from the proof, and glue the claim into the running
    // sumcheck exactly like the prover.
    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_ctxs.push(OodCtx {
            z,
            ris_start: initial_k,
            beta,
        });
    }

    // PoW grinding check for L0's query phase. With grinding_bits[0]=0 this
    // is a no-op (still absorbs the 0 nonce so the FS state matches the
    // prover side).
    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let _t = std::time::Instant::now();
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    if trace {
        t_sample_q += _t.elapsed();
    }
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
    ) {
        return false;
    }
    if trace {
        t_merkle += _t.elapsed();
    }

    // Compute enforced_sum cheaply at intro time. The induced basis poly's
    // residual evaluations are deferred to the final check (succinct path —
    // see `induce_sumcheck_evaluate_at_residual`).
    let n1 = log_n - initial_k;
    let _t = std::time::Instant::now();
    let enforced_sum_0 = induce_sumcheck_enforced_sum(
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_enforced += _t.elapsed();
    }

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    // Per-level induced-basis evaluation context — small (no dense vec).
    struct LevelCtx {
        log_msg_cols: usize,
        queries: Vec<usize>,
        alpha: Vec<F128>, // ⌈log₂ Q⌉ field elements (eq-tensor combination)
        ris_start: usize,
        beta: F128,
    }
    let mut level_ctxs: Vec<LevelCtx> = vec![LevelCtx {
        log_msg_cols: n1,
        queries: queries_0.clone(),
        alpha: alpha_0,
        ris_start: initial_k,
        beta: beta_0,
    }];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            // Fold-challenge PoW mirror (level i+1's folds), tapered per round
            // to (fold_bits − j) — see the prover's L0 loop.
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            // PoW grinding check for last level's query phase.
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }
            // (last nonce — nonce_idx is not advanced past it)

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let _t = std::time::Instant::now();
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            // Basis-induction challenge for the LAST commitment. Sampled here —
            // after `yr` was observed (top of this branch) and the queries are
            // fixed — so a forged `yr` cannot be adapted to it. Mirrors `alpha_i`
            // at every non-final level (see ~line 3377).
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if trace {
                t_sample_q += _t.elapsed();
            }
            let _t = std::time::Instant::now();
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
            ) {
                return false;
            }
            if trace {
                t_merkle += _t.elapsed();
            }

            // Bind the LAST commitment to `yr`. Every non-final level folds its
            // opened rows into the running sumcheck via induce_sumcheck; the
            // final level used to only Merkle-check its opened rows, leaving `yr`
            // (the claimed final message) constrained by a single scalar equation
            // — so a malicious prover could solve for a `yr` that opens the
            // commitment to an arbitrary value. We add the same proximity tie as
            // the other levels: `enforced_sum_last` is the α-weighted lane-fold
            // of the (Merkle-bound) opened rows, batched into `t_r` with a fresh
            // `beta_last`; its induced basis is already at the residual dimension
            // (zero further folds), so it joins `combined` below via this
            // LevelCtx. With `alpha_last` drawn after `yr`, the batched check now
            // forces `yr` to agree with the committed codeword at every queried
            // column (multilinear Schwartz–Zippel), restoring binding.
            let enforced_sum_last = induce_sumcheck_enforced_sum(
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            level_ctxs.push(LevelCtx {
                log_msg_cols: n_current,
                queries: queries_last.clone(),
                alpha: alpha_last,
                ris_start: ris.len(),
                beta: beta_last,
            });

            // Succinct residual check: per-level induced basis evaluations
            // via closed-form (no dense materialization).
            let yr_len = yr.len();
            let yr_log_n = n_current;

            let _t = std::time::Instant::now();
            let induced_residuals: Vec<Vec<F128>> = level_ctxs
                .iter()
                .map(|ctx| {
                    let sks_vks = eval_sk_at_vks(ctx.log_msg_cols);
                    let ris_for_basis =
                        &ris[ctx.ris_start..ctx.ris_start + ctx.log_msg_cols - yr_log_n];
                    induce_sumcheck_evaluate_at_residual(
                        ctx.log_msg_cols,
                        &sks_vks,
                        &ctx.queries,
                        &ctx.alpha,
                        ris_for_basis,
                        yr_log_n,
                    )
                })
                .collect();
            if trace {
                t_residual += _t.elapsed();
            }
            for resid in &induced_residuals {
                if resid.len() != yr_len {
                    return false;
                }
            }

            // OOD bases: closed-form residual. An eq(z, ·) basis introduced
            // at dim |z| and folded by the subsequent challenges contributes
            // `beta · Π_b eq(z_b, r_b)` times the eq table on z's unfolded
            // tail (char-2 eq factor: 1 + a + b).
            let mut ood_residuals: Vec<Vec<F128>> = Vec::with_capacity(ood_ctxs.len());
            for ctx in &ood_ctxs {
                if ctx.z.len() < yr_log_n || ctx.ris_start + (ctx.z.len() - yr_log_n) > ris.len() {
                    return false;
                }
                let folded = ctx.z.len() - yr_log_n;
                let mut scalar = ctx.beta;
                for b in 0..folded {
                    scalar *= F128::ONE + ctx.z[b] + ris[ctx.ris_start + b];
                }
                let mut tail = build_eq_table(&ctx.z[folded..]);
                for v in tail.iter_mut() {
                    *v *= scalar;
                }
                ood_residuals.push(tail);
            }

            // Batch-evaluate b at all yr positions in one call so the
            // caller can amortize prefix work (e.g. ring_switch tensor prefix).
            let _te = std::time::Instant::now();
            let evb_vec = eval_b_residual(&ris, yr_log_n);
            if trace {
                t_evalb += _te.elapsed();
            }
            if evb_vec.len() != yr_len {
                return false;
            }
            let mut inner = F128::ZERO;
            let _t = std::time::Instant::now();
            for y in 0..yr_len {
                let mut combined_y = evb_vec[y];
                for (k, residual) in induced_residuals.iter().enumerate() {
                    combined_y += level_ctxs[k].beta * residual[y];
                }
                for resid in &ood_residuals {
                    combined_y += resid[y];
                }
                inner += yr[y] * combined_y;
            }
            if trace {
                t_residual += _t.elapsed();
            }
            if trace {
                let total = t_start.elapsed();
                eprintln!("[lig-verify] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  merkle multi-proofs:       {:.2} ms",
                    t_merkle.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sample_distinct_queries:   {:.2} ms",
                    t_sample_q.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  enforced_sum (eq+dot):     {:.2} ms",
                    t_enforced.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  residual basis eval:       {:.2} ms",
                    t_residual.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  eval_b (yr_len positions): {:.2} ms",
                    t_evalb.as_secs_f64() * 1e3
                );
            }
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        // OOD binding mirror for the L_{i+2} commit.
        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_ctxs.push(OodCtx {
                z,
                ris_start: ris.len(),
                beta,
            });
        }

        // PoW grinding check for this iteration's query phase.
        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let _t = std::time::Instant::now();
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        if trace {
            t_sample_q += _t.elapsed();
        }
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        let _t = std::time::Instant::now();
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
        ) {
            return false;
        }
        if trace {
            t_merkle += _t.elapsed();
        }

        let _t = std::time::Instant::now();
        let enforced_sum_i =
            induce_sumcheck_enforced_sum(&rp.opened_rows, &level_rs, &queries_i, &alpha_i);
        if trace {
            t_enforced += _t.elapsed();
        }

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        level_ctxs.push(LevelCtx {
            log_msg_cols: n_current,
            queries: queries_i.clone(),
            alpha: alpha_i,
            ris_start: ris.len(),
            beta: beta_i,
        });

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

/// Verifier for [`recursive_prover_with_basis`]. Caller supplies the basis
/// `b_initial` recomputed locally (typically from the combined claims) and
/// `target`. Also supplies the L0 root (from the upstream `Commitment`).
#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    b_initial: &[F128],
    target: F128,
    expected_initial_root: &Hash,
    challenger: &mut Ch,
) -> bool {
    let log_n = b_initial.len().trailing_zeros() as usize;
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if b_initial.len() != 1usize << log_n {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    // Replay sumcheck: start msg → initial_k folds.
    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    // OOD eq bases glued into the running sumcheck, accumulated as
    // (dense eq table, ris_start, beta) and added at the residual check.
    let mut ood_bases: Vec<(Vec<F128>, usize, F128)> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge PoW mirror (L0's lane folds), tapered per round to
        // (fold_bits − j) — see the prover's L0 loop.
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    // Observe wtns_1 root + open wtns_0.
    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    // OOD binding mirror for the L1 commit.
    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_bases.push((build_eq_table(&z), initial_k, beta));
    }

    // PoW grinding check (dense verifier mirror) — keeps the FS state in
    // lockstep with the prover even at grinding_bits = 0.
    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
    ) {
        return false;
    }

    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );

    // Intro + glue.
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    // Basis poly tracking for residual check.
    // b_initial is the "level-0 basis" — it gets partial-eval'd at all ris.
    // basis_0_induced is introduced at start (before any ris from level 0+) — partial-eval at the level-0+ ris.
    let mut basis_polys: Vec<Vec<F128>> = vec![b_initial.to_vec(), basis_0_induced];
    let mut basis_ris_starts: Vec<usize> = vec![0, initial_k];
    let mut basis_separations: Vec<F128> = vec![beta_0];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            // Fold-challenge PoW mirror (level i+1's folds), tapered per round
            // to (fold_bits − j) — see the prover's L0 loop.
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            // PoW grinding check for last level (dense verifier).
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }
            // (last nonce — nonce_idx is not advanced past it)

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            // Final-level basis-induction challenge — sampled after `yr` and the
            // queries are fixed. Same position as the succinct verifier
            // (recursive_verifier_with_basis_succinct), which verifies the same
            // proof, so both stay in lockstep.
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
            ) {
                return false;
            }

            // Bind the LAST commitment to `yr`: induce its opened rows into the
            // sumcheck exactly like every non-final level, batched with a fresh
            // `beta_last`. Without this the last commitment is only Merkle-checked
            // and `yr` is left unconstrained — a forged `yr` could open to any
            // value. (Dense mirror of the succinct verifier fix.)
            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            // Residual check.
            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * rr;
                }
            }
            // OOD eq bases contribute the same way (dense tables).
            for (basis, start, beta) in &ood_bases {
                let residual = partial_eval_lsb(basis, &ris[*start..]);
                if residual.len() != yr_len {
                    return false;
                }
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += *beta * rr;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        // OOD binding mirror for the L_{i+2} commit.
        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_bases.push((build_eq_table(&z), ris.len(), beta));
        }

        // PoW grinding check for this iteration (dense verifier mirror).
        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

/// Shared body — runs after wtns_0 is in hand (whether freshly built or
/// supplied externally).
#[allow(clippy::too_many_arguments)]
fn recursive_prover_inner<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    wtns_0: LigeroWitness,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
    t_total: std::time::Instant,
    mut t_commits: std::time::Duration,
    mut t_induce: std::time::Duration,
    mut t_sumcheck: std::time::Duration,
    mut t_opens: std::time::Duration,
    trace: bool,
) -> LigeritoProof {
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    // The legacy (non-basis) path predates OOD binding and fold grinding;
    // configs that use them must go through `recursive_prover_with_basis`.
    assert!(
        config.ood_samples.iter().all(|&s| s == 0)
            && config.fold_grinding_bits.iter().all(|&b| b == 0),
        "OOD samples / fold grinding require the with_basis prover path"
    );
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];

    let initial_root = wtns_0.root();
    challenger.observe_bytes(&initial_root);

    // ---- Partial-eval at z[0..initial_k] and commit f¹ (wtns_1) ----
    let v_challenges_0 = eval_point[..initial_k].to_vec();
    let f1 = partial_eval_lsb(poly, &v_challenges_0);
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1, "n1 < k_0");
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let t = std::time::Instant::now();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
    );
    let t_l1 = t.elapsed();
    t_commits += t_l1;
    tlog!("  [ligerito]   L1 commit: {:.2?}", t_l1);
    challenger.observe_bytes(&wtns_1.root());

    // ---- Queries + open wtns_0 ----
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, wtns_0.block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(&wtns_0.tree, wtns_0.block_len, &queries_0);
    t_opens += t.elapsed();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    // ---- Induce basis from wtns_0 opens ----
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &opened_rows_0,
        &v_challenges_0,
        &queries_0,
        &alpha_0,
    );
    t_induce += t.elapsed();

    // ---- Start sumcheck: f¹ · eq(z[initial_k..], ·) = claimed_value ----
    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    let t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = SumcheckProver::new(f1, eq_z_residual, claimed_value);
    t_sumcheck += t.elapsed();
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    // ---- Introduce induced basis + glue ----
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);

    // ---- Recursive levels ----
    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let t = std::time::Instant::now();
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        t_sumcheck += t.elapsed();

        if i == r - 1 {
            tlog!(
                "  [ligerito] commits: {:.2?}  induce: {:.2?}  sumcheck: {:.2?}  opens: {:.2?}  TOTAL: {:.2?}",
                t_commits,
                t_induce,
                t_sumcheck,
                t_opens,
                t_total.elapsed()
            );
            // Last iter: send residual yr + open wtns_prev.
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            // wtns_prev's rate (= log_inv_rates[i+1] for wtns_{i+1}).
            let num_queries_last = udr_queries(config.log_inv_rates[i + 1]);
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            return LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                grinding_nonces: Vec::new(), // legacy recursive_prover_inner: no grinding plumbed
                ood_values: Vec::new(),
                fold_grinding_nonces: Vec::new(),
            };
        }

        // Non-last: commit the folded poly → wtns_next.
        // wtns_next = wtns_{i+2}, uses log_inv_rates[i+2].
        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(
            n_next >= log_num_interleaved_next,
            "f.n ({n_next}) < k_{} ({log_num_interleaved_next})",
            i + 1
        );
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let f_evals = sc_prover.f().to_vec();
        let t = std::time::Instant::now();
        let wtns_next = ligero_commit(
            &f_evals,
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
        );
        let t_li = t.elapsed();
        t_commits += t_li;
        tlog!("  [ligerito]   L{} commit: {:.2?}", i + 2, t_li);
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        // Open wtns_prev. wtns_prev = wtns_{i+1} uses log_inv_rates[i+1].
        let num_queries_i = udr_queries(config.log_inv_rates[i + 1]);
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        t_opens += t.elapsed();
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i.clone(),
            merkle_proof: merkle_proof_i,
        });

        // Induce fresh basis from these opens.
        let sks_vks_i = eval_sk_at_vks(n_next);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        // Introduce + glue.
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);

        wtns_prev = wtns_next;
    }

    unreachable!("recursive loop should return on last iter")
}

/// Verify all opened rows against one root via a single octopus multi-proof.
/// `queries` must be sorted ascending and aligned with `opened_rows`.
fn verify_level_opens(
    root: &Hash,
    block_len: usize,
    queries: &[usize],
    opened_rows: &[Vec<F128>],
    expected_num_interleaved: usize,
    multi_proof: &[Hash],
) -> bool {
    if queries.len() != opened_rows.len() {
        return false;
    }
    let mut leaf_hashes: Vec<Hash> = Vec::with_capacity(opened_rows.len());
    for row in opened_rows {
        if row.len() != expected_num_interleaved {
            return false;
        }
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                row.as_ptr() as *const u8,
                row.len() * core::mem::size_of::<F128>(),
            )
        };
        leaf_hashes.push(merkle::hash_leaf(bytes));
    }
    merkle::verify_merkle_multi_proof(root, block_len, queries, &leaf_hashes, multi_proof)
}

/// Verifier counterpart to [`recursive_prover`]. Supports arbitrary `R ≥ 1`.
pub fn recursive_verifier<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> bool {
    let log_n = eval_point.len();
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    // The legacy (non-basis) path predates OOD binding and fold grinding.
    if config.ood_samples.iter().any(|&s| s != 0)
        || config.fold_grinding_bits.iter().any(|&b| b != 0)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    // ---- Roots ----
    challenger.observe_bytes(&proof.initial_root);
    if proof.recursive_roots.len() != r {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    // ---- Open wtns_0 + α₀ ----
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));

    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
    ) {
        return false;
    }

    // ---- Induce basis_0 from wtns_0 opens ----
    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &proof.initial_proof.opened_rows,
        &eval_point[..initial_k],
        &queries_0,
        &alpha_0,
    );

    // ---- Set up running sumcheck state ----
    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    // basis_polys[k] are stored at the dim they were introduced. ris_starts[k] is
    // the index in `ris` at the time basis_polys[k] was introduced.
    let mut basis_polys: Vec<Vec<F128>> = vec![eq_z_residual];
    let mut basis_ris_starts: Vec<usize> = vec![0];
    let mut basis_separations: Vec<F128> = Vec::new(); // separation for basis_polys[k+1]
    let mut ris: Vec<F128> = Vec::new();
    let mut t_r = claimed_value;
    let mut tx_idx = 0usize;

    // ---- Start message ----
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    // ---- Intro basis_0 + glue β₀ ----
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;
    basis_polys.push(basis_0_induced);
    basis_ris_starts.push(0);
    basis_separations.push(beta_0);

    // ---- Recursive iterations ----
    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1]; // wtns_1's rate
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            // Last iter: read yr + open prev_root.
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = udr_queries(prev_log_inv_rate);
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            // Final-level basis-induction challenge (after yr + queries fixed).
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
            ) {
                return false;
            }

            // Bind the LAST commitment to `yr`: induce its opened rows into the
            // sumcheck like every non-final level (without this `yr` is
            // unconstrained and a forged `yr` opens to any value).
            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            // ---- Final residual check ----
            // Each basis_polys[k] is partially-evaluated at ris[ris_starts[k]..].
            // basis_polys[0] has separation 1, basis_polys[k+1] has separation basis_separations[k].
            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &r) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * r;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        // Non-last: read next root, sample queries on prev_root, induce basis, intro + glue.
        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = udr_queries(prev_log_inv_rate);
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));

        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        // Intro + glue
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        // Update prev for next iteration: prev_root = root_next, dims = next commit's dims.
        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!("loop should return at i = r - 1")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked example: `LigeritoSecurityConfig` for BLAKE3 m=29 at rate 1/2.
    /// Paper-compatible m=29 fast example, mechanically derived in the
    /// unique-decoding regime (Theorem 1.4, ε* = 10⁻³) targeting 100-bit
    /// security.
    fn blake3_m29_udr_example() -> LigeritoSecurityConfig {
        LigeritoSecurityConfig::derive_paper_compatible(29, 1, 100).expect("derive m29 fast")
    }

    /// Both embedded TOMLs (m29_fast at rate 1/2 and m29_slim at rate 1/4)
    /// parse, validate, and produce ProverConfig/VerifierConfig agreeing
    /// with the corresponding `default_config(22, 6, rate)` shape.
    #[test]
    fn ligerito_security_config_m29_toml_loads() {
        let toml_str = include_str!("../../configs/ligerito/m29_fast.toml");
        let cfg = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_fast.toml must parse and validate");
        assert_eq!(cfg.m, 29);
        assert_eq!(cfg.log_n, 22);
        assert_eq!(cfg.initial_k, 6);
        assert_eq!(cfg.hash, "sha256");
        assert_eq!(cfg.levels.len(), 5);
        // Fast = JohnsonOod profile: 218 L0 queries per-round at 100 bits (no
        // list union bound — single-codeword binding via the opening claim /
        // OOD samples), proximity-gap shortfall covered by fold-challenge grinding.
        assert_eq!(cfg.levels[0].regime, SoundnessRegime::JohnsonOod);
        assert_eq!(cfg.levels[0].queries, 218);
        assert_eq!(cfg.levels[0].grinding_bits, 0);
        assert!(cfg.levels[0].fold_grinding_bits > 0);
        assert_eq!(cfg.levels[0].ood_samples, 0); // L0: bound by eval claim
        assert!(cfg.levels[1].ood_samples >= 1);
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.queries[0], 218);

        // Slim mode: rates start at 1/4.
        let toml_str = include_str!("../../configs/ligerito/m29_slim.toml");
        let cfg_slim = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_slim.toml must parse and validate");
        assert_eq!(cfg_slim.levels[0].log_inv_rate, 2);
        // Slim = JohnsonOod at rate 1/4 with 16-bit query grinding.
        assert_eq!(cfg_slim.levels[0].queries, 90);
        assert_eq!(cfg_slim.levels[0].grinding_bits, 16);
        let (pv_slim, _vc_slim) = cfg_slim.to_prover_verifier_configs().unwrap();
        let default_slim = default_config(22, 6, 2).unwrap();
        assert_eq!(pv_slim.log_inv_rates, default_slim.log_inv_rates);
        assert_eq!(pv_slim.recursive_ks, default_slim.recursive_ks);
    }

    /// Helper: re-emit all the embedded TOMLs from `derive_paper_compatible`.
    /// Writes to stdout (via eprintln) so the user can `>` redirect to disk.
    /// Run with:
    ///   cargo test --release --lib regen_embedded_tomls -- --ignored --nocapture
    #[test]
    #[ignore]
    fn regen_embedded_tomls() {
        for m in [22usize, 29, 32] {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m{m}_{}: {e}", profile.as_str()));
                let toml = cfg.to_toml_string().expect("serialize");
                eprintln!(
                    "\n# ====== configs/ligerito/m{m}_{}.toml ======",
                    profile.as_str()
                );
                eprintln!("{toml}");
            }
        }
    }

    /// `validate()` rejects a config whose declared `expected_eps_pg_bits`
    /// disagrees with what Theorem 1.5 predicts for the level's
    /// `(eta, log_inv_rate, log_msg_cols)`. Enforces that the per-level
    /// diagnostics weren't hand-waved.
    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_pg() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_pg_bits = 50.0; // very wrong
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    /// Same enforcement on the query side.
    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_query() {
        let mut cfg = blake3_m29_udr_example();
        // Bump query bits by 5 — far outside tolerance.
        cfg.levels[0].expected_eps_query_bits += 5.0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    /// All 6 embedded configs validate strictly (i.e. each is paper-compat
    /// AND satisfies the security target).
    #[test]
    fn ligerito_all_embedded_configs_validate() {
        for &(key, toml) in EMBEDDED_CONFIGS {
            LigeritoSecurityConfig::from_toml_str(toml).unwrap_or_else(|e| {
                panic!(
                    "embedded config m={} profile={} invalid: {e}",
                    key.0,
                    key.1.as_str()
                )
            });
        }
    }

    /// `derive_paper_compatible` produces a config that validates for every
    /// `(m, log_inv_rate)` combination we ship.
    #[test]
    fn ligerito_derive_paper_compatible_for_all_embedded() {
        let pairs: &[(usize, usize)] = &[(22, 1), (28, 1), (29, 1), (29, 2), (30, 1), (30, 2)];
        for &(m, r) in pairs {
            let cfg = LigeritoSecurityConfig::derive_paper_compatible(m, r, 100)
                .unwrap_or_else(|e| panic!("derive m={m} r={r}: {e}"));
            cfg.validate()
                .unwrap_or_else(|e| panic!("derived m={m} r={r} fails validate: {e}"));
        }
        for m in 22..=35usize {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m={m} {}: {e}", profile.as_str()));
                cfg.validate().unwrap_or_else(|e| {
                    panic!("derived m={m} {} fails validate: {e}", profile.as_str())
                });
            }
        }
    }

    /// `prover_config_for` is **strict** — only known `(m, log_inv_rate)`
    /// pairs load. Unknown pairs return an `Err` so production callers can't
    /// silently fall back to unaudited parameters.
    #[test]
    fn ligerito_prover_config_for_lookup() {
        // m=29 fast: known → loads from TOML.
        let pv = prover_config_for(22, 6, LigeritoProfile::Fast).expect("m29 fast must load");
        assert_eq!(pv.queries[0], 218);
        assert_eq!(pv.fold_grinding_bits[0], 16);

        // m=29 slim: known → loads from TOML.
        let pv = prover_config_for(22, 6, LigeritoProfile::Slim).expect("m29 slim must load");
        assert_eq!(pv.queries[0], 90);
        assert_eq!(pv.grinding_bits[0], 16);

        // m=29 secure: known → loads from TOML (UDR, 120-bit).
        let pv = prover_config_for(22, 6, LigeritoProfile::Secure).expect("m29 secure must load");
        assert!(pv.queries[0] > 280);
        assert_eq!(pv.ood_samples.iter().sum::<usize>(), 0);

        // m=36 (unknown — above the registered 22..=35 range): errors,
        // no silent fallback.
        let err = prover_config_for(29, 6, LigeritoProfile::Fast).unwrap_err();
        assert!(
            err.contains("no security config registered"),
            "unexpected error: {err}"
        );
    }

    /// TOML round-trip via `to_toml_string` ↔ `from_toml_str` preserves
    /// the config exactly (modulo validated invariants).
    #[test]
    fn ligerito_security_config_toml_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let s = cfg.to_toml_string().expect("serialize");
        let back = LigeritoSecurityConfig::from_toml_str(&s).expect("deserialize");
        assert_eq!(back.levels.len(), cfg.levels.len());
        assert_eq!(back.levels[0].queries, cfg.levels[0].queries);
        assert_eq!(back.levels[0].grinding_bits, cfg.levels[0].grinding_bits);
        assert_eq!(back.final_block.yr_log_n, cfg.final_block.yr_log_n);
    }

    /// Schema validates the worked example end to end.
    #[test]
    fn ligerito_security_config_validates() {
        let cfg = blake3_m29_udr_example();
        cfg.validate()
            .unwrap_or_else(|e| panic!("validate failed: {e}"));
    }

    /// Lowering a level's expected_eps_query_bits below the required
    /// (target − grinding) is caught by validation.
    #[test]
    fn ligerito_security_config_rejects_insufficient_queries() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_query_bits = 50.0; // < target 100 (grinding 0)
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("expected_eps_query_bits"), "err = {err}");
    }

    /// UDR regime must not carry an `eta` value.
    #[test]
    fn ligerito_security_config_rejects_udr_with_eta() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].eta = Some(0.02); // eta is Johnson-only — should fail
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("udr") && err.contains("eta"), "err = {err}");
    }

    /// UDR regime requires `proximity_loss` to be set, not `eta`.
    #[test]
    fn ligerito_security_config_rejects_udr_without_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].proximity_loss = None; // missing!
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("udr") && err.contains("proximity_loss"),
            "err = {err}"
        );
    }

    /// `proximity_loss` is only valid for the UDR regime.
    #[test]
    fn ligerito_security_config_rejects_johnson_with_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        // JohnsonOod regime with proximity_loss set — should fail.
        cfg.levels[0].regime = SoundnessRegime::JohnsonOod;
        cfg.levels[0].eta = Some(0.02);
        cfg.levels[0].proximity_loss = Some(0.01);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("proximity_loss") && err.contains("udr"),
            "err = {err}"
        );
    }

    /// End-to-end: a hand-built UDR-regime level validates against the
    /// paper's Thm `ca-udr` bound (a = γ·n + 1) and the per-query/UDR formula.
    #[test]
    fn ligerito_security_config_udr_regime_validates() {
        let mut cfg = blake3_m29_udr_example();
        // Convert L0 to UDR at the maximal radius γ = δ/2 − 3/(δ·n) − ε*
        // (ε* = 0 → top of C.3's valid range). δ = 1 − ρ; per-query soundness
        // is log₂(1/(1−γ)) and Q is sized so Q·per_q ≥ 100 bits.
        let eps_star = 0.0f64;
        let rho = 0.5f64;
        let delta = 1.0 - rho;
        let n = ((cfg.levels[0].log_msg_cols + cfg.levels[0].log_inv_rate) as f64).exp2();
        let gamma = delta / 2.0 - 3.0 / (delta * n) - eps_star;
        let per_q = (1.0 / (1.0 - gamma)).log2();
        let queries = (100.0 / per_q).ceil() as usize;
        // a = γ·n + 1; ε_pg = 128 − log₂ a with NO row-union penalty in the
        // unique-decoding regime (list size 1; Diamond and Gruen). Any
        // shortfall below the 100-bit target is covered by fold-grinding.
        let log_a_base = (gamma * n + 1.0).log2();
        let eps_pg = 128.0 - log_a_base;
        cfg.levels[0].regime = SoundnessRegime::Udr;
        cfg.levels[0].eta = None;
        cfg.levels[0].proximity_loss = Some(eps_star);
        cfg.levels[0].queries = queries;
        cfg.levels[0].grinding_bits = 0;
        cfg.levels[0].fold_grinding_bits = (100.0 - eps_pg).ceil().max(0.0) as usize;
        cfg.levels[0].expected_eps_pg_bits = (eps_pg * 10.0).round() / 10.0;
        cfg.levels[0].expected_eps_query_bits = ((queries as f64 * per_q) * 10.0).round() / 10.0;
        cfg.validate()
            .unwrap_or_else(|e| panic!("UDR config failed to validate: {e}"));
    }

    /// Schema round-trips cleanly through serde JSON. (TOML would work too
    /// once we add a toml dep.)
    #[test]
    fn ligerito_security_config_serde_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let back: LigeritoSecurityConfig = serde_json::from_str(&json).expect("deserialize");
        back.validate().expect("roundtripped config validates");
        assert_eq!(back.levels.len(), cfg.levels.len());
        // rate 1/2, 100-bit target, full UD radius γ = δ/2 (ε* = 0):
        // per-query = log₂(1/(1−1/4)) ≈ 0.415 b/q → ⌈100/0.415⌉ = 241.
        assert_eq!(back.levels[0].queries, 241);
        assert_eq!(back.levels[0].grinding_bits, 0);
    }

    /// End-to-end: a security config with **non-zero grinding** at L0 drives
    /// an actual recursive_prover_with_basis → recursive_verifier_with_basis
    /// roundtrip. Confirms the PoW step is plumbed into the FS transcript
    /// on both sides (without grinding the proof would either be rejected
    /// or the FS state would diverge between prover and verifier).
    #[test]
    fn ligerito_security_config_drives_roundtrip_with_grinding() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x6817_D146);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        // Hand-set queries + grinding (small but non-zero c so we exercise
        // the SHA256 PoW search without blowing up test time).
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![6usize, 0]; // L0 grinds 6 bits, L1 doesn't
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, log_inv_rate, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );
        assert_eq!(proof.grinding_nonces.len(), 2, "one nonce per level");

        let v_cfg = VerifierConfig {
            log_inv_rates,
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            ok,
            "verifier should accept proof with valid grinding nonces"
        );

        // Tampering with the nonce flips the PoW check.
        let mut bad_proof = proof.clone();
        bad_proof.grinding_nonces[0] = bad_proof.grinding_nonces[0].wrapping_add(1);
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &bad_proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            !ok,
            "verifier must reject proof with tampered grinding nonce"
        );
    }

    /// The security config produces ProverConfig/VerifierConfig matching the
    /// existing `default_config(log_n=22, log_batch_size=6, log_inv_rate=1)`
    /// in shape (rates + recursive_ks + initial_k all agree).
    #[test]
    fn ligerito_security_config_matches_default_config() {
        let cfg = blake3_m29_udr_example();
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.initial_k, default.initial_k);
    }

    /// Single-lane RS encoding round-trips through inv-NTT: forward-transforming
    /// the zero-padded message and then inverse-transforming should give back the
    /// padded message.
    /// `partial_eval_lsb` followed by `eval_mle_lsb` on the residual equals
    /// `eval_mle_lsb` on the full point — i.e. partial evaluation is
    /// consistent with full evaluation under the same LSB-first convention.
    #[test]
    fn partial_eval_then_eval_equals_full_eval() {
        let n = 6;
        let len = 1usize << n;
        let evals: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xDEAD_BEEF_CAFE_BABE),
                    0xA5A5 ^ i as u64,
                )
            })
            .collect();
        let point: Vec<F128> = (0..n)
            .map(|i| F128::new(0x1111 * (i as u64 + 1), 0x2222 * (i as u64 + 1)))
            .collect();

        let full = eval_mle_lsb(&evals, &point);
        // Split the point into a (k, n-k) partial/residual prefix.
        let k = 3;
        let (lo, hi) = point.split_at(k);
        let residual = partial_eval_lsb(&evals, lo);
        assert_eq!(residual.len(), 1usize << (n - k));
        let after = eval_mle_lsb(&residual, hi);
        assert_eq!(full, after);

        // Sanity: build_eq_table evaluated at `point` and dot-producted
        // with `evals` should also equal `full` (LSB-first eq table).
        let eq = build_eq_table(&point);
        let dot = evals
            .iter()
            .zip(eq.iter())
            .map(|(&e, &q)| e * q)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(dot, full);
    }

    /// End-to-end sumcheck on a single basis poly: prove `Σ_x f(x)·b(x) = h`.
    /// Stops one round early (yr length 2 sent in clear, à la Ligerito).
    /// Verifier replays each round message, checks `q(0)+q(1)=T_r`, applies
    /// the challenge, and confirms the residual inner product matches.
    #[test]
    fn stateful_sumcheck_single_basis_roundtrip() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let f: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0),
                    0x55AA ^ i as u64,
                )
            })
            .collect();
        let b: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xFEDC_BA98_7654_3210),
                    0xAA55 ^ i as u64,
                )
            })
            .collect();
        let h: F128 = f
            .iter()
            .zip(b.iter())
            .map(|(&fi, &bi)| fi * bi)
            .fold(F128::ZERO, |a, v| a + v);

        // Prover: 1 start message + (n-1) folds, leaving a length-2 residual.
        let (mut prover, _first) = SumcheckProver::new(f.clone(), b.clone(), h);
        let mut ch = crate::challenger::RandomChallenger::new(0xC0FFEE);
        let mut ris: Vec<F128> = Vec::new();
        for _ in 0..(n - 1) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        assert_eq!(prover.f().len(), 2);
        assert_eq!(prover.combined_basis.len(), 2);

        // Verifier replay: n messages (start + n-1 folds), n-1 prover-folds challenges
        // (r_0..r_{n-2}) already in ris, plus one new r_last for the final residual.
        let msgs = prover.transcript().to_vec();
        assert_eq!(msgs.len(), n);
        let r_last = ch.sample_f128();
        let mut t_r = h;
        for (i, msg) in msgs.iter().enumerate() {
            let quad = RoundQuad::from_msg(*msg, t_r);
            assert_eq!(
                quad.eval(F128::ZERO) + quad.eval(F128::ONE),
                t_r,
                "round {i}: q(0)+q(1) != T_r"
            );
            let r_i = if i < n - 1 { ris[i] } else { r_last };
            t_r = quad.eval(r_i);
        }
        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        let b_resid = prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(f_resid * b_resid, t_r, "residual inner product != t_r");
    }

    /// Multi-basis sumcheck: introduce_new + glue mid-protocol. Verifier replays.
    #[test]
    fn stateful_sumcheck_introduce_glue() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let mk = |seed: u64| -> Vec<F128> {
            (0..len)
                .map(|i| F128::new(seed.wrapping_mul(i as u64 + 1), seed ^ (i as u64) << 7))
                .collect()
        };
        let f = mk(0xC1);
        let b1 = mk(0xB1);
        let b2 = mk(0xB2);
        let h1: F128 = f
            .iter()
            .zip(b1.iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);

        let (mut prover, _first) = SumcheckProver::new(f.clone(), b1.clone(), h1);
        let mut ch = crate::challenger::RandomChallenger::new(0xBEEF);

        // Fold once before introducing b2 (must fold at the same dim as the introduced poly).
        let r0 = ch.sample_f128();
        prover.fold(r0);
        // Partial-eval b2 too so it matches the prover's current f dim.
        let mut b2_folded = b2.clone();
        partial_eval_lsb_one(&mut b2_folded, r0);
        // The h for b2 at the folded dim is Σ b2_folded · f_folded — but the verifier
        // also gets to recompute this from the same shared inputs. For the test we
        // pass it explicitly.
        let h2_folded: F128 = b2_folded
            .iter()
            .zip(prover.f().iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);
        prover.introduce_new(b2_folded.clone(), h2_folded);
        let alpha = ch.sample_f128();
        prover.glue(alpha);

        // Continue folding to length 2 residual: n total fold-vars used, but
        // we've already used 1 (r0). One more r_last is the verifier's final.
        let mut ris = vec![r0];
        for _ in 0..(n - 2) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        let r_last = ch.sample_f128();
        ris.push(r_last);
        assert_eq!(prover.f().len(), 2);

        // Verifier replays: 1 start, 1 fold, 1 introduce_new (no T_r update), 1 glue
        // (combine running quad with introduced, update T_r), then (n-2) folds.
        let msgs = prover.transcript().to_vec();
        // start (idx 0) + fold(r0) → idx 1 + introduce_new → idx 2 + later folds
        // Note: glue doesn't add a transcript entry; it just combines internal state.
        assert_eq!(msgs.len(), 1 + 1 + 1 + (n - 2));

        let mut t_r = h1;
        // start
        let q0 = RoundQuad::from_msg(msgs[0], t_r);
        assert_eq!(q0.eval(F128::ZERO) + q0.eval(F128::ONE), t_r);
        t_r = q0.eval(r0); // fold(r0)
        // fold msg (idx 1)
        let q1 = RoundQuad::from_msg(msgs[1], t_r);
        assert_eq!(q1.eval(F128::ZERO) + q1.eval(F128::ONE), t_r);
        // introduce_new msg (idx 2): claim is h2_folded, not T_r
        let q_intro = RoundQuad::from_msg(msgs[2], h2_folded);
        assert_eq!(
            q_intro.eval(F128::ZERO) + q_intro.eval(F128::ONE),
            h2_folded
        );
        // glue: running := q1 + alpha · q_intro; T_r := T_r + alpha · h2_folded
        let combined = RoundQuad::fold(&q1, &q_intro, alpha);
        t_r += alpha * h2_folded;
        // The combined quad must satisfy sumcheck identity against the new T_r
        assert_eq!(combined.eval(F128::ZERO) + combined.eval(F128::ONE), t_r);
        // Apply the rest of the folds; each subsequent msg supersedes `combined` after eval.
        // After glue, the next fold uses challenge ris[1]. msgs[3] is from fold(ris[1]).
        let mut running = combined;
        // Remaining prover folds: ris[1..n-1] correspond to msgs[3..n+1].
        // Total prover-fold messages after start = (n-1) (single basis) ... but here we
        // have 1 start + 1 fold + 1 intro + (n-2) more folds = n+1 messages.
        assert_eq!(msgs.len(), n + 1);
        for (k, &r) in ris.iter().enumerate().skip(1).take(n - 2) {
            t_r = running.eval(r);
            let msg = msgs[2 + k]; // idx 3, 4, ...
            running = RoundQuad::from_msg(msg, t_r);
            assert_eq!(
                running.eval(F128::ZERO) + running.eval(F128::ONE),
                t_r,
                "post-glue round k={k}"
            );
        }
        // Final: apply r_last to the LAST message's quad
        t_r = running.eval(r_last);

        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        // With the collapsed-basis design, combined_basis already holds
        // eq + α·b2 at the residual dim.
        let combined_resid =
            prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(
            f_resid * combined_resid,
            t_r,
            "residual inner product != t_r"
        );
    }

    /// `induce_sumcheck_poly` is consistent with the codeword:
    ///   1. `enforced_sum` equals `Σ_i α^i · c[q_i]` computed directly,
    ///   2. `Σ_j msg[j] · basis_poly[j]` equals `enforced_sum` (the sumcheck
    ///      claim that the verifier reduces to a residual eval).
    #[test]
    fn induce_sumcheck_poly_consistent_with_codeword() {
        use crate::challenger::Challenger;
        let log_msg = 4;
        let log_inv_rate = 1;
        let msg_cols = 1usize << log_msg;
        let block_len = msg_cols << log_inv_rate;

        // Single-lane (num_interleaved = 1, no v_challenges).
        let mut ch = crate::challenger::RandomChallenger::new(0xF00DCAFE);
        let msg: Vec<F128> = (0..msg_cols).map(|_| ch.sample_f128()).collect();

        // Encode via Flock's NTT (zero-pad to block_len).
        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let mut codeword = vec![F128::ZERO; block_len];
        codeword[..msg_cols].copy_from_slice(&msg);
        ntt.forward_transform(&mut codeword);

        // Pick random distinct query positions.
        let num_queries = 6;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| vec![codeword[q]]).collect();
        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);

        let (basis_poly, enforced_sum) =
            induce_sumcheck_poly(log_msg, &sks_vks, &opened_rows, &[], &queries, &alpha);
        assert_eq!(basis_poly.len(), msg_cols);

        // Check 1: enforced_sum = Σ_i eq(α, i_bin) · c[q_i]
        let alpha_weights: Vec<F128> = crate::lincheck::build_eq_table(&alpha)
            .into_iter()
            .take(queries.len())
            .collect();
        let expected: F128 = queries
            .iter()
            .zip(alpha_weights.iter())
            .map(|(&q, &w)| w * codeword[q])
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(enforced_sum, expected, "enforced_sum != eq(α)-batched c[q]");

        // Check 2: Σ_j msg[j] · basis_poly[j] = enforced_sum.
        // This is the LCH novel-basis identity: c[q] = Σ_j msg[j] · Ŵ_j(q_field),
        // so Σ_i α^i · c[q_i] = Σ_j msg[j] · Σ_i α^i · Ŵ_j(q_i_field) = Σ_j msg[j] · basis_poly[j].
        let inner: F128 = msg
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(inner, enforced_sum, "msg · basis_poly != enforced_sum");
    }

    /// `induce_sumcheck_poly_via_ntt` must be byte-identical to dense across
    /// shapes incl. the real m30_fast level dims.
    #[test]
    fn induce_sumcheck_poly_via_ntt_matches_dense() {
        use crate::challenger::Challenger;
        let shapes = [
            (4usize, 1usize, 0usize, 6usize),
            (3, 1, 2, 5),
            (6, 2, 3, 30),
            (10, 1, 6, 218),
            (8, 3, 3, 71),
            (5, 5, 3, 43),
            (0, 2, 1, 3),
        ];
        for (si, &(log_msg, log_inv_rate, log_int, n_queries)) in shapes.iter().enumerate() {
            let block_len = 1usize << (log_msg + log_inv_rate);
            let num_interleaved = 1usize << log_int;
            let mut ch = crate::challenger::RandomChallenger::new(0xA11CE ^ si as u64);
            let mut queries: Vec<usize> = Vec::new();
            while queries.len() < n_queries.min(block_len) {
                let q = (ch.sample_f128().lo as usize) % block_len;
                if !queries.contains(&q) {
                    queries.push(q);
                }
            }
            let nq = queries.len();
            let opened_rows: Vec<Vec<F128>> = (0..nq)
                .map(|_| ch.sample_f128_vec(num_interleaved))
                .collect();
            let v_challenges = ch.sample_f128_vec(log_int);
            let alpha = ch.sample_f128_vec(ceil_log2(nq.max(1)));
            let sks_vks = eval_sk_at_vks(log_msg);

            let dense = induce_sumcheck_poly(
                log_msg,
                &sks_vks,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            let ntt = induce_sumcheck_poly_via_ntt(
                log_msg,
                log_inv_rate,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            assert_eq!(ntt.1, dense.1, "shape {si}: enforced_sum");
            assert_eq!(ntt.0, dense.0, "shape {si}: basis_poly");
        }
    }

    /// The sparse-prefix transpose must equal the baseline dense transpose on
    /// the same scattered input, across sizes (incl. > and < the k=8 prefix gate).
    #[test]
    fn transpose_sparse_matches_dense() {
        use crate::challenger::Challenger;
        for &log_d in &[6usize, 11, 12, 14, 16, 18] {
            for &nq in &[1usize, 5, 43, 218] {
                let n = 1usize << log_d;
                let nq = nq.min(n);
                let mut ch =
                    crate::challenger::RandomChallenger::new(0xC0DE ^ (log_d * 131 + nq) as u64);
                let ntt = AdditiveNttF128::standard(log_d);
                let mut positions: Vec<usize> = Vec::new();
                let mut values: Vec<F128> = Vec::new();
                while positions.len() < nq {
                    let p = (ch.sample_f128().lo as usize) % n;
                    if !positions.contains(&p) {
                        positions.push(p);
                        values.push(ch.sample_f128());
                    }
                }
                // Baseline: scatter then dense transpose.
                let mut dense = vec![F128::ZERO; n];
                for (&p, &v) in positions.iter().zip(&values) {
                    dense[p] += v;
                }
                transpose_forward_ntt(&ntt, &mut dense, log_d);
                let sparse = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d);
                assert_eq!(sparse, dense, "log_d={log_d}, nq={nq}");
            }
        }
    }

    /// As above, with num_interleaved > 1 and non-empty v_challenges (the
    /// partial-eval challenges used to fold lanes).
    #[test]
    fn induce_sumcheck_poly_with_interleaving_and_v_challenges() {
        use crate::challenger::Challenger;
        let log_msg = 3; // msg_cols = 8
        let log_interleaved = 2; // num_interleaved = 4
        let log_inv_rate = 1; // block_len = 16
        let msg_cols = 1usize << log_msg;
        let num_interleaved = 1usize << log_interleaved;
        let block_len = msg_cols << log_inv_rate;
        let poly_len = msg_cols * num_interleaved;

        let mut ch = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        // poly[lane * msg_cols + col] convention (matches ligero_commit input).
        let poly: Vec<F128> = (0..poly_len).map(|_| ch.sample_f128()).collect();

        // v_challenges fold the lanes after commit. Under the LSB-lane layout,
        // f_folded is just partial_eval_lsb of the poly at v_challenges.
        let v_challenges: Vec<F128> = (0..log_interleaved).map(|_| ch.sample_f128()).collect();
        let f_folded = partial_eval_lsb(&poly, &v_challenges);
        assert_eq!(f_folded.len(), msg_cols);

        // Encode via ligero_commit (so we use the same matrix layout).
        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(&poly, log_msg, log_interleaved, log_inv_rate, &ntt);
        assert_eq!(w.block_len, block_len);

        let num_queries = 5;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| w.row(q).to_vec()).collect();

        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);
        let (basis_poly, enforced_sum) = induce_sumcheck_poly(
            log_msg,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );

        // The folded polynomial f_folded should satisfy Σ_j f_folded[j] · basis_poly[j] = enforced_sum.
        let inner: F128 = f_folded
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(
            inner, enforced_sum,
            "folded-msg · basis_poly != enforced_sum (interleaved + v_challenges path)"
        );
    }

    /// End-to-end roundtrip: prover proves `poly(z) = v`, verifier accepts.
    /// R = 1 (one recursive step).
    #[test]
    fn ligerito_r1_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0; // unused — kept to silence the moved literal

        let mut rng = crate::challenger::RandomChallenger::new(0xCAFE_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();

        // True value v = poly(z)
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![0; log_inv_rates.len()];
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let _ = num_queries; // queries derived per-level from log_inv_rates now

        // Prove
        let mut p_ch = crate::challenger::FsChallenger::new(b"test");
        let proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        // Verify
        let mut v_ch = crate::challenger::FsChallenger::new(b"test");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "verifier rejected a valid proof");
    }

    /// Run the size measurement at the configured (log_n, initial_k, ks, rates).
    /// `log_inv_rates.len()` must equal `recursive_ks.len() + 1` (one per commit).
    /// Also times the prover (best of 3 runs). Returns the measured proof size
    /// in bytes.
    fn size_breakdown_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        use crate::challenger::Challenger;
        use std::time::Instant;
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);

        // dims sanity: n1 = 16; after k_0=4 → 12; after k_1=3 → 9 → yr = 512 elems.
        let r = recursive_ks.len();
        let mut recursive_log_msg_cols = Vec::with_capacity(r);
        let mut n_running = log_n - initial_k;
        for &k in &recursive_ks {
            assert!(n_running >= k);
            recursive_log_msg_cols.push(n_running - k);
            n_running -= k;
        }

        let mut rng = crate::challenger::RandomChallenger::new(0xBEEFCAFE);
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        eprintln!(
            "log_n={log_n}  initial_k={initial_k}  ks={:?}  log_inv_rates={:?}  queries={:?}",
            recursive_ks, log_inv_rates, queries_per_level
        );
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);
        drop(eq); // free 16 MB

        let grinding_bits = vec![0; log_inv_rates.len()];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level,
            grinding_bits,
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
        };

        // Time the prover, best of 3.
        let mut best = std::time::Duration::from_secs(3600);
        let mut proof = {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            recursive_prover(&cfg, &poly, &z, v, &mut p_ch)
        };
        for _ in 0..3 {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            let t = Instant::now();
            proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
            let el = t.elapsed();
            if el < best {
                best = el;
            }
        }
        eprintln!(
            "--- Ligerito proof: prover {:.2?} (best of 3), size: ---",
            best
        );
        proof.print_size_breakdown();

        // Smoke-check it verifies (so we know the proof is valid, not just plausibly-sized).
        let mut v_ch = crate::challenger::FsChallenger::new(b"size-test");
        assert!(recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch));
        proof.size_bytes()
    }

    /// Uniform rate (basefold-style) baseline at m=20.
    #[test]
    fn ligerito_size_breakdown_m20_uniform_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 1, 1]);
    }

    /// **The actual Ligerito design**: rate decreases at deeper levels, so
    /// fewer queries are needed there.
    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate_thin() {
        // More levels with thin lanes + aggressive rate decrease.
        size_breakdown_at(20, 4, vec![3, 3, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_aggressive() {
        // Thin initial lanes + steep rate decrease.
        size_breakdown_at(24, 3, vec![3, 3, 3, 3, 3], vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_uniform_rate() {
        size_breakdown_at(24, 5, vec![5, 4, 3], vec![1, 1, 1, 1]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_decreasing_rate() {
        size_breakdown_at(24, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22() {
        size_breakdown_at(22, 4, vec![4, 4, 3], vec![1, 2, 3, 4]);
    }

    /// Same total scale as m=22 but with initial_k=6 (64-lane initial leaves)
    /// to make the L0 commit shape exactly match basefold's.
    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22_initial_k6() {
        size_breakdown_at(22, 6, vec![3, 3, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m23() {
        size_breakdown_at(23, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    /// Count the merkle multi-proof siblings that would be needed for `positions`
    /// against a tree with `num_leaves` leaves. Same algorithm as
    /// `merkle::merkle_multi_proof` but counts only — no tree allocation,
    /// O(positions.len() · log num_leaves). For size estimation at scales where
    /// the actual tree wouldn't fit in memory.
    fn multi_proof_num_siblings(positions: &[usize], num_leaves: usize) -> usize {
        let mut active: Vec<usize> = positions.to_vec();
        active.sort_unstable();
        active.dedup();
        let mut sib_count = 0usize;
        let mut level_len = num_leaves;
        while level_len > 1 {
            let mut next = Vec::with_capacity(active.len());
            let mut i = 0;
            while i < active.len() {
                let p = active[i];
                let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
                if sib_active {
                    i += 2;
                } else {
                    sib_count += 1;
                    i += 1;
                }
                next.push(p >> 1);
            }
            active = next;
            level_len >>= 1;
        }
        sib_count
    }

    /// Analytical size estimator — runs **only** the challenger-driven query
    /// sampling + merkle-multi-proof counting. Does NOT materialize the
    /// polynomial or any merkle tree, so it scales to m=29, m=30+.
    /// Returns total bytes; prints a per-level breakdown.
    fn estimate_size_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);
        let r = recursive_ks.len();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        // Dim/lane/queries per commit (R+1 commits).
        let mut log_num_interleaved: Vec<usize> = vec![initial_k];
        log_num_interleaved.extend_from_slice(&recursive_ks);
        let mut log_msg_cols: Vec<usize> = Vec::with_capacity(r + 1);
        let mut n_running = log_n;
        for i in 0..=r {
            assert!(
                n_running >= log_num_interleaved[i],
                "config infeasible at commit {i}: dim {n_running} < lanes {}",
                log_num_interleaved[i]
            );
            log_msg_cols.push(n_running - log_num_interleaved[i]);
            n_running -= log_num_interleaved[i]; // consumes initial_k or k_{i-1}
        }
        let yr_log_n = n_running; // = log_n - initial_k - Σ k_i
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let log_block_len: Vec<usize> = log_msg_cols
            .iter()
            .zip(log_inv_rates.iter())
            .map(|(&m, &r)| m + r)
            .collect();

        eprintln!(
            "m={log_n}  initial_k={initial_k}  ks={:?}  rates={:?}  queries={:?}  yr_log={yr_log_n}",
            recursive_ks, log_inv_rates, queries_per_level
        );

        // Drive a challenger-deterministic query sampling, count siblings.
        let mut ch = crate::challenger::FsChallenger::new(b"estimate");
        let mut total_opened = 0usize;
        let mut total_merkle = 0usize;
        for i in 0..=r {
            let bl = 1usize << log_block_len[i];
            let qn = queries_per_level[i];
            if qn > bl {
                eprintln!(
                    "  INFEASIBLE at commit {i}: queries ({qn}) > block_len ({bl}). Pick a higher rate (smaller bl) or smaller queries."
                );
                return usize::MAX;
            }
            let qs = sample_distinct_queries(&mut ch, bl, qn);
            let sib = multi_proof_num_siblings(&qs, bl);
            let opened = qn * (1usize << log_num_interleaved[i]) * ELEM;
            let merkle = sib * 32;
            let label = if i == 0 {
                "L0 (initial)"
            } else if i == r {
                "L{} (final)"
            } else {
                "L{} (recursive)"
            };
            eprintln!(
                "  {label} [bl=2^{}, lanes=2^{}, q={qn}]: opened={}  merkle={} ({} sibs)",
                log_block_len[i],
                log_num_interleaved[i],
                kb(opened),
                kb(merkle),
                sib,
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let yr_b = (1usize << yr_log_n) * ELEM;
        let roots_b = (r + 1) * 32;
        // Transcript: 1 start + 1 intro per recursive boundary (R) + sum(k_i) folds, all (u_0, u_2).
        let sumcheck_msgs = 1 + r + recursive_ks.iter().sum::<usize>();
        let tx_b = sumcheck_msgs * 2 * ELEM;
        let total = total_opened + total_merkle + yr_b + roots_b + tx_b;
        eprintln!(
            "  TOTALS: opened={}  merkle={}  yr={}  roots={}  transcript={}  → GRAND={}",
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(roots_b),
            kb(tx_b),
            kb(total),
        );
        total
    }

    /// Verify the estimator matches the actual measurement at m=20.
    #[test]
    fn estimator_matches_actual_m20() {
        let estimated = estimate_size_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        // Measure the real proof at the same shape (cheap at m=20) instead of
        // hardcoding a baseline that goes stale when query counts change.
        let actual = size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        let diff = estimated.abs_diff(actual);
        eprintln!("estimator={estimated}  actual={actual}  diff={diff}");
        // Drift is from different challenger seeds producing different query
        // positions (and hence slightly different octopus sibling counts).
        // 5% is plenty of room.
        assert!(
            diff < actual / 20,
            "estimator drift too large: {diff} bytes"
        );
    }

    /// **The headline measurement**: Ligerito at m=29 with decreasing rate.
    #[test]
    fn estimate_ligerito_m29() {
        eprintln!("\n=== Ligerito m=29 — decreasing rate (the real Ligerito design) ===");
        // Pick a reasonable config: thin lanes, aggressive rate decrease.
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!(
            "\n=== Ligerito m=29 — uniform rate 1/2 (basefold-style baseline, infeasible at deepest level) ==="
        );
        // Uniform rate with deep recursion: block_len at L5 = 2^6 = 64 < 221 queries.
        // Show this is structurally bad without aggressive rate decrease.
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 1, 1, 1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — uniform rate, shallower (R=2) ===");
        // To make uniform rate feasible, use fewer levels with bigger ks.
        estimate_size_at(29, 4, vec![10, 10], vec![1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — thinner lanes ===");
        estimate_size_at(
            29,
            3,
            vec![3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );
    }

    #[test]
    fn estimate_ligerito_m30() {
        eprintln!("\n=== Ligerito m=30 — decreasing rate ===");
        estimate_size_at(30, 4, vec![4, 4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n=== Ligerito m=30 — thinner lanes ===");
        estimate_size_at(
            30,
            3,
            vec![3, 3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
    }

    /// Apples-to-apples vs basefold: same initial interleaving factor
    /// `2^6 = 64` lanes at L0 (basefold's log_batch_size = 6).
    #[test]
    fn estimate_ligerito_m29_initial_k6() {
        eprintln!(
            "\n=== Ligerito m=29 — initial_k=6 (matches basefold's 64-lane initial leaves) ==="
        );
        // initial_k = 6, then ks chosen to keep deeper levels thin.
        eprintln!("\n  Config A: thin recursive lanes, aggressive rate decrease");
        estimate_size_at(29, 6, vec![3, 3, 3, 3, 3, 2], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n  Config B: medium recursive lanes, fewer levels");
        estimate_size_at(29, 6, vec![4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!("\n  Config C: 2x6-bit recursive lanes (= basefold's epoch leaves)");
        estimate_size_at(29, 6, vec![6, 6, 4, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn estimate_ligerito_m30_initial_k6() {
        eprintln!("\n=== Ligerito m=30 — initial_k=6 ===");
        eprintln!("\n  Config A: thin recursive lanes");
        estimate_size_at(
            30,
            6,
            vec![3, 3, 3, 3, 3, 3, 2],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );

        eprintln!("\n  Config B: medium");
        estimate_size_at(30, 6, vec![4, 4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6, 7]);
    }

    /// Multi-level (R = 2) roundtrip.
    #[test]
    fn ligerito_r2_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 18;
        let initial_k = 3;
        let k_0 = 3;
        let k_1 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xABCD_1234);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        // wtns_0: log_n - initial_k = 9, num_interleaved = 8
        // wtns_1: dim n1 = 9, num_interleaved = 2^k_0 = 8, msg_cols = 2^(9-3) = 64
        // After k_0 folds: dim 6. wtns_2: num_interleaved = 2^k_1 = 4, msg_cols = 2^(6-2) = 16
        // After k_1 folds: dim 4. yr = 16 elems.
        let log_inv_rates = vec![log_inv_rate; 3];
        let _ = num_queries;
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
        assert_eq!(proof.recursive_roots.len(), 2);
        assert_eq!(proof.recursive_proofs.len(), 1);

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let ok = recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "R=2 verifier rejected valid proof");
    }

    /// `LigeritoProof` bincode-roundtrips identically.
    #[test]
    fn ligerito_proof_bincode_roundtrip() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xDEED_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let mut p_ch = crate::challenger::FsChallenger::new(b"serde");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let bytes = bincode::serialize(&proof).expect("serialize");
        let proof2: LigeritoProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(proof, proof2);
        eprintln!("LigeritoProof bincode size: {} bytes", bytes.len());
    }

    /// `recursive_prover_with_basis` + `recursive_verifier_with_basis`
    /// roundtrip — this is the basefold-compatible signature that
    /// `pcs::open_batch` will call. Single-claim case (`b = eq(z, ·)`,
    /// `target = poly(z)`) — must round-trip cleanly.
    #[test]
    fn recursive_prover_with_basis_roundtrip_single_claim() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_CAFE);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, log_inv_rate, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "basis-based verifier rejected valid proof");
    }

    /// `induce_sumcheck_evaluate_at_residual` matches dense
    /// `induce_sumcheck_poly` + `partial_eval_lsb`.
    #[test]
    fn induce_sumcheck_evaluate_at_residual_matches_dense() {
        use crate::challenger::Challenger;
        let log_msg_cols = 6;
        let yr_log_n = 2;
        let prefix_len = log_msg_cols - yr_log_n;
        let num_interleaved = 4;
        let log_num_interleaved = 2;
        let num_queries = 5;

        let mut rng = crate::challenger::RandomChallenger::new(0x2017_5052);
        let queries: Vec<usize> = (0..num_queries).map(|i| (i * 7 + 3) % (1 << 8)).collect();
        let opened_rows: Vec<Vec<F128>> = (0..num_queries)
            .map(|_| (0..num_interleaved).map(|_| rng.sample_f128()).collect())
            .collect();
        let v_challenges: Vec<F128> = (0..log_num_interleaved)
            .map(|_| rng.sample_f128())
            .collect();
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();
        let ris_for_basis: Vec<F128> = (0..prefix_len).map(|_| rng.sample_f128()).collect();
        let sks_vks = eval_sk_at_vks(log_msg_cols);

        // Dense path
        let (basis_dense, dense_enforced_sum) = induce_sumcheck_poly(
            log_msg_cols,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );
        let dense_residual = partial_eval_lsb(&basis_dense, &ris_for_basis);

        // Succinct path
        let succinct_enforced_sum =
            induce_sumcheck_enforced_sum(&opened_rows, &v_challenges, &queries, &alpha);
        let succinct_residual = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &ris_for_basis,
            yr_log_n,
        );

        assert_eq!(
            succinct_enforced_sum, dense_enforced_sum,
            "enforced_sum mismatch"
        );
        assert_eq!(
            succinct_residual.len(),
            dense_residual.len(),
            "residual length mismatch"
        );
        for (i, (s, d)) in succinct_residual
            .iter()
            .zip(dense_residual.iter())
            .enumerate()
        {
            assert_eq!(s, d, "residual mismatch at y={i}");
        }
    }

    /// Regression for the final-level proximity binding (the Ligerito
    /// soundness fix). Every non-final recursion level folds its opened rows
    /// into the running sumcheck via `induce_sumcheck`; the final level used to
    /// only Merkle-check its opened rows, leaving `yr` (the claimed final
    /// message) constrained by a single scalar equation — so a malicious prover
    /// could solve for a `yr` that opens the commitment to an arbitrary value.
    ///
    /// The fixed verifier ties `yr` to the committed codeword by checking
    /// `enforced_sum_last == ⟨yr, induced_basis_last⟩`, exactly as every other
    /// level does. This test pins that identity against a *real* `ligero_commit`
    /// codeword: the honest `yr` (the committed message) satisfies it, and any
    /// perturbed `yr` violates it. If `ligero_commit`'s additive-NTT encoding
    /// and the verifier's LCH novel-basis (`induce_sumcheck_evaluate_at_residual`)
    /// ever diverged, the honest assertion here would fail.
    #[test]
    fn final_level_binding_pins_yr_to_committed_codeword() {
        use crate::challenger::Challenger;
        let log_msg_cols = 5; // yr has 32 entries (within the shipped yr_log_n range)
        let log_inv_rate = 1;
        let num_queries = 20;
        let msg_cols = 1usize << log_msg_cols;
        let block_len = msg_cols << log_inv_rate;

        let mut rng = crate::challenger::RandomChallenger::new(0xB19D_1235);
        // num_interleaved = 1 ⇒ no lane fold (level_rs empty) ⇒ yr == the message.
        let yr: Vec<F128> = (0..msg_cols).map(|_| rng.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_msg_cols + log_inv_rate);
        let wtns = ligero_commit(&yr, log_msg_cols, 0, log_inv_rate, &ntt);

        // Distinct query positions (the protocol always samples distinct ones).
        let mut queries: Vec<usize> = Vec::new();
        let mut q = 1usize;
        while queries.len() < num_queries {
            q = (q * 73 + 41) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&p| wtns.row(p).to_vec()).collect();

        let level_rs: Vec<F128> = Vec::new(); // num_interleaved = 1
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();

        // The two quantities the fixed verifier batches into the final check.
        let enforced_sum = induce_sumcheck_enforced_sum(&opened_rows, &level_rs, &queries, &alpha);
        let sks_vks = eval_sk_at_vks(log_msg_cols);
        let induced_basis = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &[],
            log_msg_cols,
        );
        let inner = |v: &[F128]| -> F128 {
            v.iter()
                .zip(induced_basis.iter())
                .map(|(&a, &b)| a * b)
                .fold(F128::ZERO, |s, x| s + x)
        };

        // Honest yr (the committed message) satisfies the proximity tie.
        assert_eq!(
            inner(&yr),
            enforced_sum,
            "honest yr must satisfy ⟨yr, induced_basis⟩ == enforced_sum"
        );

        // A forged yr violates it: perturb a coordinate with nonzero basis weight,
        // so the change to the inner product is provably nonzero.
        let jnz = induced_basis
            .iter()
            .position(|b| !b.is_zero())
            .expect("induced basis must not be identically zero");
        let mut yr_bad = yr.clone();
        yr_bad[jnz] += F128::ONE;
        assert_ne!(
            inner(&yr_bad),
            enforced_sum,
            "a forged yr must break the final-level proximity tie"
        );
    }

    /// Succinct verifier accepts the same proof as the dense verifier when
    /// given an `eval_b` closure that returns the same values as the dense
    /// `b_initial[idx]` at multilinear `point = bit-decomp(idx)`.
    #[test]
    fn recursive_verifier_with_basis_succinct_matches_dense() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x52CC_2017);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, log_inv_rate, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        // Dense verifier
        let mut v_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let dense_ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(dense_ok, "dense verifier must accept");

        // Succinct verifier — batch eval_b is just eq(z, ris ++ y_bits) by construction
        let mut v_ch2 = crate::challenger::FsChallenger::new(b"succ-cmp");
        let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
            let yr_len = 1usize << yr_log_n;
            let mut point = ris.to_vec();
            point.resize(ris.len() + yr_log_n, F128::ZERO);
            (0..yr_len)
                .map(|y| {
                    for j in 0..yr_log_n {
                        point[ris.len() + j] = if (y >> j) & 1 == 1 {
                            F128::ONE
                        } else {
                            F128::ZERO
                        };
                    }
                    crate::zerocheck::multilinear::eq_eval(&z, &point)
                })
                .collect()
        };
        let succ_ok = recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_root,
            eval_b_residual,
            &mut v_ch2,
        );
        assert!(succ_ok, "succinct verifier must accept");
    }

    /// Build a matching (ProverConfig, VerifierConfig) pair with explicit
    /// OOD samples and fold-challenge grinding, for the OOD-path tests below.
    /// Shape: L0 (initial_k) → r recursive levels of `k`; small query counts
    /// and grind bits keep the test fast while still exercising every path.
    fn ood_test_configs(
        log_n: usize,
        initial_k: usize,
        ks: &[usize],
        ood_samples: Vec<usize>,
        fold_grinding_bits: Vec<usize>,
    ) -> (ProverConfig, VerifierConfig) {
        let r = ks.len();
        let log_inv_rates: Vec<usize> = (0..=r).map(|i| 1 + i).collect();
        let mut recursive_log_msg_cols = Vec::new();
        let mut dim = log_n - initial_k;
        for &k in ks {
            recursive_log_msg_cols.push(dim - k);
            dim -= k;
        }
        let queries = vec![20usize; r + 1];
        let grinding_bits = vec![0usize; r + 1];
        let p = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: ks.to_vec(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
        };
        let v = VerifierConfig {
            log_inv_rates,
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: ks.to_vec(),
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
        };
        (p, v)
    }

    /// End-to-end OOD binding + fold-challenge grinding: a JohnsonOod-shaped
    /// config (explicit OOD samples at L1/L2, a few fold-grind bits at every
    /// level) round-trips through BOTH the dense and succinct verifiers, and
    /// tampering with either an OOD value or a fold-grinding nonce makes both
    /// reject. Exercises every new prover/verifier code path.
    #[test]
    fn ligerito_ood_and_fold_grinding_roundtrip_and_tamper() {
        use crate::challenger::Challenger;
        let log_n = 12;
        let initial_k = 2;
        let ks = [2usize, 2];
        // OOD at L1 and L2 (L0 must be 0); 3 fold-grind bits at each level.
        let (p_cfg, v_cfg) = ood_test_configs(log_n, initial_k, &ks, vec![0, 2, 2], vec![3, 3, 3]);

        let mut rng = crate::challenger::RandomChallenger::new(0x00D_7E57);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, 1, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"ood-test");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        // Sanity: the new proof fields are populated.
        assert_eq!(proof.ood_values.len(), 4, "2 OOD samples each at L1 and L2");
        // 2 lane folds (L0) + 2 + 2 recursive folds, each with 3 grind bits.
        assert_eq!(proof.fold_grinding_nonces.len(), initial_k + ks[0] + ks[1]);

        let dense = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis(&v_cfg, proof, &b, target, &initial_root, &mut ch)
        };
        let eval_b_residual = {
            let z = z.clone();
            move |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
                let yr_len = 1usize << yr_log_n;
                let mut point = ris.to_vec();
                point.resize(ris.len() + yr_log_n, F128::ZERO);
                (0..yr_len)
                    .map(|y| {
                        for j in 0..yr_log_n {
                            point[ris.len() + j] = if (y >> j) & 1 == 1 {
                                F128::ONE
                            } else {
                                F128::ZERO
                            };
                        }
                        crate::zerocheck::multilinear::eq_eval(&z, &point)
                    })
                    .collect()
            }
        };
        let succinct = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis_succinct(
                &v_cfg,
                proof,
                log_n,
                target,
                &initial_root,
                &eval_b_residual,
                &mut ch,
            )
        };

        assert!(dense(&proof), "dense verifier must accept OOD proof");
        assert!(succinct(&proof), "succinct verifier must accept OOD proof");

        // Tamper an OOD value → both verifiers reject.
        let mut bad_ood = proof.clone();
        bad_ood.ood_values[0] += F128::ONE;
        assert!(!dense(&bad_ood), "dense must reject tampered OOD value");
        assert!(
            !succinct(&bad_ood),
            "succinct must reject tampered OOD value"
        );

        // Tamper a fold-grinding nonce → both verifiers reject (PoW fails or
        // the FS state diverges).
        let mut bad_nonce = proof.clone();
        bad_nonce.fold_grinding_nonces[0] ^= 0xDEAD_BEEF;
        assert!(!dense(&bad_nonce), "dense must reject tampered fold nonce");
        assert!(
            !succinct(&bad_nonce),
            "succinct must reject tampered fold nonce"
        );
    }

    /// A real embedded profile config (m=22 fast = JohnsonOod) drives a full
    /// prover→verifier round-trip through the basis opening path. This is the
    /// production shape: OOD samples and fold grinding come straight from the
    /// derived TOML, not a hand-built config.
    #[test]
    fn ligerito_fast_profile_m22_roundtrip() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");
        // The fast profile must actually use the new features.
        assert!(p_cfg.ood_samples.iter().skip(1).any(|&s| s > 0));
        assert!(p_cfg.fold_grinding_bits.iter().any(|&g| g > 0));

        let mut rng = crate::challenger::RandomChallenger::new(0xFA57_0022);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, 1, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "m22 fast profile proof must verify"
        );
    }

    /// Multi-claim batched basis: `b = γ_1·eq(z_1, ·) + γ_2·eq(z_2, ·)`,
    /// `target = γ_1·poly(z_1) + γ_2·poly(z_2)`. This is the shape ring_switch
    /// produces.
    #[test]
    fn recursive_prover_with_basis_roundtrip_batched_claims() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_BA51);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z1: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let z2: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let g1 = rng.sample_f128();
        let g2 = rng.sample_f128();
        let b1 = build_eq_table(&z1);
        let b2 = build_eq_table(&z2);
        let b: Vec<F128> = b1
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| g1 * a + g2 * c)
            .collect();
        let v1: F128 = poly
            .iter()
            .zip(b1.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let v2: F128 = poly
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let target = g1 * v1 + g2 * v2;

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(&poly, log_msg_cols_0, initial_k, log_inv_rate, &ntt_0);
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"batched");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"batched");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "batched-basis verifier rejected valid proof");
    }

    /// `recursive_prover_with_l0` (external L0 path, for integration with
    /// Flock's `pcs::commit`) produces a byte-identical proof to
    /// `recursive_prover` when given a matching pre-built L0.
    #[test]
    fn recursive_prover_with_l0_matches_full() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xACED_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        // Path 1: built-in L0 commit.
        let mut p_ch = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_a = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        // Path 2: build L0 externally via ligero_commit, then call _with_l0.
        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let mut wtns_0_external =
            ligero_commit(&poly, log_msg_cols_0, initial_k, log_inv_rate, &ntt_0);
        let mut p_ch_b = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_b = recursive_prover_with_l0(
            &cfg,
            &poly,
            std::mem::take(&mut wtns_0_external.mat),
            std::mem::take(&mut wtns_0_external.tree),
            &z,
            v,
            &mut p_ch_b,
        );

        // Proofs must be byte-identical (same FS state, same prover work).
        assert_eq!(proof_a.initial_root, proof_b.initial_root);
        assert_eq!(proof_a.recursive_roots, proof_b.recursive_roots);
        assert_eq!(proof_a.final_proof.yr, proof_b.final_proof.yr);
        assert_eq!(
            proof_a.sumcheck_transcript.len(),
            proof_b.sumcheck_transcript.len()
        );
        for (ma, mb) in proof_a
            .sumcheck_transcript
            .iter()
            .zip(proof_b.sumcheck_transcript.iter())
        {
            assert_eq!(ma.u_0, mb.u_0);
            assert_eq!(ma.u_2, mb.u_2);
        }
        // And both must verify against the same VerifierConfig.
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"l0-test");
        assert!(recursive_verifier(&v_cfg, &proof_b, &z, v, &mut v_ch));
    }

    /// Mutation rejection: change one element of yr → verify should fail.
    #[test]
    fn ligerito_r1_rejects_mutated_yr() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let _ = num_queries;
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let mut proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        // Mutate yr.
        proof.final_proof.yr[0] += F128::ONE;

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(!ok, "verifier accepted a proof with mutated yr");
    }

    #[test]
    fn ligero_commit_encoding_roundtrips_via_inv_ntt() {
        let log_msg = 4; // msg_cols = 16
        let log_interleaved = 3; // num_interleaved = 8
        let log_inv_rate = 1; // block_len = 32
        let msg_cols = 1 << log_msg;
        let num_interleaved = 1 << log_interleaved;
        let block_len = msg_cols << log_inv_rate;

        // Deterministic dummy polynomial.
        let poly: Vec<F128> = (0..num_interleaved * msg_cols)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x9E3779B97F4A7C15),
                    0x1234 ^ i as u64,
                )
            })
            .collect();

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(&poly, log_msg, log_interleaved, log_inv_rate, &ntt);
        assert_eq!(w.block_len, block_len);
        assert_eq!(w.num_interleaved, num_interleaved);
        assert_eq!(w.mat.len(), block_len * num_interleaved);

        // Per-lane inv-NTT should recover the padded message. Under the LSB-lane
        // layout, lane `lane`'s col `col` message lives at `poly[col * num_interleaved + lane]`.
        for lane in 0..num_interleaved {
            let mut col: Vec<F128> = (0..block_len)
                .map(|pos| w.mat[pos * num_interleaved + lane])
                .collect();
            ntt.inverse_transform(&mut col);
            for col_idx in 0..msg_cols {
                assert_eq!(
                    col[col_idx],
                    poly[col_idx * num_interleaved + lane],
                    "lane {lane} col_idx {col_idx} mismatch",
                );
            }
            for col_idx in msg_cols..block_len {
                assert_eq!(
                    col[col_idx],
                    F128::ZERO,
                    "lane {lane} pad position {col_idx} not zero",
                );
            }
        }

        // Merkle root is deterministic: re-running the same commit yields the
        // same root.
        let w2 = ligero_commit(&poly, log_msg, log_interleaved, log_inv_rate, &ntt);
        assert_eq!(w.root(), w2.root());
    }
}
