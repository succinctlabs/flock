// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from bolt-rs (https://github.com/bcc-research/bolt-rs,
// `ligerito_recursive.rs`).

//! Ligerito: recursive multilinear PCS.
//!
//! Ported from bolt-rs (`ligerito_recursive.rs`) onto Flock primitives. The
//! committed Reed--Solomon words remain in `F128` (GHASH irreducible) and use
//! [`AdditiveNttF128`], while fold challenges and running sumcheck claims use
//! the quadratic extension `F256`. Merkle commitments come from
//! [`crate::merkle`] and Fiat--Shamir from [`Challenger`].
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
use crate::field::{F128, F256, F256Unreduced};
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use crate::pcs::LOG_PACKING;
use crate::pcs::stratified;
use serde::{Deserialize, Serialize};

pub(crate) mod extension;

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
/// - `Fast`:   rate 1/2, Johnson list-decoding regime with two-point OOD
///             binding, the aggressive recursion ladder (rate +2/level) and
///             16-bit query PoW at every level: the query term targets 112
///             bits and the PoW supplies the rest, work-normalized to 128.
///             MCA/fold arithmetic over F256. Default.
/// - `Slim`:   rate 1/4, same Johnson + OOD accounting, aggressive ladder and
///             16-bit query PoW as `Fast`. Roughly half the proof, ~2x L0
///             encoding work.
/// - `Secure`: rate 1/2, unique-decoding regime (list size 1, no OOD),
///             120-bit overall soundness. Largest proof, most conservative
///             analysis.
///
/// History (2026-08-27): the grind-free `Fast`/`Slim` with the +1/level
/// ladder were deleted and the `Fast128`/`Slim128` twins took their names
/// (Ron's call, bloat ledger §C). Proof-IO v22 marks the transcript move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LigeritoProfile {
    /// The aggressive ladder climbs the rate +2 per level (vs +1), so the
    /// same folds reach higher rates sooner and each tail level needs fewer
    /// consistency queries; codewords still shrink per level because
    /// `rate_gain` stays below the fold count. See
    /// [`Self::derive_profile_ladder`].
    #[default]
    Fast,
    /// The 100-bit cost point kept for deployments on the chain100
    /// configuration: rate 1/2, the shipped +1/level ladder, NO query PoW,
    /// and the consistency-query term targets the profile's own 100 bits
    /// (m27 ladder [218, 106, 71, 53]). Since the 2026-08-27 consolidation
    /// it differs from `Fast` in ladder, PoW and target — it is a frozen
    /// historical schedule, not "`Fast` with one knob changed".
    Fast100,
    Slim,
    /// `Slim`'s 100-bit cost point — the chain100 track's envelope outers run
    /// on it exactly as the leaf track runs on [`Self::Fast100`]. Rate 1/4,
    /// the shipped +1/level ladder, 16-bit query PoW; the query term targets
    /// the profile's own 100 bits (m29 ladder total 262, the schedule the
    /// envelope's fixed point was iterated against). Frozen, like `Fast100`.
    Slim100,
    Secure,
}

impl LigeritoProfile {
    /// L0 code rate index for this profile (`rho_0 = 2^-log_inv_rate`).
    pub fn log_inv_rate(self) -> usize {
        match self {
            Self::Fast100 | Self::Fast | Self::Secure => 1,
            Self::Slim100 | Self::Slim => 2,
        }
    }
    /// Historical profile-local target. Strict Fast/Slim configs retain the
    /// value 100 for compatibility but independently override every Johnson
    /// component to the 128-bit floors enforced by `validate()`; Fast100 and
    /// Slim100 keep the 100-bit query floor, and Secure keeps its 120-bit UDR
    /// query floor.
    pub fn security_bits(self) -> usize {
        match self {
            Self::Fast100 | Self::Fast | Self::Slim100 | Self::Slim => 100,
            Self::Secure => 120,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Fast100 => "fast100",
            Self::Slim => "slim",
            Self::Slim100 => "slim100",
            Self::Secure => "secure",
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
    /// Per-level PoW bits for scalar challenges that batch evaluation claims
    /// (OOD claims and the recursive consistency claim). Length =
    /// recursive_steps + 1.
    pub claim_batch_grinding_bits: Vec<usize>,
    /// Per-level PoW bits for the vector challenge that batches the queried
    /// consistency equations. Length = recursive_steps + 1.
    pub consistency_batch_grinding_bits: Vec<usize>,
    /// Per-commit-level out-of-domain samples (L0, ..., L_r), taken right
    /// after the level's Merkle root enters the transcript. `[0]` must be 0:
    /// L0 is bound by the opening's own (post-commit, random-point)
    /// evaluation claim. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
    /// Hash backing every Merkle commitment this prover makes (L0 and each
    /// recursive level). Comes from the `hash` field of the security config;
    /// [`Default`] is SHA-256.
    pub merkle_hash: HashKind,
    /// Per-level stratified query schedules (L0, ..., L_r), same indexing as
    /// `queries`. Computed from the query counts ONCE, at config-build time
    /// (every constructor ends in [`Self::with_default_stratified`]) and
    /// consumed verbatim — never derived at proof time, and never from a
    /// proof's shape: the allocation is statement authority
    /// (`docs/stratified-queries.tex`). Custom schedules are allowed but
    /// must pass [`stratified::validate_schedules`].
    pub stratified: Vec<stratified::LevelSchedule>,
}

impl ProverConfig {
    /// Per-level block logs from the declared ladder shape (the block
    /// `queries[ℓ]` opens): see [`stratified::level_block_logs`].
    pub fn level_block_logs(&self) -> Vec<usize> {
        stratified::level_block_logs(
            self.initial_log_msg_cols,
            &self.recursive_log_msg_cols,
            &self.log_inv_rates,
        )
    }

    /// Attach the canonical stratified schedule (the binary decomposition of
    /// each level's query count). Every construction site ends with this;
    /// hand-customized schedules replace the field afterwards and must pass
    /// [`stratified::validate_schedules`].
    pub fn with_default_stratified(mut self) -> Self {
        self.stratified = stratified::schedules(&self.queries, &self.level_block_logs());
        self
    }

    /// Validate the stored schedules against the query counts and ladder
    /// shape (the load-time authority check).
    pub fn validate_stratified(&self) -> Result<(), String> {
        stratified::validate_schedules(&self.stratified, &self.queries, &self.level_block_logs())
    }

    /// The L0 cap depth this config implies — the ONE rule commit-time
    /// sizing (`PcsParams::l0_cap_depth`), the open entries'
    /// belt-and-braces asserts, and the prover's own absorb must share:
    /// the L0 schedule's cap (top set bit).
    pub fn l0_cap_depth(&self) -> usize {
        self.stratified[0].cap_depth()
    }
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
    /// Per-level evaluation-claim batching PoW bits. Length = recursive_steps + 1.
    pub claim_batch_grinding_bits: Vec<usize>,
    /// Per-level queried-consistency batching PoW bits. Length = recursive_steps + 1.
    pub consistency_batch_grinding_bits: Vec<usize>,
    /// Per-commit-level OOD samples. Length = recursive_steps + 1.
    pub ood_samples: Vec<usize>,
    /// Hash the prover's Merkle commitments were built under. Must match the
    /// prover's — a mismatch makes every opening fail to verify, which is the
    /// correct outcome: the root commits to the hash as much as to the data.
    pub merkle_hash: HashKind,
    /// Per-level stratified query schedules — the verifier's own copy of the
    /// statement-side allocation, same contract as
    /// [`ProverConfig::stratified`]: config-build-time data, never derived
    /// at proof time or from the proof.
    pub stratified: Vec<stratified::LevelSchedule>,
}

impl VerifierConfig {
    /// Per-level block logs from the declared ladder shape: see
    /// [`stratified::level_block_logs`].
    pub fn level_block_logs(&self) -> Vec<usize> {
        stratified::level_block_logs(
            self.initial_log_msg_cols,
            &self.recursive_log_msg_cols,
            &self.log_inv_rates,
        )
    }

    /// Attach the canonical stratified schedule; see
    /// [`ProverConfig::with_default_stratified`].
    pub fn with_default_stratified(mut self) -> Self {
        self.stratified = stratified::schedules(&self.queries, &self.level_block_logs());
        self
    }

    /// Validate the stored schedules (the load-time authority check).
    pub fn validate_stratified(&self) -> Result<(), String> {
        stratified::validate_schedules(&self.stratified, &self.queries, &self.level_block_logs())
    }

    /// The L0 cap depth this config implies; see
    /// [`ProverConfig::l0_cap_depth`].
    pub fn l0_cap_depth(&self) -> usize {
        self.stratified[0].cap_depth()
    }
}

/// Proximity loss `ε*` for the UDR (unique-decoding regime) analysis. It
/// would back the proximity radius off to `γ = δ/2 − ε*` (δ = 1 − ρ the
/// code's relative distance); set to `0`, so we decode to the full
/// unique-decoding radius `γ = δ/2` with no backoff. Per our paper's Appendix
/// C.3 (Theorem `ca-udr`, BCHKS25 Cor. 1.4) the proximity-gap exceptional set
/// is then `a = γ·n + 1` — length-dependent (see [`paper_thm_1_4_log_a`]), so
/// `eps_pg = 256 − log₂ a` shrinks ~1 bit per witness doubling. The quadratic
/// extension leaves ample margin for the shipped shapes, so their
/// `fold_grinding_bits` are zero.
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
///
/// **This count is already the with-replacement count.** The bad event is
/// "every queried position lands in the agreement set `S`", `|S|/n ≤ 1−γ`.
/// For `q` independent uniform draws that is exactly `(|S|/n)^q ≤ (1−γ)^q`,
/// which is the `q = ⌈100 / log₂(1/(1−γ))⌉` solved for here. Sampling
/// *without* replacement is the strictly better hypergeometric
/// `∏_{i<q} (|S|−i)/(n−i) ≤ (1−γ)^q`, so the old distinct-query sampler was
/// buying slack this function never spent. Moving to [`sample_queries`] (with
/// replacement) therefore changes **no** query count — it only stops earning
/// an unclaimed bonus. See also the alpha-batching note in
/// [`induce_sumcheck_evaluate_at_residual`]: repeated positions merge into one
/// constraint with a combined weight, which leaves the multilinear
/// Schwartz–Zippel term `⌈log₂ Q⌉/2^128` untouched.
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
/// Strategy: three original-variable folds per recursive level plus the
/// coordinate bit introduced by the F256-to-F128 code switch (`k_i = 4`),
/// with **decreasing rate**
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
        let original_folds = 3.min(n_running);
        let k = original_folds + 1;
        let log_msg_cols_next = n_running - original_folds;
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
        n_running -= original_folds;
        rate_running = next_rate;
    }

    if recursive_ks.is_empty() {
        return Err("log_n too small — no recursive levels for the Ligerito recursion");
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
        claim_batch_grinding_bits: vec![0usize; n_levels],
        consistency_batch_grinding_bits: vec![0usize; n_levels],
        ood_samples: vec![0usize; n_levels],
        merkle_hash: HashKind::default(),
        stratified: vec![],
    }
    .with_default_stratified())
}

/// Recursion-ladder shape: per-level dims (index 0 = L0) plus the residual.
struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    log_num_interleaved: Vec<usize>,
    k_recursive: Vec<usize>,
    yr_log_n: usize,
}

/// Ladder-shape generator with tunable recursion aggressiveness — the shape
/// derivation behind [`default_config`] and
/// [`LigeritoSecurityConfig::derive_profile`]: each recursive commitment adds
/// one base-coordinate bit, so a four-round level removes three variables
/// from the extension table. The rate index increases by ≥ `rate_gain` per
/// level, bumped further whenever the block length is narrower than
/// `queries_at_rate(rate)`. That width rule is a **proof-size convention,
/// not a soundness bound**: [`sample_queries`] draws with replacement and
/// stays sound for any width (see [`udr_queries`]). A block narrower than
/// its query count would just open most of itself, which is a shape worth
/// refusing. Keeping the rule unchanged also keeps every shipped ladder
/// byte-identical to before the with-replacement switch.
///
/// `folds_per_level` original variables are folded at each recursive level
/// (the interleave is `folds_per_level + 1`, one extra for the F256→F128
/// code-switch coordinate), and the rate climbs by `rate_gain` per level
/// (bumped further only if a level's block length would fall below its query
/// count). The default `(3, 1)` reproduces the shipped ladder; larger values
/// approach WHIR's shape (fold more, gain rate faster → fewer, higher-rate
/// tail levels). NB in F128 (16 B/element) a large `folds_per_level` widens
/// the opened row to `2^(folds+1)` elements, which can cost more than the
/// query reduction saves — the reason to measure, not assume.
fn derive_ladder_shape_tuned(
    log_n: usize,
    initial_k: usize,
    log_inv_rate: usize,
    folds_per_level: usize,
    rate_gain: usize,
    queries_at_rate: &dyn Fn(usize) -> usize,
) -> Result<LadderShape, String> {
    assert!(
        folds_per_level >= 1 && rate_gain >= 1,
        "ladder tuning must be >= 1"
    );
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
        let original_folds = folds_per_level.min(n_running);
        let k = original_folds + 1;
        let log_msg_cols_next = n_running - original_folds;
        let mut next_rate = rate_running + rate_gain;
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
        n_running -= original_folds;
        rate_running = next_rate;
    }
    if shape.k_recursive.len() < 2 {
        return Err("log_n too small — no recursive levels for the Ligerito recursion".into());
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
                (($m, LigeritoProfile::Fast100),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_fast100.toml"))),
                (($m, LigeritoProfile::Slim),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_slim.toml"))),
                (($m, LigeritoProfile::Slim100),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_slim100.toml"))),
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

/// The `initial_k` (L0 interleave = `log_batch_size`) the embedded config
/// for `(m, profile)` was derived with. **The TOML is the source of truth**:
/// callers building `PcsParams` at a content-derived `m` must use this as
/// `log_batch_size` — `prover_config_for` rejects a mismatch. 6 everywhere
/// except the m28 fast/slim families (4) and the m29 fast/slim families
/// (5 — the recursion-node row-width choices; see `derive_profile`).
/// Returns `None` when no config is registered.
pub fn embedded_initial_k(m: usize, profile: LigeritoProfile) -> Option<usize> {
    let toml = embedded_security_config(m, profile)?;
    // Cheap scan — the TOML serializer always writes `initial_k = <n>` as
    // its own line; full parse+validate happens at config load.
    toml.lines().find_map(|l| {
        l.strip_prefix("initial_k = ")
            .and_then(|v| v.trim().parse().ok())
    })
}

/// [`embedded_initial_k`] with the universal pre-m29 default for shapes
/// without a registered config (ad-hoc/test geometries).
pub fn embedded_initial_k_or_default(m: usize, profile: LigeritoProfile) -> usize {
    embedded_initial_k(m, profile).unwrap_or(6)
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
    sec.validate_profile(profile)?;
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
    sec.validate_profile(profile)?;
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
        claim_batch_grinding_bits: p.claim_batch_grinding_bits,
        consistency_batch_grinding_bits: p.consistency_batch_grinding_bits,
        ood_samples: p.ood_samples,
        merkle_hash: p.merkle_hash,
        stratified: vec![],
    }
    .with_default_stratified())
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
    /// so the proximity-gap term is length-dependent. Fold/MCA arithmetic is
    /// over F256; `eta` is `None` for this regime.
    Udr,
    /// Johnson radius with explicit slack `η` (γ = (1 − √ρ) − η) **with
    /// out-of-domain binding**. Theorem 1.5 of the same paper gives the
    /// proximity-gap exceptional set `a = O_ρ(n / η^5)`; the level's
    /// `fold_grinding_bits` should be ≥ (target_bits − log₂(q²/a)) when the
    /// fold challenge is sampled in F256.
    /// Binding to a single codeword of the (Johnson-bounded) interleaved list
    /// uses two independent post-commit evaluations. At L0 the ordinary
    /// ring-switched opening supplies the first, conservatively accounted at
    /// degree `m = μ + 7`, and `ood_samples = 1` supplies a packed-polynomial
    /// evaluation of degree `μ`. Deeper levels use two explicit degree-`μ`
    /// samples.
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
    /// PoW bits on every scalar coefficient that batches evaluation claims at
    /// this level. The Flock paper's Appendix C.3 bounds the bad event by
    /// `L_max / |F|`.
    #[serde(default)]
    pub claim_batch_grinding_bits: usize,
    /// PoW bits on the multilinear challenge used to batch this level's
    /// queried consistency equations. Its Schwartz--Zippel numerator is
    /// `L_max * ceil(log2(queries))`.
    #[serde(default)]
    pub consistency_batch_grinding_bits: usize,
    /// Additional out-of-domain samples taken right after this level's commit
    /// enters the transcript (`JohnsonOod` only). Two total binding points are
    /// required: exactly 1 explicit sample at L0 (whose opening claim is the
    /// other point), and exactly 2 at every deeper level.
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
    /// Diagnostic — OOD collision-binding bits (`JohnsonOod` only). Deeper
    /// levels use `2·(128 − log₂μ) − (2·log₂L − 1)`. At L0 the ordinary
    /// ring-switched opening has degree at most `m = μ + LOG_PACKING`, so the
    /// conservative bound is `(128 − log₂m) + (128 − log₂μ)
    /// − (2·log₂L − 1)`. Here `L` is the Johnson interleaved-list bound.
    #[serde(default)]
    pub expected_eps_ood_bits: Option<f64>,
    /// Diagnostic -- unground claim-batching bits
    /// `128 - log2(L_max)` from the Flock paper's Appendix C.3.
    #[serde(default)]
    pub expected_eps_claim_batch_bits: f64,
    /// Diagnostic -- unground per-round sumcheck bits
    /// `256 - log2(2 * L_max)` from the Flock paper's Appendix C.3, because
    /// the sumcheck challenge and running claim are in F256.
    #[serde(default)]
    pub expected_eps_sumcheck_bits: f64,
    /// Diagnostic -- unground queried-consistency batching bits
    /// `128 - log2(L_max * ceil(log2(queries)))`.
    #[serde(default)]
    pub expected_eps_consistency_batch_bits: f64,
}

/// Descriptor for the final-residual block (`yr`) sent in the clear at the
/// end of the last recursive level. It has no commit and no queries, so the
/// only meaningful parameter is its dimension.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalBlockConfig {
    /// Dimension of the final extension table. The clear residual contains
    /// `2^(yr_log_n + 1)` F128 words after its final coordinate split.
    pub yr_log_n: usize,
}

/// Complete security spec for one Ligerito instance, covering a single
/// `(hash, m)` pair. Designed to round-trip cleanly via serde (TOML/JSON).
///
/// **Validation invariants** (checked by [`Self::validate`]):
/// 1. `initial_k + Σ levels[1..](k_recursive - 1) +
///    final_block.yr_log_n == log_n`.
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
    /// Field used by the fold/MCA arithmetic. Commitments remain over F128.
    /// The implemented secure value is `"f256"`.
    pub field: String,
    /// Hash function used by the Merkle commitments: `"sha256"` or
    /// `"blake3"`. Read via [`LigeritoSecurityConfig::merkle_hash`] and
    /// carried into the prover/verifier configs; [`validate`] rejects any
    /// other value.
    ///
    /// This selects the **Merkle** hash only. The Fiat-Shamir transcript hash
    /// is a separate, independent choice made where the challenger is built
    /// ([`crate::challenger::FsChallenger::with_hash`]) — the challenger is
    /// constructed by the caller, upstream of any PCS config, so there is
    /// deliberately no field for it here rather than one that cannot drive
    /// anything.
    ///
    /// [`validate`]: LigeritoSecurityConfig::validate
    pub hash: String,
    /// Where in the per-level FS transcript grinding is placed.
    pub grinding_step: GrindingStep,
    /// Per-level parameters, in order L0, L1, L2, ....
    pub levels: Vec<LigeritoLevelConfig>,
    /// Final residual block descriptor.
    pub final_block: FinalBlockConfig,
}

/// Base field used by commitments, OOD points and answers, and the batching
/// challenges that retain their existing grinding schedules.
const BASE_FIELD_LOG_Q: f64 = 128.0;
/// Quadratic extension used by every recursive fold challenge and running
/// sumcheck claim.
const FOLD_FIELD_LOG_Q: f64 = 256.0;
/// OOD list binding is a 128-bit base-field component.
const OOD_BINDING_TARGET_BITS: f64 = 128.0;
/// The correlated-agreement/proximity term is evaluated over F256 and must
/// independently be strictly below 2^-128 in every shipped profile. The
/// legacy profile target still controls its query schedule, not this floor.
const MCA_TARGET_BITS: f64 = 128.0;
/// Algebraic error terms in Appendix C.3 of "Flock: Fast Proving for Batch
/// Boolean Computations" are required to be strictly below 2^-128,
/// independently of the legacy MCA target.
const ALGEBRAIC_TARGET_BITS: f64 = 128.0;
/// Every Johnson consistency-query component, including optional query-phase
/// grinding, must have a work-normalized error below 2^-128.
const LIST_DECODING_QUERY_TARGET_BITS: f64 = 128.0;

/// Round a float to one decimal place. Used to round paper-predicted
/// soundness diagnostics so the generated TOMLs stay readable.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Smallest integer lambda >= 0 for which `2^(log2_numerator-lambda) < 1`.
/// The strict inequality matters when the numerator is an exact power of two.
fn strict_grinding_bits(log2_numerator: f64) -> usize {
    if log2_numerator < 0.0 {
        0
    } else {
        log2_numerator.floor() as usize + 1
    }
}

/// Smallest positive Q such that
/// `Q * per_query_bits + grinding_bits > target_bits`.
///
/// The floor-plus-one form deliberately implements a strict inequality, even
/// when the quotient lands exactly on an integer boundary.
fn strict_query_count(target_bits: f64, grinding_bits: usize, per_query_bits: f64) -> usize {
    debug_assert!(per_query_bits > 0.0);
    let remaining = (target_bits - grinding_bits as f64).max(0.0);
    (remaining / per_query_bits).floor() as usize + 1
}

/// Return `(claim_bits, sumcheck_bits, consistency_bits, raw diagnostics)` for
/// the Flock paper's Appendix C.3 algebraic terms. `eta = None` denotes the
/// UDR list `L=1`.
fn algebraic_grinding_schedule(
    log_inv_rate: usize,
    eta: Option<f64>,
    queries: usize,
) -> (usize, usize, usize, f64, f64, f64) {
    let log2_l = eta
        .map(|eta| johnson_interleaved_list_log2(log_inv_rate, eta))
        .unwrap_or(0.0);
    let consistency_degree = ceil_log2(queries);
    let log2_consistency_degree = if consistency_degree <= 1 {
        0.0
    } else {
        (consistency_degree as f64).log2()
    };
    let claim_log_num = log2_l;
    let sumcheck_log_num = 1.0 + log2_l;
    let consistency_log_num = log2_l + log2_consistency_degree;
    let sumcheck_bits = FOLD_FIELD_LOG_Q - sumcheck_log_num;
    (
        strict_grinding_bits(claim_log_num),
        strict_grinding_bits(ALGEBRAIC_TARGET_BITS - sumcheck_bits),
        strict_grinding_bits(consistency_log_num),
        BASE_FIELD_LOG_Q - claim_log_num,
        sumcheck_bits,
        BASE_FIELD_LOG_Q - consistency_log_num,
    )
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
/// length**, so larger witnesses give a smaller `eps_pg = 256 − log₂ a`.
/// The shipped F256 shapes still remain well above the component target.
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

/// OOD collision-binding bits for a `JohnsonOod` level. Explicit OOD samples
/// evaluate the packed polynomial, whose total degree is at most
/// `explicit_degree`. L0 additionally uses its ordinary ring-switched opening
/// as an implicit sample; that check may have the separate
/// `implicit_degree = explicit_degree + LOG_PACKING` because its basis also
/// depends on the seven-coordinate ring-switch point.
///
/// Schwartz--Zippel bounds each agreement by `degree / 2^128`. Independence
/// of the post-commit samples lets those factors multiply. Union bounding over
/// unordered pairs in the Johnson list gives
///
///   bits = sum_j (128 - log2 degree_j) - (2 log2 L_int - 1).
fn paper_ood_bits(
    log_inv_rate: usize,
    eta: f64,
    explicit_degree: usize,
    explicit_samples: usize,
    implicit_degree: Option<usize>,
) -> f64 {
    debug_assert!(explicit_samples + usize::from(implicit_degree.is_some()) >= 1);
    debug_assert!(explicit_degree >= 1);
    let log2_l = johnson_interleaved_list_log2(log_inv_rate, eta);
    let explicit_bits =
        explicit_samples as f64 * (BASE_FIELD_LOG_Q - (explicit_degree as f64).log2());
    let implicit_bits = implicit_degree
        .map(|degree| BASE_FIELD_LOG_Q - (degree as f64).log2())
        .unwrap_or(0.0);
    explicit_bits + implicit_bits - (2.0 * log2_l - 1.0)
}

impl LigeritoLevelConfig {
    /// Algebraic terms from the Flock paper's Appendix C.3, before grinding:
    /// `(claim batching, one sumcheck round, queried-consistency batching)`.
    fn paper_predicted_algebraic_bits(&self) -> (f64, f64, f64) {
        let log2_l = match self.regime {
            SoundnessRegime::JohnsonOod => johnson_interleaved_list_log2(
                self.log_inv_rate,
                self.eta.expect("JohnsonOod must have eta"),
            ),
            SoundnessRegime::Udr => 0.0,
        };
        let consistency_degree = ceil_log2(self.queries);
        let log2_consistency_degree = if consistency_degree <= 1 {
            0.0
        } else {
            (consistency_degree as f64).log2()
        };
        (
            BASE_FIELD_LOG_Q - log2_l,
            FOLD_FIELD_LOG_Q - (1.0 + log2_l),
            BASE_FIELD_LOG_Q - (log2_l + log2_consistency_degree),
        )
    }

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
                let eps_pg = FOLD_FIELD_LOG_Q - log_a;
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
                // row union is unnecessary. So eps_pg = 256 − log₂ a.
                let log_a =
                    paper_thm_1_4_log_a(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_pg = FOLD_FIELD_LOG_Q - log_a;
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
    pub fn paper_predicted_ood_bits(&self, is_l0: bool) -> Option<f64> {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                let mu = self.log_msg_cols + self.log_num_interleaved;
                Some(paper_ood_bits(
                    self.log_inv_rate,
                    eta,
                    mu,
                    self.ood_samples,
                    is_l0.then_some(mu + LOG_PACKING),
                ))
            }
            SoundnessRegime::Udr => None,
        }
    }
}

impl LigeritoSecurityConfig {
    /// Check that a named embedded profile carries the analysis identity that
    /// profile promises. Security classification is deliberately not inferred
    /// from a free-text substring: a `fast` file cannot silently opt into the
    /// `fast100` query floor by changing its metadata.
    fn validate_profile(&self, profile: LigeritoProfile) -> Result<(), String> {
        let expected = match profile {
            LigeritoProfile::Secure => "f256_split_no_row_union_over_ben_sasson_2025_cor_1_4",
            LigeritoProfile::Fast | LigeritoProfile::Slim => {
                "f256_split_johnson_two_point_ood_query128_c3_algebraic_row_union_over_bchks25_thm_4_6"
            }
            LigeritoProfile::Fast100 | LigeritoProfile::Slim100 => {
                "f256_split_johnson_two_point_ood_query100_c3_algebraic_row_union_over_bchks25_thm_4_6"
            }
        };
        if self.analysis_version != expected {
            return Err(format!(
                "profile {} requires analysis_version {expected:?}, got {:?}",
                profile.as_str(),
                self.analysis_version
            ));
        }
        Ok(())
    }

    /// Validate that the config is internally consistent and matches the
    /// declared analysis. Returns the first violation found, if any.
    pub fn validate(&self) -> Result<(), String> {
        let query100_analysis = match self.analysis_version.as_str() {
            "f256_split_no_row_union_over_ben_sasson_2025_cor_1_4"
            | "f256_split_johnson_two_point_ood_query128_c3_algebraic_row_union_over_bchks25_thm_4_6" => {
                false
            }
            "f256_split_johnson_two_point_ood_query100_c3_algebraic_row_union_over_bchks25_thm_4_6" => {
                true
            }
            other => return Err(format!("unrecognized analysis_version {other:?}")),
        };
        if self.log_n + 7 != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING (7) != m ({})",
                self.log_n, self.m
            ));
        }
        if self.field != "f256" {
            return Err(format!(
                "security config `field` must be f256 for correlated-agreement folds, got {:?}",
                self.field
            ));
        }

        // Reject a `hash` we do not implement here, so a bad spelling is caught
        // at config-load time rather than silently committing under SHA-256.
        self.merkle_hash()?;

        // Each recursive level spends one fold on its coordinate bit.
        let levels_recursive_sum: usize = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.k_recursive.saturating_sub(1))
            .sum();
        let yr_log_n = self.final_block.yr_log_n;
        if self.initial_k + levels_recursive_sum + yr_log_n != self.log_n {
            return Err(format!(
                "shape mismatch: initial_k ({}) + Σ(k_recursive - 1) ({}) + yr_log_n ({}) = {} ≠ log_n ({})",
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
            if i != 0 {
                dim_in += 1;
            }
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

            // OOD samples match regime: UDR has no list, so no OOD. In the
            // Johnson regime every commitment is bound at two independent
            // post-commit points. L0's opening evaluation is the first, so it
            // carries one additional explicit OOD sample; later levels carry
            // two explicit samples.
            match lv.regime {
                SoundnessRegime::Udr if lv.ood_samples != 0 => {
                    return Err(format!(
                        "L{i}: regime=udr but ood_samples={} (unique decoding \
                         has list size 1 — no OOD binding step exists)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i == 0 && lv.ood_samples != 1 => {
                    return Err(format!(
                        "L0: ood_samples={} but two-point binding requires one \
                         explicit sample in addition to the opening claim",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i > 0 && lv.ood_samples != 2 => {
                    return Err(format!(
                        "L{i}: two-point Johnson binding requires exactly two \
                         explicit OOD samples, got {}",
                        lv.ood_samples
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
                        .paper_predicted_ood_bits(i == 0)
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

            // Enforce the exact query formula for every regime. In
            // particular, Secure/UDR must not rely on the rounded one-decimal
            // diagnostic above: a declaration can round upward by a fraction
            // of a bit while the real query count remains under its floor.
            let exact_query_bits = q_pred + lv.grinding_bits as f64;
            if exact_query_bits < lv.target_security_bits as f64 {
                return Err(format!(
                    "L{i}: exact query soundness ({q_pred:.6} + {} grinding = \
                     {exact_query_bits:.6}) is below target {}",
                    lv.grinding_bits, lv.target_security_bits
                ));
            }

            let (claim_pred, sumcheck_pred, consistency_pred) = lv.paper_predicted_algebraic_bits();
            for (name, declared, predicted) in [
                (
                    "expected_eps_claim_batch_bits",
                    lv.expected_eps_claim_batch_bits,
                    claim_pred,
                ),
                (
                    "expected_eps_sumcheck_bits",
                    lv.expected_eps_sumcheck_bits,
                    sumcheck_pred,
                ),
                (
                    "expected_eps_consistency_batch_bits",
                    lv.expected_eps_consistency_batch_bits,
                    consistency_pred,
                ),
            ] {
                if (declared - predicted).abs() > PAPER_COMPAT_TOL_BITS {
                    return Err(format!(
                        "L{i}: {name} ({declared:.2}) doesn't match Appendix C.3 \
                         prediction ({predicted:.2}); tolerance +/-{:.2} bits",
                        PAPER_COMPAT_TOL_BITS
                    ));
                }
            }

            // The list-unioned algebraic terms from the Flock paper's
            // Appendix C.3 must be STRICTLY below 2^-128. Check the exact
            // (unrounded) numerators so a power-of-two boundary cannot pass by
            // rounding.
            let claim_log_numerator = ALGEBRAIC_TARGET_BITS - claim_pred;
            let sumcheck_log_numerator = ALGEBRAIC_TARGET_BITS - sumcheck_pred;
            let consistency_log_numerator = ALGEBRAIC_TARGET_BITS - consistency_pred;
            let required_claim = strict_grinding_bits(claim_log_numerator);
            let required_sumcheck = strict_grinding_bits(sumcheck_log_numerator);
            let required_consistency = strict_grinding_bits(consistency_log_numerator);
            if lv.claim_batch_grinding_bits < required_claim {
                return Err(format!(
                    "L{i}: claim_batch_grinding_bits ({}) < required ({required_claim})",
                    lv.claim_batch_grinding_bits
                ));
            }
            if lv.fold_grinding_bits < required_sumcheck {
                return Err(format!(
                    "L{i}: fold_grinding_bits ({}) < Appendix C.3 sumcheck \
                     requirement ({required_sumcheck})",
                    lv.fold_grinding_bits
                ));
            }
            if lv.consistency_batch_grinding_bits < required_consistency {
                return Err(format!(
                    "L{i}: consistency_batch_grinding_bits ({}) < required \
                     ({required_consistency})",
                    lv.consistency_batch_grinding_bits
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

            // Johnson/list-decoding consistency queries are an independently
            // 128-bit component. Use the exact prediction, not the one-decimal
            // TOML diagnostic, and require a strict bound:
            //   (1-gamma)^Q * 2^-lambda_query < 2^-128
            // iff Q*log2(1/(1-gamma)) + lambda_query > 128.
            if lv.regime == SoundnessRegime::JohnsonOod {
                // The Johnson query floor is an ANALYSIS property, and
                // `analysis_version` is the config's accounting
                // discriminator: strict 128 for the list-decoding
                // milestone configs, the profile's own 100 for the
                // Fast100 pre-list-decoding cost point.
                let floor = if query100_analysis {
                    self.target_security_bits as f64
                } else {
                    LIST_DECODING_QUERY_TARGET_BITS
                };
                let delivered = q_pred + lv.grinding_bits as f64;
                if delivered <= floor {
                    return Err(format!(
                        "L{i}: query soundness ({q_pred:.6} + {} grinding = \
                         {delivered:.6} bits) must be strictly above the \
                         {floor}-bit list-decoding target",
                        lv.grinding_bits
                    ));
                }
            }

            // The F256 proximity/MCA component is independently required to
            // be strictly below 2^-128. Use the exact prediction rather than
            // the rounded TOML diagnostic. The profile-local check below is
            // retained for compatibility profiles with a separate query
            // target.
            let delivered_pg = pg_pred + lv.fold_grinding_bits as f64;
            if delivered_pg <= MCA_TARGET_BITS {
                return Err(format!(
                    "L{i}: F256 MCA soundness ({pg_pred:.6} + {} grinding = \
                     {delivered_pg:.6} bits) must be strictly above \
                     {MCA_TARGET_BITS} bits",
                    lv.fold_grinding_bits
                ));
            }

            // Per-application proximity gap + fold-challenge grinding must
            // also reach the profile-local target. The pg bad event lives on
            // the fold challenges, so query-phase grinding does not apply.
            if lv.expected_eps_pg_bits + lv.fold_grinding_bits as f64 + 1e-3
                < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) + fold_grinding ({}) < target ({})",
                    lv.expected_eps_pg_bits, lv.fold_grinding_bits, lv.target_security_bits
                ));
            }

            // OOD binding is independently and strictly a 128-bit component.
            // Again use the exact formula, not its one-decimal diagnostic.
            if let Some(ood) = lv.paper_predicted_ood_bits(i == 0)
                && ood <= OOD_BINDING_TARGET_BITS
            {
                return Err(format!(
                    "L{i}: OOD soundness ({ood:.6} bits) must be strictly \
                     above {OOD_BINDING_TARGET_BITS}; increase ood_samples"
                ));
            }

            if lv.target_security_bits < self.target_security_bits {
                return Err(format!(
                    "L{i}: target_security_bits ({}) < config target ({})",
                    lv.target_security_bits, self.target_security_bits
                ));
            }

            // Recursive inputs include one coordinate bit added at their code
            // switch; subtracting all k rounds leaves the extension dimension.
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
    ///     (length-dependent), so `eps_pg = 256 − log₂(γ·n+1)` decreases with
    ///     witness size; any shortfall below target is made up by
    ///     `fold_grinding_bits` (query-phase `grinding_bits` stays 0).
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
            let eps_pg = FOLD_FIELD_LOG_Q - log_a;
            // Any pg shortfall is ground on the fold challenges (where the
            // pg bad event lives); 0 at the 100-bit target.
            let proximity_fold_grinding_bits =
                ((target_security_bits as f64) - eps_pg).ceil().max(0.0) as usize;
            let eps_query = queries as f64 * per_q;
            let (
                claim_batch_grinding_bits,
                sumcheck_grinding_bits,
                consistency_batch_grinding_bits,
                eps_claim_batch,
                eps_sumcheck,
                eps_consistency_batch,
            ) = algebraic_grinding_schedule(rate, None, queries);
            let fold_grinding_bits = proximity_fold_grinding_bits.max(sumcheck_grinding_bits);
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
                claim_batch_grinding_bits,
                consistency_batch_grinding_bits,
                ood_samples: 0,
                target_security_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: None,
                expected_eps_claim_batch_bits: round1(eps_claim_batch),
                expected_eps_sumcheck_bits: round1(eps_sumcheck),
                expected_eps_consistency_batch_bits: round1(eps_consistency_batch),
            });
        }
        // One recursive fold per level consumes its coordinate bit.
        let total_recursive: usize = prover.recursive_ks.iter().map(|&k| k - 1).sum();
        let yr_log_n = log_n - initial_k - total_recursive;
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits,
            analysis_version: "f256_split_no_row_union_over_ben_sasson_2025_cor_1_4".into(),
            field: "f256".into(),
            hash: "blake3".into(),
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
    /// - `Fast`:   JohnsonOod, rate 1/2, η = 0.02, 128-bit query component
    ///             and F256 MCA arithmetic.
    /// - `Slim`:   JohnsonOod, rate 1/4, η = 0.02, 16-bit query grinding at
    ///             every level, 128-bit combined query work factor, and F256
    ///             MCA arithmetic.
    /// - `Secure`: Udr, rate 1/2, ε* = 1e-3, 120 bits per round.
    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        // Ladder: 3 folds/level. Strict Fast/Slim climb the rate +2 per level
        // (the aggressive ladder they inherited from Fast128/Slim128 in the
        // 2026-08-27 consolidation; still < 3 folds, so codewords keep
        // shrinking). Fast100/Slim100/Secure keep the shipped +1/level.
        let rate_gain = match profile {
            LigeritoProfile::Slim | LigeritoProfile::Fast => 2,
            _ => 1,
        };
        Self::derive_profile_ladder(m, profile, 3, rate_gain)
    }

    /// [`Self::derive_profile`] with a tunable recursion ladder — fold
    /// `folds_per_level` original variables per recursive level and climb the
    /// rate by `rate_gain` per level (see [`derive_ladder_shape_tuned`]).
    /// `(3, 1)` is the shipped shape; larger values collapse the tail into
    /// fewer, higher-rate levels (WHIR-like). Experimental: used to measure
    /// whether an aggressive ladder shrinks the proof in F128.
    pub fn derive_profile_ladder(
        m: usize,
        profile: LigeritoProfile,
        folds_per_level: usize,
        rate_gain: usize,
    ) -> Result<Self, String> {
        /// Johnson slack below the Johnson radius, flat across levels.
        const JOHNSON_ETA: f64 = 0.02;
        let target_bits = profile.security_bits();
        let log_inv_rate = profile.log_inv_rate();
        // 16-bit query PoW: each level's grinding substitutes for 16 bits of
        // query soundness (strict_query_count subtracts it from the target),
        // cutting Σq ~12% at the Fast m32 leaf for ~2^16 hash trials per
        // level natively. Slim has shipped this since its first schedule;
        // Fast joined 2026-08-14 as Fast128 (Ron's call, the leanVM
        // comparison — its Johnson configs grind 16 bits at every round) and
        // took the `Fast` name in the 2026-08-27 consolidation. Fast100 stays
        // grind-free: it is the frozen historical cost point.
        let query_grind: usize = match profile {
            LigeritoProfile::Slim100 | LigeritoProfile::Slim | LigeritoProfile::Fast => 16,
            LigeritoProfile::Fast100 | LigeritoProfile::Secure => 0,
        };
        let query_target_bits = match profile {
            LigeritoProfile::Fast | LigeritoProfile::Slim => LIST_DECODING_QUERY_TARGET_BITS,
            // Fast100 IS Fast at the pre-list-decoding cost point: the
            // query term targets the profile's own 100 bits, reproducing
            // the schedule shipped before the strict-128 milestone.
            LigeritoProfile::Fast100 | LigeritoProfile::Slim100 | LigeritoProfile::Secure => {
                target_bits as f64
            }
        };
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        // `initial_k` (= L0 interleave; the committed row width is
        // content_words / 2^(log_n − initial_k)) is 6 except where noted.
        // m29 Fast AND Slim run initial_k = 5 (Ron, 2026-08-05): the
        // recursion node lands on dense_m 29 since the BLAKE3 Option-E
        // narrowing, and at initial_k 6 the identity log_msg_cols =
        // log_n − initial_k halves the column count vs m30 while content
        // shrank only ~21% — committed rows fatten 39 → 55 words and every
        // node proof grows ~60 KiB. initial_k 5 restores 2^17 columns
        // (rows ≈ 28-31 words). Soundness derives identically: Johnson
        // per-query bits depend on rate/η only, so per-level query counts
        // are unchanged; at the 2026-08-27 consolidation the Fast ladder
        // re-derives as rates 1/3/5/7/9 (Σq 435) and the Slim ladder as
        // 2/4/6/8/10 (Σq 279). Secure keeps 6 (unused by the recursion
        // track).
        // m28 joined 2026-08-05 when the transcript-v3 duplex pushed the
        // slim L1 recursion node under 2^21 words: initial_k = 4 keeps the
        // same 2^17 columns (cols = log_n − initial_k = 21 − 4).
        let initial_k = match (m, profile) {
            (
                29,
                LigeritoProfile::Fast100
                | LigeritoProfile::Fast
                | LigeritoProfile::Slim100
                | LigeritoProfile::Slim,
            ) => 5usize,
            (
                28,
                LigeritoProfile::Fast100
                | LigeritoProfile::Fast
                | LigeritoProfile::Slim100
                | LigeritoProfile::Slim,
            ) => 4usize,
            _ => 6usize,
        };

        // Length-agnostic per-query estimate for ladder-shape feasibility
        // (the per-level codeword length `n` is not known until the shape is
        // fixed). UDR uses the asymptotic γ = δ/2; the actual per-level config
        // below uses the n-aware `udr_per_query_bits`.
        let per_query_bits_feas = |rate: usize| -> f64 {
            match profile {
                LigeritoProfile::Secure => udr_per_query_bits_asymptotic(rate),
                LigeritoProfile::Fast100
                | LigeritoProfile::Fast
                | LigeritoProfile::Slim100
                | LigeritoProfile::Slim => paper_per_query_bits(rate, JOHNSON_ETA),
            }
        };

        // Shape derivation needs per-level query counts for block-length
        // feasibility before the level count (and hence the exact per-term
        // target) is known. Use a conservative target of query_target_bits + 5
        // (≥ log₂(3 terms · 10 levels)); the final counts are ≤ this.
        let t_feas = query_target_bits + 5.0;
        let queries_feas = |rate: usize| -> usize {
            strict_query_count(t_feas, query_grind, per_query_bits_feas(rate))
        };
        let shape = derive_ladder_shape_tuned(
            log_n,
            initial_k,
            log_inv_rate,
            folds_per_level,
            rate_gain,
            &queries_feas,
        )?;
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
                LigeritoProfile::Fast100
                | LigeritoProfile::Fast
                | LigeritoProfile::Slim100
                | LigeritoProfile::Slim => paper_per_query_bits(rate, JOHNSON_ETA),
            };
            let queries = strict_query_count(query_target_bits, query_grind, per_q);
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
                        FOLD_FIELD_LOG_Q - paper_thm_1_4_log_a(rate, cols, UDR_PROXIMITY_LOSS);
                    (
                        SoundnessRegime::Udr,
                        None,
                        Some(UDR_PROXIMITY_LOSS),
                        eps_pg,
                        0usize,
                        None,
                    )
                }
                LigeritoProfile::Fast100
                | LigeritoProfile::Fast
                | LigeritoProfile::Slim100
                | LigeritoProfile::Slim => {
                    let eps_pg =
                        FOLD_FIELD_LOG_Q - paper_johnson_log_a(rate, JOHNSON_ETA, cols, ilv);
                    let mu = cols + ilv;
                    // Two independent binding points at every commitment.
                    // L0's ordinary opening point is already post-commit and
                    // supplies the first; all later points are explicit.
                    let is_l0 = i == 0;
                    let ood_samples = if is_l0 { 1 } else { 2 };
                    let eps_ood = paper_ood_bits(
                        rate,
                        JOHNSON_ETA,
                        mu,
                        ood_samples,
                        is_l0.then_some(mu + LOG_PACKING),
                    );
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
            let proximity_fold_grinding_bits = (t - eps_pg).ceil().max(0.0) as usize;
            let (
                claim_batch_grinding_bits,
                sumcheck_grinding_bits,
                consistency_batch_grinding_bits,
                eps_claim_batch,
                eps_sumcheck,
                eps_consistency_batch,
            ) = algebraic_grinding_schedule(rate, eta, queries);
            // One fold challenge carries both bad events from the Flock
            // paper's Appendix C.3. A single PoW therefore protects both, at
            // the larger requirement.
            // F256 makes this zero for the shipped shapes; validation still
            // enforces the independent strict 128-bit MCA floor.
            let fold_grinding_bits = proximity_fold_grinding_bits.max(sumcheck_grinding_bits);

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
                claim_batch_grinding_bits,
                consistency_batch_grinding_bits,
                ood_samples,
                target_security_bits: target_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: eps_ood,
                expected_eps_claim_batch_bits: round1(eps_claim_batch),
                expected_eps_sumcheck_bits: round1(eps_sumcheck),
                expected_eps_consistency_batch_bits: round1(eps_consistency_batch),
            });
        }

        let analysis_version = match profile {
            LigeritoProfile::Secure => "f256_split_no_row_union_over_ben_sasson_2025_cor_1_4",
            LigeritoProfile::Fast | LigeritoProfile::Slim => {
                "f256_split_johnson_two_point_ood_query128_c3_algebraic_row_union_over_bchks25_thm_4_6"
            }
            // Same analysis as Fast; only the query term's target differs
            // (the profile's own 100 bits, the pre-list-decoding schedule).
            LigeritoProfile::Fast100 | LigeritoProfile::Slim100 => {
                "f256_split_johnson_two_point_ood_query100_c3_algebraic_row_union_over_bchks25_thm_4_6"
            }
        };
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits: target_bits,
            analysis_version: analysis_version.into(),
            field: "f256".into(),
            hash: "blake3".into(),
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
        let merkle_hash = self.merkle_hash()?;
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
        let claim_batch_grinding_bits: Vec<usize> = self
            .levels
            .iter()
            .map(|lv| lv.claim_batch_grinding_bits)
            .collect();
        let consistency_batch_grinding_bits: Vec<usize> = self
            .levels
            .iter()
            .map(|lv| lv.consistency_batch_grinding_bits)
            .collect();
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
            claim_batch_grinding_bits: claim_batch_grinding_bits.clone(),
            consistency_batch_grinding_bits: consistency_batch_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
            merkle_hash,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits,
            consistency_batch_grinding_bits,
            ood_samples,
            merkle_hash,
            stratified: vec![],
        }
        .with_default_stratified();
        Ok((prover, verifier))
    }

    /// The Merkle hash this config selects, parsed from its `hash` field.
    ///
    /// Errors on any spelling we do not implement rather than defaulting —
    /// a config asking for a hash that is not wired up must fail loudly, not
    /// silently produce SHA-256 proofs under a `hash = "…"` that says
    /// otherwise.
    pub fn merkle_hash(&self) -> Result<HashKind, String> {
        HashKind::parse(&self.hash).map_err(|e| format!("security config `hash`: {e}"))
    }
}

// ===================================================================
// Proof
// ===================================================================

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursiveProof {
    /// One row per query, each of `num_interleaved` F128 entries, in SAMPLE
    /// order (replacement sampling: a duplicate query is just a repeated
    /// row).
    pub opened_rows: Vec<Vec<F128>>,
    /// Per-query CAPPED Merkle paths, flat: `queries.len()` paths of
    /// `depth − cap_depth` siblings each, concatenated in sample order. A
    /// duplicate query repeats its path. Verified against the level's cap.
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalProof {
    /// Remaining polynomial sent in clear at the last recursive step.
    pub yr: Vec<F128>,
    /// Same flat per-query capped-path convention as [`RecursiveProof`].
    pub opened_rows: Vec<Vec<F128>>,
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LigeritoProof {
    /// The L0 commitment CAP: the `2^c₀` tree nodes at depth `c₀ =
    /// cap_depth(queries[0], d₀)` below the root. There is no root — the
    /// transcript absorbs the cap itself.
    pub initial_cap: Vec<Hash>,
    pub initial_proof: RecursiveProof,
    /// Per recursive level, that tree's cap (`2^cᵢ` nodes) — the prover's
    /// commit message for the level, absorbed in full.
    pub recursive_caps: Vec<Vec<Hash>>,
    pub recursive_proofs: Vec<RecursiveProof>,
    pub final_proof: FinalProof,
    pub sumcheck_transcript: Vec<SumcheckMessage>,
    /// Sumcheck messages whose running claim and fold challenges are in the
    /// quadratic extension. Empty for legacy base-field proofs.
    #[serde(default)]
    pub sumcheck_transcript_f256: Vec<SumcheckMessage256>,
    /// Per-level PoW nonces (one entry per query phase). When all
    /// `grinding_bits` are 0 (the default config), each entry is just 0
    /// and the verifier's PoW check is a no-op. `#[serde(default)]` keeps
    /// older serialized proofs that pre-date this field readable.
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
    /// Claimed multilinear OOD evaluations, flattened in transcript order
    /// (L0's additional `ood_samples[0]` values, then level 1's, ...). Empty
    /// when the config takes no OOD samples (UDR profiles, legacy paths).
    #[serde(default)]
    pub ood_values: Vec<F128>,
    /// Fold-challenge PoW nonces, flattened in transcript order — one per
    /// fold challenge at every level with `fold_grinding_bits > 0`. Empty
    /// when no level fold-grinds.
    #[serde(default)]
    pub fold_grinding_nonces: Vec<u64>,
    /// Scalar claim-batching PoW nonces, flattened in transcript order: OOD
    /// batching coefficients and one recursive-consistency glue coefficient
    /// per level.
    #[serde(default)]
    pub claim_batch_grinding_nonces: Vec<u64>,
    /// One PoW nonce per level for the multilinear challenge that batches the
    /// queried consistency equations.
    #[serde(default)]
    pub consistency_batch_grinding_nonces: Vec<u64>,
}

impl LigeritoProof {
    pub fn size_bytes(&self) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        let level_bytes = |p: &RecursiveProof| -> usize {
            p.opened_rows.iter().map(|r| r.len() * ELEM).sum::<usize>() + p.merkle_proof.len() * 32
        };
        let mut total = self.initial_cap.len() * 32;
        total += self
            .recursive_caps
            .iter()
            .map(|c| c.len() * 32)
            .sum::<usize>();
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
        total += self.sumcheck_transcript_f256.len() * 4 * ELEM;
        total += self.ood_values.len() * ELEM;
        total += (self.grinding_nonces.len()
            + self.fold_grinding_nonces.len()
            + self.claim_batch_grinding_nonces.len()
            + self.consistency_batch_grinding_nonces.len())
            * 8;
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

        let roots_b = 32
            * (self.initial_cap.len() + self.recursive_caps.iter().map(|c| c.len()).sum::<usize>());
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
        let tx_base_b = self.sumcheck_transcript.len() * 2 * ELEM;
        let tx_f256_b = self.sumcheck_transcript_f256.len() * 4 * ELEM;
        let tx_b = tx_base_b + tx_f256_b;
        eprintln!(
            "  TOTALS: roots={}  opened={}  merkle={}  yr={}  transcript={} \
             (base {}×2×{}B + F256 {}×4×{}B)  GRAND={}",
            kb(roots_b),
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(tx_b),
            self.sumcheck_transcript.len(),
            ELEM,
            self.sumcheck_transcript_f256.len(),
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
pub fn eval_sk_at_vks(log_n: usize) -> Vec<F128> {
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
/// Compute just the `enforced_sum` half of [`induce_sumcheck_poly`]:
///   `enforced_sum = Σ_i eq(α, i_bin) · ⟨opened_rows[i], eq(v_challenges, ·)⟩`
/// Cheap: O(num_queries × num_interleaved). Verifier needs this at level
/// intro time (before residual challenges are known).
#[cfg(test)]
pub(crate) fn induce_sumcheck_enforced_sum(
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> F128 {
    assert_eq!(opened_rows.len(), queries.len());
    let eq = build_eq_table(v_challenges);
    // Rows may be NARROWER than the lane-fold weights under a high-bit-lane
    // commit (lanes past `t` are definitionally zero and never committed);
    // the `zip` below truncates, which is exactly the zero-fill.
    debug_assert!(opened_rows.iter().all(|r| r.len() <= eq.len()));
    let n_queries = queries.len();
    let alpha_weights: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        build_eq_table(alpha).into_iter().take(n_queries).collect()
    };
    let mut sum = F128::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
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

#[cfg(test)]
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
    //
    // `queries` is a multiset — sampling is with replacement — so two slots
    // `i ≠ i'` may share a position and merge into that position's single
    // constraint with combined weight `eq(α,i) + eq(α,i')`. The batching
    // argument is indifferent: a violated position still yields a nonzero
    // entry of the violation vector at both `i` and `i'`, so the S-Z term
    // stays `⌈log₂ Q⌉/2^128`. What a duplicate does cost is one fewer
    // *independent* position, and that is already priced into the query count
    // (see [`udr_queries`]).
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
pub fn induce_sumcheck_poly(
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
    let eq = build_eq_table(v_challenges); // length 2^v_challenges.len() = num_interleaved
    // Rows may be narrower than `eq` under a high-bit-lane commit — see
    // `induce_sumcheck_enforced_sum`.
    debug_assert!(opened_rows.iter().all(|r| r.len() <= eq.len()));

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
pub fn induce_sumcheck_poly_via_ntt(
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
pub struct LigeroWitness {
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

    /// The cap layer at depth `c` — this witness's commitment message.
    #[inline]
    pub fn cap(&self, c: usize) -> &[Hash] {
        merkle::cap_layer(&self.tree, self.block_len, c)
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
pub fn ligero_commit(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
    kind: HashKind,
) -> LigeroWitness {
    let msg_cols = 1usize << log_msg_cols;
    let num_interleaved = 1usize << log_num_interleaved;
    let block_len = msg_cols << log_inv_rate;
    let log_block_len = log_msg_cols + log_inv_rate;
    assert_eq!(poly.len(), num_interleaved * msg_cols);
    assert!(log_block_len <= ntt.log_domain_size());

    // LSB-lane layout: input matches the SoA layout `data[pos * num_interleaved + lane]`
    // directly. The first `log_inv_rate` NTT layers on the zero-padded
    // coefficients are pure copies, so the encode starts past those layers
    // with `poly` replicated `2^log_inv_rate` times.
    //
    // Fill fusion (`from_message`, which sources the first pass's rows from
    // `poly` and skips the replicate write) was tried here and measured
    // SLOWER — see the 2026-08-31 log entry. These levels run at rate 1/8
    // .. 1/2048, where the replicate is a cheap 1->2^r broadcast, unlike the
    // rate-1/2 L0 shape the fused pass is tuned for.
    let codeword_len = block_len * num_interleaved;
    let mut mat = crate::scratch::take_f128(codeword_len);

    // RS-encode every lane in one call (each lane is one independent NTT).
    let lig_timing = std::env::var_os("FLOCK_LIG_TIMING").is_some();
    let t_enc = std::time::Instant::now();
    super::commit::replicate_message_fill(&mut mat, poly);
    ntt.forward_transform_interleaved_from_layer(&mut mat, num_interleaved, log_inv_rate);
    let t_enc = t_enc.elapsed();

    // Merkle over rows. One leaf = `num_interleaved` consecutive F128 = 16·num_interleaved bytes.
    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    let t_mk = std::time::Instant::now();
    let tree = merkle::merkle_tree(data_bytes, block_len, kind);
    if lig_timing {
        eprintln!(
            "[lig-timing] ligero_commit log_cols={log_msg_cols} lanes={num_interleaved} rate=1/{}: encode {:.2} ms + merkle {:.2} ms",
            1usize << log_inv_rate,
            t_enc.as_secs_f64() * 1e3,
            t_mk.elapsed().as_secs_f64() * 1e3
        );
    }

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

/// `(u_0, u_2)` for a sumcheck whose challenges and running claim live in the
/// quadratic extension field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckMessage256 {
    pub u_0: F256,
    pub u_2: F256,
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

/// Block-pairing counterpart of [`round_msg_and_eval_lsb`]. `d = 1` is the
/// ordinary LSB order; `d > 1` pairs corresponding entries in adjacent
/// `d`-word blocks, as required by a lane-major L0 fold.
fn round_msg_and_eval_blocked(f: &[F128], b: &[F128], d: usize) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    debug_assert!(d.is_power_of_two());
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(2 * d));
    if d == 1 {
        return round_msg_and_eval_lsb(f, b);
    }
    let (u_0, u_2, y) = (0..f.len() / (2 * d))
        .into_par_iter()
        .map(|j| {
            let (mut u0, mut u2, mut eval) = (F128::ZERO, F128::ZERO, F128::ZERO);
            let b0 = 2 * j * d;
            let b1 = b0 + d;
            for k in 0..d {
                let (f0, f1) = (f[b0 + k], f[b1 + k]);
                let (e0, e1) = (b[b0 + k], b[b1 + k]);
                let f0e0 = f0 * e0;
                u0 += f0e0;
                u2 += (f0 + f1) * (e0 + e1);
                eval += f0e0 + f1 * e1;
            }
            (u0, u2, eval)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ae), (b0, b2, be)| (a0 + b0, a2 + b2, ae + be),
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
    use crate::field::F256Unreduced;
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;

    // Fold with ONE mul per output via `v0 + r·(v0 + v1)` (= `(1+r)·v0 + r·v1`
    // exactly in GF(2^128)), and accumulate the message as unreduced 256-bit
    // products, reduced once at the end — reduction is XOR-linear, so the
    // result is bit-identical to the reduced-per-term sum.
    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        let mut nf = Vec::with_capacity(half);
        let mut nb = Vec::with_capacity(half);
        for j in 0..half {
            let f0 = f[2 * j];
            let b0 = b[2 * j];
            nf.push(f0 + (f0 + f[2 * j + 1]) * r);
            nb.push(b0 + (b0 + b[2 * j + 1]) * r);
        }
        let mut u_0 = F256Unreduced::ZERO;
        let mut u_2 = F256Unreduced::ZERO;
        let mut k = 0;
        while k + 1 < half {
            let f0 = nf[k];
            let f1 = nf[k + 1];
            let b0 = nb[k];
            let b1 = nb[k + 1];
            u_0 ^= f0.mul_unreduced(b0);
            u_2 ^= (f0 + f1).mul_unreduced(b0 + b1);
            k += 2;
        }
        return (
            nf,
            nb,
            SumcheckMessage {
                u_0: u_0.reduce(),
                u_2: u_2.reduce(),
            },
        );
    }

    // Parallel path: `half` is a power of two ≥ PAR_THRESHOLD and CHUNK is a
    // power of two, so every chunk has even length and starts at an even
    // global index — message pairs (2k, 2k+1) never straddle a chunk boundary.
    //
    // Output buffers come from the scratch pool: a fresh 64 MB alloc per fold
    // round pays first-touch page faults inside the parallel loop and a
    // single-threaded munmap on drop of the old buffer — measured as the
    // dominant cost of the open's initial sumcheck (~18 ms → ~7 ms at m=30
    // for the 6-fold chain once both buffers recycle through the pool; the
    // caller returns the outgoing pair via `scratch::give_f128`, see
    // [`SumcheckProver::fold`]).
    const CHUNK: usize = 2048;
    let mut nf = crate::scratch::take_f128(half);
    let mut nb = crate::scratch::take_f128(half);
    let (u_0, u_2) = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            let len = fc.len();
            let mut u0 = F256Unreduced::ZERO;
            let mut u2 = F256Unreduced::ZERO;
            // Fold this slice via the arch-dispatched slice kernel (AVX-512 on
            // x86, NEON on aarch64), then pair up the just-folded values for
            // the msg. `src[2j]·(1+r) + src[2j+1]·r = v0 + (v0+v1)·r` exactly
            // in GF(2^128), so the kernel choice cannot change the bits.
            crate::field::f128_slice::fold_pairs(f, base, fc, r);
            crate::field::f128_slice::fold_pairs(b, base, bc, r);
            let mut k = 0;
            while k + 1 < len {
                let f0 = fc[k];
                let f1 = fc[k + 1];
                let b0 = bc[k];
                let b1 = bc[k + 1];
                u0 ^= f0.mul_unreduced(b0);
                u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
                k += 2;
            }
            (u0, u2)
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(a0, a2), (c0, c2)| (a0 ^ c0, a2 ^ c2),
        );
    (
        nf,
        nb,
        SumcheckMessage {
            u_0: u_0.reduce(),
            u_2: u_2.reduce(),
        },
    )
}

/// Quadratic coefficients of the NEXT round's message as a polynomial in the
/// not-yet-sampled fold challenge `r`:
/// `u_0(r) = u0[0] + r·u0[1] + r²·u0[2]` (same for `u_2`).
///
/// Produced by the lookahead fold passes ([`fold1_lookahead_lsb`] /
/// [`fold2_lookahead_lsb`]) so the round in between two passes costs an O(1)
/// polynomial evaluation instead of a full array pass. The evaluated message
/// equals the direct `round_msg_lsb` over the folded arrays by exact
/// polynomial identity (every op is exact in GF(2^128)), so the transcript is
/// bit-identical.
#[derive(Clone, Copy, Debug)]
pub struct FoldLookahead {
    u0: [F128; 3],
    u2: [F128; 3],
}

/// In-process A/B override for the F256 ladder's WHOLE alternating fold
/// schedule — the round-1 lookahead skip AND every mid-ladder `La256`
/// production + skip round it seeds, through the initial folds and all
/// recursive levels: `0` follows the `FLOCK_NO_FOLD_LOOKAHEAD` env knob,
/// `1` forces the schedule on, `2` forces the plain per-round folds.
/// Byte-identical either way (exact polynomial identity) — the oracle
/// tests alternate this to prove it.
pub static FOLD_LOOKAHEAD_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

impl FoldLookahead {
    #[inline]
    fn eval(&self, r: F128) -> SumcheckMessage {
        let r2 = r * r;
        SumcheckMessage {
            u_0: self.u0[0] + r * self.u0[1] + r2 * self.u0[2],
            u_2: self.u2[0] + r * self.u2[1] + r2 * self.u2[2],
        }
    }

    /// Componentwise addition. The coefficients are LINEAR in the basis, so
    /// a post-combine basis delta corrects a stored lookahead by adding the
    /// delta's own coefficients.
    pub(crate) fn add(&mut self, other: &FoldLookahead) {
        for i in 0..3 {
            self.u0[i] += other.u0[i];
            self.u2[i] += other.u2[i];
        }
    }
}

/// Shared per-group-of-4 kernel for the lookahead passes: given 4 freshly
/// folded consecutive outputs of each array (still in registers), accumulate
/// - this round's message contribution (pairs `(0,1)` and `(2,3)`), and
/// - the quadratic coefficients of the NEXT round's message in the future
///   challenge `r'` (next fold pairs `(0,1)`→slot 0 and `(2,3)`→slot 1; next
///   message pairs those two slots).
///
/// Accumulator layout: `[u0, u2, c0, c1, c2, d0, d1, d2]` where `c*` are the
/// `u_0(r')` coefficients and `d*` the `u_2(r')` coefficients.
///
/// 8 unreduced muls per group (Karatsuba middle terms: the `r'`-linear
/// coefficient is `m1 + m2 + (A+dA)(B+dB)` — exact in char 2).
#[inline(always)]
pub(crate) fn lookahead_accum_group(fq: &[F128; 4], bq: &[F128; 4], acc: &mut [F256Unreduced; 8]) {
    let a0 = fq[0];
    let da0 = fq[0] + fq[1];
    let a1 = fq[2];
    let da1 = fq[2] + fq[3];
    let b0 = bq[0];
    let db0 = bq[0] + bq[1];
    let b1 = bq[2];
    let db1 = bq[2] + bq[3];

    let m1 = a0.mul_unreduced(b0);
    let m2 = da0.mul_unreduced(db0);
    let m3 = (a0 + da0).mul_unreduced(b0 + db0);
    let p1 = a1.mul_unreduced(b1);
    let p2 = da1.mul_unreduced(db1);
    let n1 = (a0 + a1).mul_unreduced(b0 + b1);
    let n2 = (da0 + da1).mul_unreduced(db0 + db1);
    let n3 = (a0 + a1 + da0 + da1).mul_unreduced(b0 + b1 + db0 + db1);

    // This round's message: pairs (0,1) and (2,3).
    acc[0] ^= m1 ^ p1; // u_0 += A0·B0 + A1·B1
    acc[1] ^= m2 ^ p2; // u_2 += dA0·dB0 + dA1·dB1
    // Next round's u_0(r') = Σ (A0 + r'·dA0)(B0 + r'·dB0).
    acc[2] ^= m1;
    acc[3] ^= m1 ^ m2 ^ m3;
    acc[4] ^= m2;
    // Next round's u_2(r') = Σ (S + r'·dS)(T + r'·dT), S = A0+A1 etc.
    acc[5] ^= n1;
    acc[6] ^= n1 ^ n2 ^ n3;
    acc[7] ^= n2;
}

/// Reduce the 8 lookahead accumulators into `(msg, coeffs)`.
#[inline]
pub(crate) fn lookahead_finish(acc: [F256Unreduced; 8]) -> (SumcheckMessage, FoldLookahead) {
    (
        SumcheckMessage {
            u_0: acc[0].reduce(),
            u_2: acc[1].reduce(),
        },
        FoldLookahead {
            u0: [acc[2].reduce(), acc[3].reduce(), acc[4].reduce()],
            u2: [acc[5].reduce(), acc[6].reduce(), acc[7].reduce()],
        },
    )
}

#[inline]
pub(crate) fn xor_acc8(mut a: [F256Unreduced; 8], b: [F256Unreduced; 8]) -> [F256Unreduced; 8] {
    for k in 0..8 {
        a[k] ^= b[k];
    }
    a
}

/// [`round_msg_and_eval_lsb`] that ALSO accumulates the round-1 message's
/// quadratic coefficients in the round-0 fold challenge (the entry-pass use
/// of [`lookahead_accum_group`], plus one odd-dot accumulator for the full
/// evaluation `y = Σ f·b`). One sweep, LSB pairing (`d = 1`).
///
/// The F256 ladder's L0 OOD loop uses this to keep a caller-provided
/// [`FoldLookahead`] LIVE across the OOD β-glues: the coefficients are
/// linear in the basis, so `b += β·eq_z` corrects the lookahead by
/// `β · (coefficients of (f, eq_z))` — which this returns.
pub(crate) fn round_msg_eval_and_lookahead(
    f: &[F128],
    b: &[F128],
) -> (SumcheckMessage, F128, FoldLookahead) {
    use rayon::prelude::*;
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len() >= 4 && f.len().is_multiple_of(4));
    // 9 accumulators: the 8 lookahead slots + the odd dot (y = u_0 + odd).
    let group =
        |fq: &[F128; 4], bq: &[F128; 4], acc: &mut [F256Unreduced; 8], odd: &mut F256Unreduced| {
            lookahead_accum_group(fq, bq, acc);
            *odd ^= fq[1].mul_unreduced(bq[1]) ^ fq[3].mul_unreduced(bq[3]);
        };
    const CHUNK: usize = 1 << 12;
    let (acc, odd) = f
        .par_chunks(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .map(|(fc, bc)| {
            let mut acc = [F256Unreduced::ZERO; 8];
            let mut odd = F256Unreduced::ZERO;
            for (fq, bq) in fc.as_chunks::<4>().0.iter().zip(bc.as_chunks::<4>().0) {
                group(fq, bq, &mut acc, &mut odd);
            }
            (acc, odd)
        })
        .reduce(
            || ([F256Unreduced::ZERO; 8], F256Unreduced::ZERO),
            |(a, ao), (b, bo)| {
                (xor_acc8(a, b), {
                    let mut o = ao;
                    o ^= bo;
                    o
                })
            },
        );
    let (msg, la) = lookahead_finish(acc);
    (msg, msg.u_0 + odd.reduce(), la)
}

/// Entry lookahead pass: fold ONE variable (`n → n/2`) and return this
/// round's message plus the next round's [`FoldLookahead`] coefficients.
/// Identical fold/message values to [`fold_and_msg_lsb`]; the extra
/// coefficients cost ~1 extra mul per output in the same pass.
fn fold1_lookahead_lsb(
    f: &[F128],
    b: &[F128],
    r: F128,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, FoldLookahead) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    assert!(
        n.is_power_of_two() && half >= 4 && half.is_multiple_of(4),
        "fold1_lookahead_lsb: need n ≥ 8"
    );

    const CHUNK: usize = 2048;
    let mut nf = crate::scratch::take_f128(half);
    let mut nb = crate::scratch::take_f128(half);
    let acc = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            let len = fc.len();
            debug_assert!(len.is_multiple_of(4) || len == half - base);
            let mut acc = [F256Unreduced::ZERO; 8];
            // Fold this slice (1 mul per output per array)…
            for t in 0..len {
                let j = base + t;
                let f0 = f[2 * j];
                let f1 = f[2 * j + 1];
                let b0 = b[2 * j];
                let b1 = b[2 * j + 1];
                fc[t] = f0 + (f0 + f1) * r;
                bc[t] = b0 + (b0 + b1) * r;
            }
            // …then message + lookahead over groups of 4 just-written outputs.
            let mut g = 0;
            while g + 4 <= len {
                let fq = [fc[g], fc[g + 1], fc[g + 2], fc[g + 3]];
                let bq = [bc[g], bc[g + 1], bc[g + 2], bc[g + 3]];
                lookahead_accum_group(&fq, &bq, &mut acc);
                g += 4;
            }
            acc
        })
        .reduce(|| [F256Unreduced::ZERO; 8], xor_acc8);
    let (msg, la) = lookahead_finish(acc);
    (nf, nb, msg, la)
}

/// Steady-state lookahead pass: fold TWO variables (`n → n/4`, challenges
/// `r_a` then `r_b`) and return the message after both folds plus the next
/// round's [`FoldLookahead`]. Values are bit-identical to two sequential
/// [`fold_and_msg_lsb`] rounds, at ~55% of their memory traffic (the
/// intermediate half-size arrays are never materialized).
fn fold2_lookahead_lsb(
    f: &[F128],
    b: &[F128],
    r_a: F128,
    r_b: F128,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, FoldLookahead) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert_eq!(b.len(), n);
    let quarter = n / 4;
    assert!(
        n.is_power_of_two() && quarter >= 4 && quarter.is_multiple_of(4),
        "fold2_lookahead_lsb: need n ≥ 16"
    );

    const CHUNK: usize = 2048;
    let mut nf = crate::scratch::take_f128(quarter);
    let mut nb = crate::scratch::take_f128(quarter);
    let acc = nf
        .par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            let len = fc.len();
            let mut acc = [F256Unreduced::ZERO; 8];
            // Fold 4→1 (3 muls per output per array), 4 outputs per group.
            let mut g = 0;
            while g < len {
                let glen = (len - g).min(4);
                let mut fq = [F128::ZERO; 4];
                let mut bq = [F128::ZERO; 4];
                for t in 0..glen {
                    let j = base + g + t;
                    let i = 4 * j;
                    let gf0 = f[i] + (f[i] + f[i + 1]) * r_a;
                    let gf1 = f[i + 2] + (f[i + 2] + f[i + 3]) * r_a;
                    let gb0 = b[i] + (b[i] + b[i + 1]) * r_a;
                    let gb1 = b[i + 2] + (b[i + 2] + b[i + 3]) * r_a;
                    let vf = gf0 + (gf0 + gf1) * r_b;
                    let vb = gb0 + (gb0 + gb1) * r_b;
                    fc[g + t] = vf;
                    bc[g + t] = vb;
                    fq[t] = vf;
                    bq[t] = vb;
                }
                debug_assert_eq!(glen, 4);
                lookahead_accum_group(&fq, &bq, &mut acc);
                g += glen;
            }
            acc
        })
        .reduce(|| [F256Unreduced::ZERO; 8], xor_acc8);
    let (msg, la) = lookahead_finish(acc);
    (nf, nb, msg, la)
}

/// Fold both arrays by `r` with NO message computation (write-only drain for
/// a pending lookahead challenge at the end of an odd-length schedule).
fn fold_pair_no_msg(f: &[F128], b: &[F128], r: F128) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    let mut nf = crate::scratch::take_f128(half);
    let mut nb = crate::scratch::take_f128(half);
    const CHUNK: usize = 2048;
    nf.par_chunks_mut(CHUNK)
        .zip(nb.par_chunks_mut(CHUNK))
        .enumerate()
        .for_each(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            for t in 0..fc.len() {
                let j = base + t;
                let f0 = f[2 * j];
                let f1 = f[2 * j + 1];
                let b0 = b[2 * j];
                let b1 = b[2 * j + 1];
                fc[t] = f0 + (f0 + f1) * r;
                bc[t] = b0 + (b0 + b1) * r;
            }
        });
    (nf, nb)
}

/// Fused **blocked** fold + next-round message: view `f`/`b` as blocks of `d`
/// and bind the LOW bit of the BLOCK index, so output block `c` combines input
/// blocks `2c` and `2c+1`:
///
/// ```text
///   out[c·d + p] = a[2c·d + p] + r·(a[(2c+1)·d + p] + a[2c·d + p])
/// ```
///
/// `d = 1` is exactly [`fold_and_msg_lsb`] (and delegates to it, keeping the
/// tuned interleaved kernel on the power-of-two path).
///
/// **Why:** under the high-bit-lane commit the dense stack is LANE-MAJOR —
/// lane `l` is the contiguous block `q[l·D .. (l+1)·D)` — so the lane variable
/// Ligerito must bind first sits in the HIGH index bits, not the low ones.
/// Folding at block granularity `d = D` binds exactly those bits, in the same
/// order and with the same eq weights as folding the rotated array
/// element-wise would. That makes the rotation a matter of ADDRESSING rather
/// than DATA: no `2^m`-word transpose of `q` and `W_ρ` (2 × 134 MB of traffic
/// plus two big allocations at `M = 30`), and the emitted proof is
/// byte-identical either way — see the merged open (`pcs::open_batch_merged`).
fn fold_and_msg_blocked(
    f: &[F128],
    b: &[F128],
    r: F128,
    d: usize,
    live_in: usize,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    if d == 1 {
        return fold_and_msg_lsb(f, b, r);
    }
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2 * d);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    let blocks_out = half / d;
    // Live blocks are always a PREFIX, so the live count simply halves
    // (rounded up) each round. Output block `c` reads input blocks `2c` and
    // `2c+1`: `2c` is live whenever `c < live_out`, and `2c+1` is live unless
    // `c` is the last live block of an odd-sized prefix.
    let live_in = live_in.min(2 * blocks_out).max(1);
    let live_out = live_in.div_ceil(2).min(blocks_out);
    let one_plus_r = F128::ONE + r;

    // Combine one aligned run: `out = lo + r·(hi + lo)`, with a DEAD `hi`
    // read as zero (`out = lo·(1+r)`) rather than touched — dead blocks are
    // never written, so their contents are stale, not zero.
    let combine = |out: &mut [F128], lo: &[F128], hi: Option<&[F128]>| match hi {
        Some(hi) => {
            for ((o, &l), &h) in out.iter_mut().zip(lo).zip(hi) {
                *o = l + r * (h + l);
            }
        }
        None => {
            for (o, &l) in out.iter_mut().zip(lo) {
                *o = l * one_plus_r;
            }
        }
    };
    // Source runs for output block `c`, offset `o`, length `len`.
    fn src(
        v: &[F128],
        d: usize,
        live_in: usize,
        c: usize,
        o: usize,
        len: usize,
    ) -> (&[F128], Option<&[F128]>) {
        let lo = &v[2 * c * d + o..2 * c * d + o + len];
        let hi = (2 * c + 1 < live_in).then(|| &v[(2 * c + 1) * d + o..(2 * c + 1) * d + o + len]);
        (lo, hi)
    }

    let mut nf = crate::scratch::take_f128(half);
    let mut nb = crate::scratch::take_f128(half);

    if blocks_out == 1 {
        // Last blocked round: one output block, so the NEXT round pairs
        // elements — fold, then take the ordinary LSB message over the result.
        const CH: usize = 2048;
        nf.par_chunks_mut(CH)
            .zip(nb.par_chunks_mut(CH))
            .enumerate()
            .for_each(|(ci, (fc, bc))| {
                let o = ci * CH;
                let len = fc.len();
                let (flo, fhi) = src(f, d, live_in, 0, o, len);
                let (blo, bhi) = src(b, d, live_in, 0, o, len);
                combine(fc, flo, fhi);
                combine(bc, blo, bhi);
            });
        let msg = round_msg_lsb(&nf, &nb);
        return (nf, nb, msg);
    }

    // Each task owns one PAIR of output blocks — the unit the next round's
    // message pairs over — so fold and message fuse into a single pass. The
    // inner nesting keeps the parallelism fine-grained even in late rounds,
    // where there are few block pairs but each is still large.
    //
    // Tasks past the live prefix are skipped ENTIRELY: their inputs are zero,
    // so their outputs and message terms are zero, and the next round knows
    // not to read them. That is the payoff of the high-bit-lane layout — the
    // committed stack's zero tail is whole lanes, so `2^initial_k − t` of the
    // L0 fold's blocks are known-zero up front (~36% of this round's work at
    // t = 37 of 64). The final round leaves `live_out == 1`, so nothing dead
    // survives into the L1 witness.
    const CH: usize = 2048;
    let live_tasks = live_out.div_ceil(2);
    let (u_0, u_2) = nf
        .par_chunks_mut(2 * d)
        .zip(nb.par_chunks_mut(2 * d))
        .enumerate()
        .take(live_tasks)
        .map(|(q, (nfc, nbc))| {
            let (nf0, nf1) = nfc.split_at_mut(d);
            let (nb0, nb1) = nbc.split_at_mut(d);
            // Output blocks 2q (always live here) and 2q+1 (may be past it).
            let odd_live = 2 * q + 1 < live_out;
            nf0.par_chunks_mut(CH)
                .zip(nf1.par_chunks_mut(CH))
                .zip(nb0.par_chunks_mut(CH))
                .zip(nb1.par_chunks_mut(CH))
                .enumerate()
                .map(|(ci, (((f0, f1), b0), b1))| {
                    let o = ci * CH;
                    let len = f0.len();
                    let (flo, fhi) = src(f, d, live_in, 2 * q, o, len);
                    let (blo, bhi) = src(b, d, live_in, 2 * q, o, len);
                    combine(f0, flo, fhi);
                    combine(b0, blo, bhi);
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    if odd_live {
                        let (flo, fhi) = src(f, d, live_in, 2 * q + 1, o, len);
                        let (blo, bhi) = src(b, d, live_in, 2 * q + 1, o, len);
                        combine(f1, flo, fhi);
                        combine(b1, blo, bhi);
                        for i in 0..len {
                            u0 += f0[i] * b0[i];
                            u2 += (f0[i] + f1[i]) * (b0[i] + b1[i]);
                        }
                    } else {
                        // Odd output block is dead => reads as zero in the
                        // next round's message: u2 term collapses to u0's.
                        for i in 0..len {
                            let t = f0[i] * b0[i];
                            u0 += t;
                            u2 += t;
                        }
                    }
                    (u0, u2)
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
                )
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    (nf, nb, SumcheckMessage { u_0, u_2 })
}

/// Fills a window of the basis: `out = b[g0 .. g0 + out.len()]`. Lets L0
/// source its basis from a compact factored form
/// (`jagged::fill_weight_range`) instead of a materialized `2^m` array.
pub type BasisWindowFn<'a> = &'a (dyn Fn(&mut [F128], usize) + Sync);

/// [`fold_and_msg_blocked`] with the basis supplied JUST-IN-TIME. Identical
/// arithmetic and identical output; the only difference is that each task
/// fills two small L1-resident windows of `b` rather than streaming them from
/// a `2^m` array — which at `m = 30` removes both the 134 MB materialization
/// and the 134 MB read of it, and measures FASTER than the read (measured
/// by the since-deleted `tests/jit_fold.rs` probe, bloat ledger §E).
///
/// `d = 1` is the ordinary adjacent/LSB pairing; `d > 1` is the blocked
/// pairing used by a lane-major L0 fold.  The same task decomposition covers
/// both cases: at `d = 1`, one task folds four adjacent input entries into
/// two output entries and accumulates exactly one next-round message pair.
// Test-only since the F128 `SumcheckProver::fold_blocked_jit` wrapper was
// deleted (bloat ledger §A): the production JIT fold is the F256 one in
// `extension`; the in-file jit-equivalence test keeps this as its oracle.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fold_and_msg_blocked_jit(
    f: &[F128],
    fill: BasisWindowFn<'_>,
    r: F128,
    d: usize,
    live_in: usize,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    let n = f.len();
    debug_assert!(d.is_power_of_two() && n.is_power_of_two() && n >= 2 * d);
    let half = n / 2;
    let blocks_out = half / d;
    let live_in = live_in.min(2 * blocks_out).max(1);
    let live_out = live_in.div_ceil(2).min(blocks_out);
    let one_plus_r = F128::ONE + r;

    let mut nf = crate::scratch::take_f128(half);
    let mut nb = crate::scratch::take_f128(half);
    const CH: usize = 1 << 11;

    // Combine one aligned run with an optionally-dead `hi`.
    let comb = |out: &mut [F128], lo: &[F128], hi: Option<&[F128]>| match hi {
        Some(hi) => {
            for ((o, &l), &h) in out.iter_mut().zip(lo).zip(hi) {
                *o = l + r * (h + l);
            }
        }
        None => {
            for (o, &l) in out.iter_mut().zip(lo) {
                *o = l * one_plus_r;
            }
        }
    };

    let live_tasks = live_out.div_ceil(2);
    let (u_0, u_2) = nf
        .par_chunks_mut(2 * d)
        .zip(nb.par_chunks_mut(2 * d))
        .enumerate()
        .take(live_tasks)
        .map(|(qq, (nfc, nbc))| {
            let (nf0, nf1) = nfc.split_at_mut(d);
            let (nb0, nb1) = nbc.split_at_mut(d);
            let odd_live = 2 * qq + 1 < live_out;
            nf0.par_chunks_mut(CH)
                .zip(nf1.par_chunks_mut(CH))
                .zip(nb0.par_chunks_mut(CH))
                .zip(nb1.par_chunks_mut(CH))
                .enumerate()
                // Scratch once per worker, not once per window.
                .map_init(
                    || (vec![F128::ZERO; CH], vec![F128::ZERO; CH]),
                    |(wlo, whi), (ci, (((f0, f1), b0), b1))| {
                        let o = ci * CH;
                        let len = f0.len();
                        let src_of = |c: usize| 2 * c * d + o;
                        let mut u0 = F256Unreduced::ZERO;
                        let mut u2 = F256Unreduced::ZERO;
                        let one_side = |c: usize,
                                        fo: &mut [F128],
                                        bo: &mut [F128],
                                        wlo: &mut [F128],
                                        whi: &mut [F128]| {
                            let g_lo = src_of(c);
                            let hi_live = 2 * c + 1 < live_in;
                            fill(&mut wlo[..len], g_lo);
                            let fs = &f[g_lo..g_lo + len];
                            if hi_live {
                                let g_hi = g_lo + d;
                                fill(&mut whi[..len], g_hi);
                                comb(fo, fs, Some(&f[g_hi..g_hi + len]));
                                comb(bo, &wlo[..len], Some(&whi[..len]));
                            } else {
                                comb(fo, fs, None);
                                comb(bo, &wlo[..len], None);
                            }
                        };
                        one_side(2 * qq, f0, b0, wlo, whi);
                        if odd_live {
                            one_side(2 * qq + 1, f1, b1, wlo, whi);
                            for i in 0..len {
                                u0 ^= f0[i].mul_unreduced(b0[i]);
                                u2 ^= (f0[i] + f1[i]).mul_unreduced(b0[i] + b1[i]);
                            }
                        } else {
                            for i in 0..len {
                                let t = f0[i].mul_unreduced(b0[i]);
                                u0 ^= t;
                                u2 ^= t;
                            }
                        }
                        (u0.reduce(), u2.reduce())
                    },
                )
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
                )
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    (nf, nb, SumcheckMessage { u_0, u_2 })
}

/// The merged transport's inner-open basis kept FACTORED across ALL of L0's
/// folds. `b[u] = γ·eq(ρ, u)` is a rank-1 tensor, and binding one index bit
/// keeps it rank-1 — folding bit `p` at challenge `r` drops coordinate `p`
/// and scales the whole basis by `(1 + ρ_p + r)`:
///
/// ```text
///   b'[u'] = (1+r)·b|_{u_p=0} + r·b|_{u_p=1}
///          = [(1+r)(1+ρ_p) + r·ρ_p] · Π_{j≠p} f_j(u_j)
///          = (1 + ρ_p + r) · Π_{j≠p} f_j(u_j)            (char 2)
/// ```
///
/// So no L0 round needs a materialized basis: each round rebuilds two √L eq
/// tables (a few thousand words, L1-resident) from the surviving coordinates
/// and fills the message's b-windows from them. That removes the half-size
/// basis WRITE of every round and its READ in the next — the b-side is ~40%
/// of the initial sumcheck's traffic — at the price of one multiply per
/// b-word (`lo[i]·e_hi`, the hi factor hoisted per run), the same trade the
/// round-0 JIT fill ([`fold_and_msg_blocked_jit`]) already measured as a win.
/// Value-identical BY CONSTRUCTION: same b values, same message arithmetic.
///
/// The basis is materialized exactly once, at the LAST L0 round (where
/// `blocks_out == 1` and the next round pairs elements anyway), at the
/// handoff size `2^(log_n − initial_k)` the recursion needs.
struct VirtualEqTerm {
    /// Surviving point coordinates; coordinate `j` ↔ index bit `j` (the
    /// [`build_eq_table`] convention the caller's tensor was built under).
    coords: Vec<F128>,
    /// γ times every fold factor absorbed so far. Baked into `lo`.
    scale: F128,
    lo: Vec<F128>,
    hi: Vec<F128>,
    n_lo: usize,
}

impl VirtualEqTerm {
    fn new(point: Vec<F128>, gamma: F128) -> Self {
        let mut term = Self {
            coords: point,
            scale: gamma,
            lo: Vec::new(),
            hi: Vec::new(),
            n_lo: 0,
        };
        term.rebuild();
        term
    }

    fn rebuild(&mut self) {
        self.n_lo = self.coords.len() / 2;
        self.lo = crate::pcs::ring_switch::build_eq_scaled_parallel(
            &self.coords[..self.n_lo],
            self.scale,
        );
        self.hi =
            crate::pcs::ring_switch::build_eq_scaled_parallel(&self.coords[self.n_lo..], F128::ONE);
    }

    #[inline]
    fn value_at(&self, u: usize) -> F128 {
        let mask = (1usize << self.n_lo) - 1;
        self.lo[u & mask] * self.hi[u >> self.n_lo]
    }

    fn add_to(&self, out: &mut [F128], g0: usize) {
        let span = 1usize << self.n_lo;
        let mask = span - 1;
        let mut i = 0;
        while i < out.len() {
            let u = g0 + i;
            let e_hi = self.hi[u >> self.n_lo];
            let off = u & mask;
            let n = (out.len() - i).min(span - off);
            for (k, s) in out[i..i + n].iter_mut().enumerate() {
                *s += self.lo[off + k] * e_hi;
            }
            i += n;
        }
    }
}

/// A factored sum of scaled equality tensors. It starts with the opening's
/// basis and can absorb L0's additional OOD term without materializing a
/// `2^log_n` vector.
pub(crate) struct VirtualEqBasis {
    terms: Vec<VirtualEqTerm>,
}

impl VirtualEqBasis {
    pub(crate) fn new(point: Vec<F128>, gamma: F128) -> Self {
        Self {
            terms: vec![VirtualEqTerm::new(point, gamma)],
        }
    }

    fn add_term(&mut self, point: Vec<F128>, scale: F128) {
        assert_eq!(point.len(), self.terms[0].coords.len());
        self.terms.push(VirtualEqTerm::new(point, scale));
    }

    fn add_to(&self, out: &mut [F128], g0: usize) {
        for term in &self.terms {
            term.add_to(out, g0);
        }
    }
}

/// Round-0 message and MLE evaluation for an unscaled equality tensor,
/// retaining its square-root factorization throughout the pass.
fn round_msg_and_eval_eq_point_blocked(
    f: &[F128],
    point: &[F128],
    d: usize,
) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    debug_assert_eq!(f.len(), 1usize << point.len());
    debug_assert!(f.len().is_multiple_of(2 * d));
    let eq = VirtualEqTerm::new(point.to_vec(), F128::ONE);
    let (u_0, u_2, y) = (0..f.len() / (2 * d))
        .into_par_iter()
        .map(|j| {
            let (mut u0, mut u2, mut eval) = (F128::ZERO, F128::ZERO, F128::ZERO);
            let i0 = 2 * j * d;
            let i1 = i0 + d;
            for k in 0..d {
                let (f0, f1) = (f[i0 + k], f[i1 + k]);
                let (e0, e1) = (eq.value_at(i0 + k), eq.value_at(i1 + k));
                let f0e0 = f0 * e0;
                u0 += f0e0;
                u2 += (f0 + f1) * (e0 + e1);
                eval += f0e0 + f1 * e1;
            }
            (u0, u2, eval)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ae), (b0, b2, be)| (a0 + b0, a2 + b2, ae + be),
        );
    (SumcheckMessage { u_0, u_2 }, y)
}

/// [`round_msg_and_eval_eq_point_blocked`] that ALSO accumulates the round-1
/// message's quadratic coefficients in the round-0 fold challenge, under the
/// same BLOCK pairing (`d = 1` is the LSB order). Quad `b` covers the four
/// consecutive `d`-blocks `[4bd, 4bd+4d)`: fold 0 pairs blocks (0,1) and
/// (2,3), fold 1 pairs the two results — exactly the fused fold kernels'
/// geometry, so [`lookahead_accum_group`] applies verbatim with blocked
/// gathering. Used by the F256 ladder's factored L0 OOD loop to keep a
/// caller lookahead live across the β-glues.
pub(crate) fn round_msg_eval_and_lookahead_eq_point_blocked(
    f: &[F128],
    point: &[F128],
    d: usize,
) -> (SumcheckMessage, F128, FoldLookahead) {
    use rayon::prelude::*;
    debug_assert_eq!(f.len(), 1usize << point.len());
    debug_assert!(d.is_power_of_two() && f.len().is_multiple_of(4 * d));
    let eq = VirtualEqTerm::new(point.to_vec(), F128::ONE);
    let (acc, odd) = (0..f.len() / (4 * d))
        .into_par_iter()
        .map(|q| {
            let mut acc = [F256Unreduced::ZERO; 8];
            let mut odd = F256Unreduced::ZERO;
            let i0 = 4 * q * d;
            for k in 0..d {
                let i = i0 + k;
                let fq = [f[i], f[i + d], f[i + 2 * d], f[i + 3 * d]];
                let bq = [
                    eq.value_at(i),
                    eq.value_at(i + d),
                    eq.value_at(i + 2 * d),
                    eq.value_at(i + 3 * d),
                ];
                lookahead_accum_group(&fq, &bq, &mut acc);
                odd ^= fq[1].mul_unreduced(bq[1]) ^ fq[3].mul_unreduced(bq[3]);
            }
            (acc, odd)
        })
        .reduce(
            || ([F256Unreduced::ZERO; 8], F256Unreduced::ZERO),
            |(a, ao), (b, bo)| {
                (xor_acc8(a, b), {
                    let mut o = ao;
                    o ^= bo;
                    o
                })
            },
        );
    let (msg, la) = lookahead_finish(acc);
    (msg, msg.u_0 + odd.reduce(), la)
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
    /// Lookahead bookkeeping: a fold challenge whose array fold is deferred
    /// to the next lookahead pass (the round's message was already produced
    /// by [`FoldLookahead::eval`]). Must be `None` (drained) before any
    /// non-lookahead operation touches `f`/`combined_basis`.
    pending_fold: Option<F128>,
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
            pending_fold: None,
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
            pending_fold: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        self.fold_blocked(r, 1, usize::MAX)
    }

    /// [`Self::fold`] binding the low bit of the BLOCK index for block size
    /// `d` (`d = 1` is [`Self::fold`] itself) — see [`fold_and_msg_blocked`].
    /// Used for L0's lane folds under a lane-major (high-bit-lane) witness.
    /// `live_in` is how many leading blocks are not known-zero.
    pub fn fold_blocked(&mut self, r: F128, d: usize, live_in: usize) -> SumcheckMessage {
        debug_assert!(self.pending_fold.is_none(), "fold with pending lookahead");
        // Fused: fold f and combined_basis at r AND build the next-round
        // message in one parallel pass (was three passes). See
        // [`fold_and_msg_lsb`].
        let (nf, nb, msg) = fold_and_msg_blocked(&self.f, &self.combined_basis, r, d, live_in);
        // On x86_64, recycle the just-consumed buffers into the scratch pool
        // (same ownership as the Drop impl) so the next round's
        // `fold_and_msg_lsb` takes resident pages. aarch64 measured slower with
        // this pooling, so there we just move the new buffers in and drop the
        // old ones.
        #[cfg(target_arch = "x86_64")]
        {
            crate::scratch::give_f128(std::mem::replace(&mut self.f, nf));
            crate::scratch::give_f128(std::mem::replace(&mut self.combined_basis, nb));
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.f = nf;
            self.combined_basis = nb;
        }
        self.transcript.push(msg);
        msg
    }

    /// Lookahead entry: like [`Self::fold`] but also returns the next round's
    /// message coefficients (see [`FoldLookahead`]). Same fold + message
    /// values as `fold(r)`.
    pub fn fold1_lookahead(&mut self, r: F128) -> (SumcheckMessage, FoldLookahead) {
        debug_assert!(self.pending_fold.is_none(), "fold1 with pending lookahead");
        let (nf, nb, msg, la) = fold1_lookahead_lsb(&self.f, &self.combined_basis, r);
        crate::scratch::give_f128(std::mem::replace(&mut self.f, nf));
        crate::scratch::give_f128(std::mem::replace(&mut self.combined_basis, nb));
        self.transcript.push(msg);
        (msg, la)
    }

    /// Lookahead skip round: produce this round's message by evaluating the
    /// previous pass's coefficients at `r` — O(1), no array pass. The actual
    /// fold by `r` is deferred to the next [`Self::fold2_lookahead`] (or
    /// [`Self::drain_pending_fold`]). The message value is identical to
    /// `fold(r)`'s by exact polynomial identity.
    pub fn fold_skip(&mut self, la: &FoldLookahead, r: F128) -> SumcheckMessage {
        debug_assert!(self.pending_fold.is_none(), "double lookahead skip");
        let msg = la.eval(r);
        self.pending_fold = Some(r);
        self.transcript.push(msg);
        msg
    }

    /// Lookahead steady state: fold the pending challenge AND `r` in one
    /// 4→1 pass, returning the post-fold message + next coefficients.
    pub fn fold2_lookahead(&mut self, r: F128) -> (SumcheckMessage, FoldLookahead) {
        let r_a = self
            .pending_fold
            .take()
            .expect("fold2_lookahead without pending challenge");
        let (nf, nb, msg, la) = fold2_lookahead_lsb(&self.f, &self.combined_basis, r_a, r);
        crate::scratch::give_f128(std::mem::replace(&mut self.f, nf));
        crate::scratch::give_f128(std::mem::replace(&mut self.combined_basis, nb));
        self.transcript.push(msg);
        (msg, la)
    }

    /// Materialize a pending lookahead fold (message was already sent by
    /// [`Self::fold_skip`]). No-op when nothing is pending.
    pub fn drain_pending_fold(&mut self) {
        if let Some(r) = self.pending_fold.take() {
            let (nf, nb) = fold_pair_no_msg(&self.f, &self.combined_basis, r);
            crate::scratch::give_f128(std::mem::replace(&mut self.f, nf));
            crate::scratch::give_f128(std::mem::replace(&mut self.combined_basis, nb));
        }
    }

    /// Introduce a fresh basis poly with claimed sum `h_new`. Sends the
    /// (u_0, u_2) for `Σ_x f(x) · b_new(x)` at the current dim.
    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        debug_assert!(self.pending_fold.is_none(), "introduce with pending fold");
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
        debug_assert!(self.pending_fold.is_none(), "introduce with pending fold");
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
        debug_assert!(self.pending_fold.is_none(), "f() with pending fold");
        &self.f
    }

    pub fn has_pending_fold(&self) -> bool {
        self.pending_fold.is_some()
    }

    pub fn transcript(&self) -> &[SumcheckMessage] {
        &self.transcript
    }
}

// ===================================================================
// Prover / Verifier — stubs
// ===================================================================

/// Sample `count` positions in `[0, block_len)` via the challenger, **with
/// replacement**: exactly `count` draws, duplicates allowed, returned in
/// sample order (unsorted).
///
/// Shaped for arithmetisation. The old sampler rejected repeats, which cost a
/// `HashSet`, a sort, and an unbounded loop — none of which a circuit can
/// express. All three are gone: this is one fixed-length squeeze followed by a
/// low-bit mask per element, a fully static shape.
///
/// The single `sample_f128_vec` squeeze also matters for the FS chain the
/// recursive verifier has to replay. Per-element `sample_f128` calls are
/// `count` sequentially-dependent duplex rounds (absorb tag, finalize, absorb
/// the 16 squeezed bytes); the batched form is one finalize, one XOF fill —
/// counter-mode, so its blocks are mutually independent — and one bulk
/// re-absorb. At L0 (243 queries) that trades 243 finalizations for **one**,
/// and 243 XOF output blocks for **61**: the old path squeezed 16 bytes at a
/// time and threw away 48 of every 64. Measured over a whole verification
/// (`--features hash-count`, `verifier_hash_count`), the finalization count is
/// where the entire win lands — m=30 rate 2 goes 363 → 106, while absorb
/// blocks and XOF output barely move. Finalizations are also the part that
/// actually serializes, since each squeeze's output is re-absorbed.
///
/// `block_len` is a power of two (it is `2^(log_msg_cols + log_inv_rate)`), so
/// masking the low bits is exact and unbiased — no modular reduction and no
/// rejection needed for uniformity either.
///
/// `count` may exceed `block_len` without harm; the soundness bound
/// (see [`udr_queries`]) is the independent-sample one and never depended on
/// distinctness. Ladder shapes still keep `block_len >= count` — see
/// [`derive_ladder_shape_tuned`] — but that is now a proof-size convention rather
/// than a correctness requirement.
fn sample_queries<Ch: Challenger>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
    sched: &stratified::LevelSchedule,
) -> Vec<usize> {
    assert!(
        block_len.is_power_of_two(),
        "sample_queries: block_len ({block_len}) must be a power of two"
    );
    if count == 0 {
        return Vec::new();
    }
    // STRATIFIED (docs/stratified-queries.tex): one F128 squeezed per query
    // — but query j is confined to its schedule-constant stratum: the low
    // `d − c_j` bits come from the squeeze, the top `c_j` bits ARE the
    // stratum index.
    let d = block_len.trailing_zeros() as usize;
    assert_eq!(sched.log_block_len, d, "sample_queries: schedule block log");
    assert_eq!(
        sched.queries(),
        count,
        "sample_queries: schedule query count"
    );
    let words = challenger.sample_f128_vec(count);
    queries_from_words(block_len, count, sched, &words)
}

fn queries_from_words(
    block_len: usize,
    count: usize,
    sched: &stratified::LevelSchedule,
    words: &[F128],
) -> Vec<usize> {
    let d = block_len.trailing_zeros() as usize;
    assert_eq!(words.len(), count, "one challenge word per query");
    sched
        .query_strata()
        .zip(words)
        .map(|((c, stratum), v)| {
            let lo_bits = d - c;
            let mask = (1usize << lo_bits) - 1;
            (stratum << lo_bits) | ((v.lo as usize) & mask)
        })
        .collect()
}

fn grind_and_sample_queries<Ch: Challenger>(
    challenger: &mut Ch,
    bits: u32,
    block_len: usize,
    count: usize,
    sched: &stratified::LevelSchedule,
) -> (u64, Vec<usize>) {
    assert!(
        count != 0,
        "a grinded query phase must sample at least one query"
    );
    let (nonce, words) = challenger.grind_pow_and_sample_f128_vec(bits, count);
    (nonce, queries_from_words(block_len, count, sched, &words))
}

fn verify_and_sample_queries<Ch: Challenger>(
    challenger: &mut Ch,
    nonce: u64,
    bits: u32,
    block_len: usize,
    count: usize,
    sched: &stratified::LevelSchedule,
) -> Option<Vec<usize>> {
    let words = challenger.verify_pow_and_sample_f128_vec(nonce, bits, count)?;
    Some(queries_from_words(block_len, count, sched, &words))
}

/// Per-query CAPPED Merkle paths for `queries` against `tree`, flat in
/// sample order: every path stops at the schedule's cap depth `c_1`, so
/// paths are uniformly `d − c_1` siblings (`total_path_siblings` in all).
/// A shallower summand's remaining `c_1 − c_j` levels are folds of the
/// absorbed cap — verifier-derivable, never emitted. Duplicates repeat
/// their path — no sorting, no dedup.
fn merkle_paths_for(
    tree: &[Hash],
    block_len: usize,
    queries: &[usize],
    sched: &stratified::LevelSchedule,
) -> Vec<Hash> {
    let d = block_len.trailing_zeros() as usize;
    assert_eq!(
        sched.log_block_len, d,
        "merkle_paths_for: schedule block log"
    );
    assert_eq!(
        sched.queries(),
        queries.len(),
        "merkle_paths_for: query count"
    );
    let c1 = sched.cap_depth();
    let mut out = Vec::with_capacity(sched.total_path_siblings());
    for &q in queries {
        out.extend(merkle::merkle_proof_capped(tree, block_len, q, c1));
    }
    out
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
    let wtns_0 = ligero_commit(
        poly,
        log_msg_cols_0,
        initial_k,
        log_inv_rate_0,
        &ntt_0,
        config.merkle_hash,
    );
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

/// Drop-in replacement for the legacy `basefold::prove`: takes a generic basis poly +
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
    extension::recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        1usize << config.initial_k,
        false,
        None,
        None,
        None,
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
    round1_lookahead: Option<FoldLookahead>,
    challenger: &mut Ch,
) -> LigeritoProof {
    extension::recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        1usize << config.initial_k,
        false,
        None,
        None,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        round1_lookahead,
        challenger,
    )
}

/// Lane-aware variant of [`recursive_prover_with_basis_precomputed_round0`]
/// for the merged transport's inner open: the commitment may be lane-major
/// (integer lanes, high-bit blocks), the basis is VIRTUAL (`l0_virtual_basis`
/// — factored across every L0 fold), JIT for the first fold only
/// (`l0_jit_basis`), or MATERIALIZED, and — unlike
/// the jagged fused entry — the live-block skip is DISABLED
/// (`l0_live_blocks = full`): the eq basis is nonzero on the zero-padding
/// lanes, so the b-side must be folded honestly there (the f-side terms are
/// zero regardless, since q's dead lanes are zero). No trailing verifier
/// mirror: the transcript ends at the opening on this path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_precomputed_round0_lanes<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    l0_num_lanes: usize,
    l0_lane_major: bool,
    round0_uv: (F128, F128),
    round1_lookahead: Option<FoldLookahead>,
    l0_jit_basis: Option<BasisWindowFn<'_>>,
    l0_virtual_basis: Option<VirtualEqBasis>,
    challenger: &mut Ch,
) -> LigeritoProof {
    extension::recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        l0_num_lanes,
        l0_lane_major,
        l0_jit_basis,
        l0_virtual_basis,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        round1_lookahead,
        challenger,
    )
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
    // The legacy (non-basis) path predates OOD binding and all round grinding;
    // configs that use them must go through `recursive_prover_with_basis`.
    assert!(
        config.ood_samples.iter().all(|&s| s == 0)
            && config.fold_grinding_bits.iter().all(|&b| b == 0)
            && config.claim_batch_grinding_bits.iter().all(|&b| b == 0)
            && config
                .consistency_batch_grinding_bits
                .iter()
                .all(|&b| b == 0),
        "OOD samples / round grinding require the with_basis prover path"
    );
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];

    // Config-static cap depths (the legacy/UDR path derives its query
    // counts from `udr_queries`, so the caps do too).
    let cap_depth_of = |level: usize| -> usize { config.stratified[level].cap_depth() };
    let strat = |l: usize| &config.stratified[l];
    config
        .validate_stratified()
        .expect("stratified schedules invalid (prover entry)");
    let c_0 = cap_depth_of(0);
    let initial_cap: Vec<Hash> = wtns_0.cap(c_0).to_vec();
    challenger.observe_bytes(initial_cap.as_flattened());

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
        config.merkle_hash,
    );
    let t_l1 = t.elapsed();
    t_commits += t_l1;
    tlog!("  [ligerito]   L1 commit: {:.2?}", t_l1);
    challenger.observe_bytes(wtns_1.cap(cap_depth_of(1)).as_flattened());

    // ---- Queries + open wtns_0 ----
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_queries(challenger, wtns_0.block_len, num_queries_0, strat(0));
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_paths_for(&wtns_0.tree, wtns_0.block_len, &queries_0, strat(0));
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
    let mut recursive_caps: Vec<Vec<Hash>> = vec![wtns_prev.cap(cap_depth_of(1)).to_vec()];
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
            let queries_last = sample_queries(
                challenger,
                wtns_prev.block_len,
                num_queries_last,
                strat(i + 1),
            );
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last = merkle_paths_for(
                &wtns_prev.tree,
                wtns_prev.block_len,
                &queries_last,
                strat(i + 1),
            );
            return LigeritoProof {
                initial_cap,
                initial_proof,
                recursive_caps,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                sumcheck_transcript_f256: Vec::new(),
                grinding_nonces: Vec::new(), // legacy recursive_prover_inner: no grinding plumbed
                ood_values: Vec::new(),
                fold_grinding_nonces: Vec::new(),
                claim_batch_grinding_nonces: Vec::new(),
                consistency_batch_grinding_nonces: Vec::new(),
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
            config.merkle_hash,
        );
        let t_li = t.elapsed();
        t_commits += t_li;
        tlog!("  [ligerito]   L{} commit: {:.2?}", i + 2, t_li);
        let cap_next = wtns_next.cap(cap_depth_of(i + 2)).to_vec();
        challenger.observe_bytes(cap_next.as_flattened());
        recursive_caps.push(cap_next);

        // Open wtns_prev. wtns_prev = wtns_{i+1} uses log_inv_rates[i+1].
        let num_queries_i = udr_queries(config.log_inv_rates[i + 1]);
        let queries_i =
            sample_queries(challenger, wtns_prev.block_len, num_queries_i, strat(i + 1));
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i = merkle_paths_for(
            &wtns_prev.tree,
            wtns_prev.block_len,
            &queries_i,
            strat(i + 1),
        );
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

/// Check the level's opened rows against its CAP: one independent capped
/// Merkle path per slot, in sample order — no sorting, no dedup, and no
/// repeated-position consistency check. A duplicate query simply repeats
/// its path, and a prover answering one position with two DIFFERENT rows
/// fails outright: the tree has one leaf there, so at most one of the rows
/// can fold to the transcript-fixed cap node. The cap's size is
/// config-static (`cap_depth(queries.len(), d)` — `queries.len()` IS the
/// config count, the verifier sampled them itself), so a wrong-size cap or
/// a wrong-length flat path vector rejects on shape before any hashing.
fn verify_level_opens(
    cap: &[Hash],
    block_len: usize,
    queries: &[usize],
    opened_rows: &[Vec<F128>],
    expected_num_interleaved: usize,
    paths: &[Hash],
    kind: HashKind,
    sched: &stratified::LevelSchedule,
) -> bool {
    if queries.len() != opened_rows.len() {
        return false;
    }
    let d = block_len.trailing_zeros() as usize;
    let leaf_of = |row: &Vec<F128>| {
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                row.as_ptr() as *const u8,
                row.len() * core::mem::size_of::<F128>(),
            )
        };
        merkle::hash_leaf(bytes, kind)
    };
    let s = sched;

    // STRATIFIED: the absorbed cap sits at the schedule's cap depth (the
    // top summand), and every path truncates there — all nodes above the
    // cap are its own folds, so siblings past `c_1` could never carry
    // evidence. Each query verifies with the ordinary capped walk against
    // the cap itself; stratum membership needs no enforcement here because
    // the verifier derived the index (`sample_queries` puts the stratum in
    // the top bits), and the compare pins all `c_1 ≥ c_j` of them.
    if s.log_block_len != d || s.queries() != queries.len() {
        return false;
    }
    let c1 = s.cap_depth();
    if cap.len() != (1 << c1) {
        return false;
    }
    if paths.len() != s.total_path_siblings() {
        return false;
    }
    let path_len = d - c1;
    let mut off = 0usize;
    for (&q, row) in queries.iter().zip(opened_rows) {
        if row.len() != expected_num_interleaved {
            return false;
        }
        let leaf = leaf_of(row);
        if !merkle::verify_merkle_proof_capped(
            cap,
            block_len,
            &leaf,
            q,
            &paths[off..][..path_len],
            kind,
        ) {
            return false;
        }
        off += path_len;
    }
    true
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
    // The legacy (non-basis) path predates OOD binding and round grinding.
    if config.ood_samples.iter().any(|&s| s != 0)
        || config.fold_grinding_bits.iter().any(|&b| b != 0)
        || config.claim_batch_grinding_bits.iter().any(|&b| b != 0)
        || config
            .consistency_batch_grinding_bits
            .iter()
            .any(|&b| b != 0)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    // ---- Roots ----
    challenger.observe_bytes(proof.initial_cap.as_flattened());
    if proof.recursive_caps.len() != r {
        return false;
    }
    let cap_1: &[Hash] = &proof.recursive_caps[0];
    challenger.observe_bytes(cap_1.as_flattened());

    // ---- Open wtns_0 + α₀ ----
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    let strat = |l: usize| &config.stratified[l];
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_queries(challenger, block_len_0, num_queries_0, strat(0));
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));

    if !verify_level_opens(
        &proof.initial_cap,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
        strat(0),
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
    let mut prev_cap: &[Hash] = cap_1;
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
                sample_queries(challenger, prev_block_len, num_queries_last, strat(i + 1));
            // Final-level basis-induction challenge (after yr + queries fixed).
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                prev_cap,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
                strat(i + 1),
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
        if next_root_idx >= proof.recursive_caps.len() {
            return false;
        }
        let cap_next: &[Hash] = &proof.recursive_caps[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(cap_next.as_flattened());

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = udr_queries(prev_log_inv_rate);
        let queries_i = sample_queries(challenger, prev_block_len, num_queries_i, strat(i + 1));
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));

        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            prev_cap,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
            strat(i + 1),
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
        prev_cap = cap_next;
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

    /// The factored EqPoint opening uses the JIT basis on both commitment
    /// layouts.  A full/power-of-two lane grid has `d = 1`, so pin that the
    /// JIT fold is byte-for-byte identical to the ordinary materialized LSB
    /// fold: folded witness, folded basis, and the next sumcheck message.
    #[test]
    fn jit_fold_matches_materialized_lsb_pairing() {
        use crate::challenger::Challenger;

        for log_n in [4usize, 5, 10, 13] {
            let n = 1usize << log_n;
            let mut rng = crate::challenger::RandomChallenger::new(
                0xD100_0000_u64.wrapping_add(log_n as u64),
            );
            let f: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
            let r = rng.sample_f128();
            let fill = |out: &mut [F128], g0: usize| {
                out.copy_from_slice(&b[g0..g0 + out.len()]);
            };

            let (want_f, want_b, want_msg) = fold_and_msg_blocked(&f, &b, r, 1, usize::MAX);
            let (got_f, got_b, got_msg) = fold_and_msg_blocked_jit(&f, &fill, r, 1, usize::MAX);

            assert_eq!(got_f, want_f, "folded witness at log_n={log_n}");
            assert_eq!(got_b, want_b, "folded basis at log_n={log_n}");
            assert_eq!(got_msg, want_msg, "next message at log_n={log_n}");
        }
    }

    /// L0 OOD batching computes its first message directly from a factored
    /// equality tensor. Pin both natural and lane-major block pairings against
    /// the materialized equality-table oracle, including the claimed MLE.
    #[test]
    fn factored_ood_round_message_matches_materialized() {
        use crate::challenger::Challenger;

        let log_n = 10usize;
        let n = 1usize << log_n;
        let mut rng = crate::challenger::RandomChallenger::new(0x00D0_0D00);
        let f: Vec<F128> = (0..n).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        for d in [1usize, 1 << 6] {
            let factored = round_msg_and_eval_eq_point_blocked(&f, &z, d);
            let dense = round_msg_and_eval_blocked(&f, &eq, d);
            assert_eq!(factored, dense, "round-0 OOD batch at block size {d}");
        }
    }

    #[test]
    fn l0_ood_accounting_includes_the_ring_switch_degree() {
        let cfg = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m22_fast.toml"
        ))
        .expect("m22 Fast config validates");
        let l0 = &cfg.levels[0];
        let eta = l0.eta.expect("Fast uses Johnson OOD");
        let mu = l0.log_msg_cols + l0.log_num_interleaved;

        let conservative = l0
            .paper_predicted_ood_bits(true)
            .expect("Johnson OOD has a prediction");
        let treating_both_points_as_degree_mu =
            paper_ood_bits(l0.log_inv_rate, eta, mu, l0.ood_samples + 1, None);
        let expected_loss = ((mu + LOG_PACKING) as f64 / mu as f64).log2();
        assert!(conservative < treating_both_points_as_degree_mu);
        assert!(((treating_both_points_as_degree_mu - conservative) - expected_loss).abs() < 1e-10);

        let l1 = &cfg.levels[1];
        let mu1 = l1.log_msg_cols + l1.log_num_interleaved;
        assert_eq!(
            l1.paper_predicted_ood_bits(false),
            Some(paper_ood_bits(
                l1.log_inv_rate,
                l1.eta.expect("Fast uses Johnson OOD"),
                mu1,
                l1.ood_samples,
                None,
            ))
        );
    }

    /// Lookahead fold state machine vs the plain per-round fused folds:
    /// every round message AND the final arrays must be bit-identical
    /// (exact polynomial identity). Covers even k (drain needed) and odd k.
    #[test]
    fn lookahead_folds_match_plain_folds() {
        let mut s = 0xFACE_FEED_0123_4567u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let n = 1usize << 12;
        let f: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let b: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let target = F128 {
            lo: next(),
            hi: next(),
        };
        let first = SumcheckMessage {
            u_0: F128 {
                lo: next(),
                hi: next(),
            },
            u_2: F128 {
                lo: next(),
                hi: next(),
            },
        };
        for k in [5usize, 6] {
            let rs: Vec<F128> = (0..k)
                .map(|_| F128 {
                    lo: next(),
                    hi: next(),
                })
                .collect();

            let (mut plain, _) =
                SumcheckProver::new_with_first_msg(f.clone(), b.clone(), target, first);
            let plain_msgs: Vec<SumcheckMessage> = rs.iter().map(|&r| plain.fold(r)).collect();

            let (mut la_prover, _) =
                SumcheckProver::new_with_first_msg(f.clone(), b.clone(), target, first);
            let mut lookahead: Option<FoldLookahead> = None;
            let mut la_msgs = Vec::with_capacity(k);
            for &r in &rs {
                let msg = if let Some(la) = lookahead.take() {
                    la_prover.fold_skip(&la, r)
                } else if la_prover.has_pending_fold() {
                    let (msg, la) = la_prover.fold2_lookahead(r);
                    lookahead = Some(la);
                    msg
                } else {
                    let (msg, la) = la_prover.fold1_lookahead(r);
                    lookahead = Some(la);
                    msg
                };
                la_msgs.push(msg);
            }
            la_prover.drain_pending_fold();

            for (j, (pm, lm)) in plain_msgs.iter().zip(la_msgs.iter()).enumerate() {
                assert_eq!(pm.u_0, lm.u_0, "k={k} round {j}: u_0 mismatch");
                assert_eq!(pm.u_2, lm.u_2, "k={k} round {j}: u_2 mismatch");
            }
            assert_eq!(plain.f(), la_prover.f(), "k={k}: folded f mismatch");
            assert_eq!(
                plain.combined_basis, la_prover.combined_basis,
                "k={k}: folded basis mismatch"
            );
        }
    }

    /// Worked example: `LigeritoSecurityConfig` for BLAKE3 m=29 at rate 1/2.
    /// Paper-compatible m=29 fast example, mechanically derived in the
    /// unique-decoding regime (Theorem 1.4, ε* = 10⁻³) targeting 100-bit
    /// security.
    fn blake3_m29_udr_example() -> LigeritoSecurityConfig {
        LigeritoSecurityConfig::derive_paper_compatible(29, 1, 100).expect("derive m29 fast")
    }

    /// Both embedded TOMLs (m29_fast at rate 1/2 and m29_slim at rate 1/4)
    /// parse, validate, and produce ProverConfig/VerifierConfig with the
    /// aggressive +2/level ladder and the same fold shape as
    /// `default_config(22, 5, rate)` (which climbs +1/level).
    #[test]
    fn ligerito_security_config_m29_toml_loads() {
        let toml_str = include_str!("../../configs/ligerito/m29_fast.toml");
        let cfg = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_fast.toml must parse and validate");
        assert_eq!(cfg.m, 29);
        assert_eq!(cfg.log_n, 22);
        // m29 Fast is the one initial_k-5 config (the recursion-node
        // row-width choice — see `derive_profile`).
        assert_eq!(cfg.initial_k, 5);
        assert_eq!(cfg.hash, "blake3");
        assert_eq!(cfg.levels.len(), 5);
        // Fast = JohnsonOod profile with 16-bit query PoW at every level:
        // 244 L0 queries put the raw query term at ~112 bits and the PoW
        // supplies the rest, work-normalized above 2^-128 (no list union
        // bound — single-codeword binding via the opening claim / OOD
        // samples). The MCA/proximity algebraic terms are evaluated over
        // F256. (The grind-free 279-query Fast was deleted 2026-08-27.)
        assert_eq!(cfg.levels[0].regime, SoundnessRegime::JohnsonOod);
        assert_eq!(cfg.levels[0].queries, 244);
        assert_eq!(cfg.levels[0].grinding_bits, 16);
        assert!(cfg.levels.iter().all(|lv| lv.grinding_bits == 16));
        assert!(cfg.levels.iter().all(|lv| lv.fold_grinding_bits == 0));
        assert_eq!(cfg.field, "f256");
        assert_eq!(cfg.levels[0].ood_samples, 1); // plus L0's opening point
        assert!(cfg.levels.iter().skip(1).all(|lv| lv.ood_samples == 2));
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 5, 1).unwrap();
        // Aggressive ladder: +2 rate per level (default_config climbs +1).
        assert_eq!(pv.log_inv_rates, vec![1, 3, 5, 7, 9]);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.queries[0], 244);

        // Slim mode: rates start at 1/4.
        let toml_str = include_str!("../../configs/ligerito/m29_slim.toml");
        let cfg_slim = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_slim.toml must parse and validate");
        assert_eq!(cfg_slim.levels[0].log_inv_rate, 2);
        // Slim = JohnsonOod at rate 1/4 with 16-bit query grinding.
        assert_eq!(cfg_slim.levels[0].queries, 119);
        assert_eq!(cfg_slim.levels[0].grinding_bits, 16);
        // m29 Slim is initial_k 5 like Fast (the recursion-node choice).
        assert_eq!(cfg_slim.initial_k, 5);
        let (pv_slim, _vc_slim) = cfg_slim.to_prover_verifier_configs().unwrap();
        let default_slim = default_config(22, 5, 2).unwrap();
        assert_eq!(pv_slim.log_inv_rates, vec![2, 4, 6, 8, 10]);
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

    /// Every embedded config validates strictly (i.e. each is paper-compatible
    /// and satisfies its declared component targets).
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

    /// Checked-in TOMLs are exactly the canonical output of the profile
    /// derivation. Validation alone would allow a hand-edited but internally
    /// consistent file to drift away from the generator.
    #[test]
    fn ligerito_all_embedded_configs_match_derivation() {
        for &((m, profile), toml) in EMBEDDED_CONFIGS {
            let derived = LigeritoSecurityConfig::derive_profile(m, profile)
                .unwrap_or_else(|e| panic!("derive m={m} profile={}: {e}", profile.as_str()));
            let canonical = derived
                .to_toml_string()
                .unwrap_or_else(|e| panic!("serialize m={m} profile={}: {e}", profile.as_str()));
            assert_eq!(
                toml,
                canonical,
                "embedded m={m} profile={} differs from generator output",
                profile.as_str()
            );
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
        // m=29 fast: known → loads from TOML at ITS initial_k (5 — the
        // recursion-node row-width choice); a stale batch-6 request is a
        // hard error, never a silent fallback.
        let pv = prover_config_for(22, 5, LigeritoProfile::Fast).expect("m29 fast must load");
        assert_eq!(pv.queries[0], 244);
        assert_eq!(pv.grinding_bits[0], 16);
        assert!(pv.fold_grinding_bits.iter().all(|&bits| bits == 0));
        assert_eq!(embedded_initial_k(29, LigeritoProfile::Fast), Some(5));
        let err = prover_config_for(22, 6, LigeritoProfile::Fast).unwrap_err();
        assert!(err.contains("initial_k=5"), "unexpected error: {err}");

        // m=29 slim: known → loads from TOML (initial_k 5, like Fast).
        let pv = prover_config_for(22, 5, LigeritoProfile::Slim).expect("m29 slim must load");
        assert_eq!(pv.queries[0], 119);
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

    /// The config's `hash` field selects the Merkle hash and reaches both
    /// derived configs — this is the knob the option is exposed through.
    #[test]
    fn ligerito_security_config_hash_field_selects_merkle_hash() {
        let mut cfg = blake3_m29_udr_example();
        assert_eq!(cfg.hash, "blake3", "example config baseline");
        let (p, v) = cfg.to_prover_verifier_configs().expect("blake3 configs");
        assert_eq!(p.merkle_hash, HashKind::Blake3);
        assert_eq!(v.merkle_hash, HashKind::Blake3);

        cfg.hash = "sha256".into();
        let (p, v) = cfg.to_prover_verifier_configs().expect("sha256 configs");
        assert_eq!(p.merkle_hash, HashKind::Sha256);
        assert_eq!(v.merkle_hash, HashKind::Sha256);

        // Survives a TOML round-trip, so the option is settable from a file.
        cfg.validate().expect("sha256 config validates");
        let back = LigeritoSecurityConfig::from_toml_str(&cfg.to_toml_string().unwrap())
            .expect("toml roundtrip");
        assert_eq!(back.merkle_hash().unwrap(), HashKind::Sha256);
    }

    /// A `hash` we do not implement must fail at validation rather than
    /// silently committing under SHA-256.
    #[test]
    fn ligerito_security_config_rejects_unknown_hash() {
        let mut cfg = blake3_m29_udr_example();
        cfg.hash = "keccak256".into();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("hash") && err.contains("keccak256"),
            "err = {err}"
        );
        assert!(cfg.to_prover_verifier_configs().is_err());
    }

    /// Every embedded config must name a hash we actually implement — a typo
    /// in a checked-in TOML should fail here, not at proving time.
    #[test]
    fn embedded_configs_all_declare_a_supported_hash() {
        for &((m, profile), toml) in EMBEDDED_CONFIGS {
            let cfg = LigeritoSecurityConfig::from_toml_str(toml)
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
            cfg.merkle_hash()
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
        }
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

    /// Generated Johnson schedules are minimal: the configured query count
    /// plus optional query grinding is strictly above 128 bits, while removing
    /// one query is at or below 128 bits.
    #[test]
    fn ligerito_list_decoding_queries_strictly_clear_128_bits() {
        for m in 22..=35 {
            for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m{m} {profile:?}: {e}"));
                for (i, lv) in cfg.levels.iter().enumerate() {
                    let per_q = paper_per_query_bits(
                        lv.log_inv_rate,
                        lv.eta.expect("Johnson profile has eta"),
                    );
                    let delivered = lv.queries as f64 * per_q + lv.grinding_bits as f64;
                    let one_fewer = (lv.queries - 1) as f64 * per_q + lv.grinding_bits as f64;
                    assert!(
                        delivered > LIST_DECODING_QUERY_TARGET_BITS,
                        "m{m} {profile:?} L{i}: delivered {delivered}"
                    );
                    assert!(
                        one_fewer <= LIST_DECODING_QUERY_TARGET_BITS,
                        "m{m} {profile:?} L{i}: non-minimal ({one_fewer})"
                    );
                }
            }
        }
    }

    /// Validation uses the exact query formula rather than trusting the
    /// rounded diagnostic in the TOML.
    #[test]
    fn ligerito_security_config_rejects_sub_128_list_query_schedule() {
        let mut cfg = LigeritoSecurityConfig::derive_profile(29, LigeritoProfile::Fast)
            .expect("derive m29 Fast");
        let lv = &mut cfg.levels[0];
        lv.queries -= 1;
        let per_q = paper_per_query_bits(lv.log_inv_rate, lv.eta.expect("Johnson eta"));
        lv.expected_eps_query_bits = round1(lv.queries as f64 * per_q);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("list-decoding target"), "err = {err}");
    }

    /// `Fast100` is `Fast` at the pre-list-decoding cost point: its query
    /// floor is the profile's own 100 bits (keyed off `analysis_version`),
    /// its schedule reproduces the shipped pre-128 counts, and the strict
    /// one-query boundary holds against THAT floor.
    #[test]
    fn ligerito_security_config_fast100_reproduces_pre_128_schedule() {
        let cfg = LigeritoSecurityConfig::derive_profile(27, LigeritoProfile::Fast100)
            .expect("derive m27 Fast100");
        let queries: Vec<usize> = cfg.levels.iter().map(|l| l.queries).collect();
        assert_eq!(
            queries,
            [218, 106, 71, 53],
            "the pre-list-decoding m27 ladder"
        );

        let mut cfg = LigeritoSecurityConfig::derive_profile(29, LigeritoProfile::Fast100)
            .expect("derive m29 Fast100");
        let lv = &mut cfg.levels[0];
        lv.queries -= 1;
        let per_q = paper_per_query_bits(lv.log_inv_rate, lv.eta.expect("Johnson eta"));
        lv.expected_eps_query_bits = round1(lv.queries as f64 * per_q);
        // One fewer query must be rejected against the 100-bit floor —
        // whichever of the two query-term validators catches it first.
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("100"),
            "the Fast100 floor is the profile's own 100-bit target: {err}"
        );
    }

    /// The algebraic schedule from Appendix C.3 of the Flock paper is
    /// enforced independently of the profile-local query/MCA target. The
    /// F128 batching challenges still require grinding, while F256 makes the
    /// sumcheck challenges independently 128-bit secure without fold PoW.
    #[test]
    fn ligerito_security_config_rejects_insufficient_algebraic_grinding() {
        let baseline = LigeritoSecurityConfig::derive_profile(29, LigeritoProfile::Fast)
            .expect("derive m29 Fast");

        let mut cfg = baseline.clone();
        cfg.levels[0].claim_batch_grinding_bits = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("claim_batch_grinding_bits"), "err = {err}");

        assert!(
            baseline
                .levels
                .iter()
                .all(|level| level.fold_grinding_bits == 0),
            "F256 sumcheck challenges need no fold grinding"
        );

        let mut cfg = baseline;
        cfg.levels[0].consistency_batch_grinding_bits = 0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("consistency_batch_grinding_bits"),
            "err = {err}"
        );
    }

    /// The old one-point Johnson schedule must not remain loadable after the
    /// collision analysis moved to two independent points.
    #[test]
    fn ligerito_security_config_rejects_one_point_ood() {
        let mut cfg = LigeritoSecurityConfig::derive_profile(29, LigeritoProfile::Fast)
            .expect("derive m29 Fast");
        cfg.levels[0].ood_samples = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("two-point"), "err = {err}");

        let mut cfg = LigeritoSecurityConfig::derive_profile(29, LigeritoProfile::Fast)
            .expect("derive m29 Fast");
        cfg.levels[1].ood_samples = 1;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("two-point"), "err = {err}");
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
        // a = γ·n + 1; ε_pg = 256 − log₂ a with NO row-union penalty in the
        // unique-decoding regime (list size 1; Diamond and Gruen). Any
        // shortfall below the 100-bit target is covered by fold-grinding.
        let log_a_base = (gamma * n + 1.0).log2();
        let eps_pg = FOLD_FIELD_LOG_Q - log_a_base;
        cfg.levels[0].regime = SoundnessRegime::Udr;
        cfg.levels[0].eta = None;
        cfg.levels[0].proximity_loss = Some(eps_star);
        cfg.levels[0].queries = queries;
        cfg.levels[0].grinding_bits = 0;
        let (claim_bits, sumcheck_bits, consistency_bits, e_claim, e_sumcheck, e_consistency) =
            algebraic_grinding_schedule(cfg.levels[0].log_inv_rate, None, queries);
        cfg.levels[0].fold_grinding_bits = sumcheck_bits;
        cfg.levels[0].claim_batch_grinding_bits = claim_bits;
        cfg.levels[0].consistency_batch_grinding_bits = consistency_bits;
        cfg.levels[0].expected_eps_pg_bits = (eps_pg * 10.0).round() / 10.0;
        cfg.levels[0].expected_eps_query_bits = ((queries as f64 * per_q) * 10.0).round() / 10.0;
        cfg.levels[0].expected_eps_claim_batch_bits = round1(e_claim);
        cfg.levels[0].expected_eps_sumcheck_bits = round1(e_sumcheck);
        cfg.levels[0].expected_eps_consistency_batch_bits = round1(e_consistency);
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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok = extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_cap(&v_cfg),
            1usize << initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        );
        assert!(
            ok,
            "verifier should accept proof with valid grinding nonces"
        );

        // Tampering with the nonce flips the PoW check.
        let mut bad_proof = proof.clone();
        bad_proof.grinding_nonces[0] = bad_proof.grinding_nonces[0].wrapping_add(1);
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok = extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &bad_proof,
            log_n,
            target,
            &initial_cap(&v_cfg),
            1usize << initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        );
        assert!(
            !ok,
            "verifier must reject proof with tampered grinding nonce"
        );
    }

    /// STRATIFIED roundtrips across query counts (docs/stratified-queries.tex).
    /// The schedule is config-build data (`with_default_stratified` +
    /// `with_stratified_open`); this sweep pins that ANY integer query count
    /// roundtrips — powers of two, odd, all-ones popcount, tiny — that the
    /// proof geometry follows the schedule (cap at the top set bit, paths
    /// `Σ 2^c·(d−c)` siblings), and that tampered openings, truncated paths,
    /// and a legacy/stratified mode mismatch all reject.
    #[test]
    fn stratified_roundtrip_sweeps_query_counts() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x57A7_1F1E);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mk_cfgs = |queries: Vec<usize>| {
            let p = ProverConfig {
                log_inv_rates: vec![log_inv_rate, log_inv_rate],
                recursive_steps: 1,
                initial_log_msg_cols: log_n - initial_k,
                initial_log_num_interleaved: initial_k,
                initial_k,
                recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
                recursive_ks: vec![k_0],
                queries: queries.clone(),
                grinding_bits: vec![0; 2],
                fold_grinding_bits: vec![0; 2],
                claim_batch_grinding_bits: vec![0; 2],
                consistency_batch_grinding_bits: vec![0; 2],
                ood_samples: vec![0; 2],
                merkle_hash: HashKind::Sha256,
                stratified: vec![],
            }
            .with_default_stratified();
            let v = VerifierConfig {
                log_inv_rates: vec![log_inv_rate, log_inv_rate],
                recursive_steps: 1,
                initial_log_msg_cols: log_n - initial_k,
                initial_log_num_interleaved: initial_k,
                initial_k,
                recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
                recursive_ks: vec![k_0],
                queries,
                grinding_bits: vec![0; 2],
                fold_grinding_bits: vec![0; 2],
                claim_batch_grinding_bits: vec![0; 2],
                consistency_batch_grinding_bits: vec![0; 2],
                ood_samples: vec![0; 2],
                merkle_hash: HashKind::Sha256,
                stratified: vec![],
            }
            .with_default_stratified();
            (p, v)
        };

        // (q0, q1): powers of two, the doc's L0 example, all-ones popcount,
        // a power-of-two against an odd partner, tiny counts.
        for (q0, q1) in [(64usize, 32usize), (90, 33), (127, 7), (96, 5), (5, 2)] {
            let (cfg, v_cfg) = mk_cfgs(vec![q0, q1]);
            let sched0 = &cfg.stratified[0];
            let cap0 = wtns_0.cap(sched0.cap_depth()).to_vec();

            let mut p_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            let proof = recursive_prover_with_basis(
                &cfg,
                poly.clone(),
                b.clone(),
                target,
                &wtns_0.mat,
                &wtns_0.tree,
                &mut p_ch,
            );

            // Geometry follows the schedule, not ceil(lg q): the absorbed cap
            // is the top summand's layer and the flat path vec sums the
            // per-summand walks.
            assert_eq!(proof.initial_cap.len(), 1 << sched0.cap_depth(), "q0={q0}");
            assert_eq!(
                proof.initial_proof.merkle_proof.len(),
                sched0.total_path_siblings(),
                "q0={q0}"
            );
            assert_eq!(
                proof.final_proof.merkle_proof.len(),
                v_cfg.stratified[1].total_path_siblings(),
                "q1={q1}"
            );

            let mut v_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            assert!(
                extension::recursive_verifier_with_basis_succinct(
                    &v_cfg,
                    &proof,
                    log_n,
                    target,
                    &cap0,
                    1usize << initial_k,
                    |ris, residual_log| {
                        extension::evaluate_dense_at_residual(&b, ris, residual_log)
                    },
                    &mut v_ch,
                ),
                "honest stratified proof rejected (q0={q0}, q1={q1})"
            );

            // Tampered opened row: the leaf hash moves, the walk misses its
            // stratum terminal.
            let mut bad = proof.clone();
            bad.initial_proof.opened_rows[0][0] += F128::ONE;
            let mut v_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            assert!(
                !extension::recursive_verifier_with_basis_succinct(
                    &v_cfg,
                    &bad,
                    log_n,
                    target,
                    &cap0,
                    1usize << initial_k,
                    |ris, residual_log| {
                        extension::evaluate_dense_at_residual(&b, ris, residual_log)
                    },
                    &mut v_ch,
                ),
                "tampered row accepted (q0={q0})"
            );

            // Tampered path sibling.
            let mut bad = proof.clone();
            bad.initial_proof.merkle_proof[0][0] ^= 1;
            let mut v_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            assert!(
                !extension::recursive_verifier_with_basis_succinct(
                    &v_cfg,
                    &bad,
                    log_n,
                    target,
                    &cap0,
                    1usize << initial_k,
                    |ris, residual_log| {
                        extension::evaluate_dense_at_residual(&b, ris, residual_log)
                    },
                    &mut v_ch,
                ),
                "tampered sibling accepted (q0={q0})"
            );

            // Truncated path vec: rejected on the total_path_siblings shape.
            let mut bad = proof.clone();
            bad.initial_proof.merkle_proof.pop();
            let mut v_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            assert!(
                !extension::recursive_verifier_with_basis_succinct(
                    &v_cfg,
                    &bad,
                    log_n,
                    target,
                    &cap0,
                    1usize << initial_k,
                    |ris, residual_log| {
                        extension::evaluate_dense_at_residual(&b, ris, residual_log)
                    },
                    &mut v_ch,
                ),
                "truncated paths accepted (q0={q0})"
            );

            // Padded path vec — a prover re-emitting the pre-truncation
            // above-cap siblings: rejected on the same shape check. Paths
            // are exactly q·(d − c1) since the cap-truncation landed.
            let mut bad = proof.clone();
            let extra = bad.initial_proof.merkle_proof[0];
            bad.initial_proof.merkle_proof.push(extra);
            let mut v_ch = crate::challenger::FsChallenger::new(b"strat-sweep");
            assert!(
                !extension::recursive_verifier_with_basis_succinct(
                    &v_cfg,
                    &bad,
                    log_n,
                    target,
                    &cap0,
                    1usize << initial_k,
                    |ris, residual_log| {
                        extension::evaluate_dense_at_residual(&b, ris, residual_log)
                    },
                    &mut v_ch,
                ),
                "padded paths accepted (q0={q0})"
            );
        }
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
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; r + 1],
            consistency_batch_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; r + 1],
            consistency_batch_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

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

        // CLOSED FORM under stratified capping: per tree, the schedule of
        // `q` gives `Σ 2^{c_i}·(d − c_i)` path siblings plus the top-set-bit
        // cap (the commitment) — deterministic, no sampling
        // (docs/stratified-queries.tex).
        let mut total_opened = 0usize;
        let mut total_merkle = 0usize;
        let mut total_caps = 0usize;
        for i in 0..=r {
            let bl = 1usize << log_block_len[i];
            let qn = queries_per_level[i];
            if qn > bl {
                eprintln!(
                    "  INFEASIBLE at commit {i}: queries ({qn}) > block_len ({bl}). Pick a higher rate (smaller bl) or smaller queries."
                );
                return usize::MAX;
            }
            let sched = stratified::LevelSchedule::decompose(qn, log_block_len[i]);
            let sib = sched.total_path_siblings();
            let cap_b = (1usize << sched.cap_depth()) * 32;
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
                "  {label} [bl=2^{}, lanes=2^{}, q={qn}, summands={:?}]: opened={}  merkle={} ({} sibs)  cap={}",
                log_block_len[i],
                log_num_interleaved[i],
                sched.summand_depths,
                kb(opened),
                kb(merkle),
                sib,
                kb(cap_b),
            );
            total_opened += opened;
            total_merkle += merkle;
            total_caps += cap_b;
        }
        let yr_b = (1usize << yr_log_n) * ELEM;
        // Transcript: 1 start + 1 intro per recursive boundary (R) + sum(k_i) folds, all (u_0, u_2).
        let sumcheck_msgs = 1 + r + recursive_ks.iter().sum::<usize>();
        let tx_b = sumcheck_msgs * 2 * ELEM;
        let total = total_opened + total_merkle + total_caps + yr_b + tx_b;
        eprintln!(
            "  TOTALS: opened={}  merkle={}  caps={}  yr={}  transcript={}  → GRAND={}",
            kb(total_opened),
            kb(total_merkle),
            kb(total_caps),
            kb(yr_b),
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
        eprintln!("estimator={estimated}  actual={actual}");
        // Under capping every term is deterministic — q·(d−c) path siblings
        // and a 2^c cap per level, config-fixed opened rows, yr, transcript,
        // nonces — so the estimate is EXACT, not a bound.
        assert_eq!(estimated, actual, "the capped size estimate is closed-form");
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
            claim_batch_grinding_bits: vec![0; 3],
            consistency_batch_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; 3],
            consistency_batch_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
        assert_eq!(proof.recursive_caps.len(), 2);
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
        let mut p_ch = crate::challenger::FsChallenger::new(b"serde");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let bytes = bincode::serialize(&proof).expect("serialize");
        let proof2: LigeritoProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(proof, proof2);
        eprintln!("LigeritoProof bincode size: {} bytes", bytes.len());
    }

    /// The round-1 lookahead skip is an exact polynomial identity, so the
    /// F256 ladder must emit BYTE-IDENTICAL proofs with and without it —
    /// including across the L0 OOD β-glues, which the lookahead survives via
    /// the per-OOD coefficient correction ([`round_msg_eval_and_lookahead`]).
    /// Registered m22 fast config (ood_samples[0] = 1, real PoW schedule),
    /// plus an ood_samples[0] = 2 variant to exercise repeated corrections;
    /// the registered-config proof is also verified.
    #[test]
    fn f256_round1_lookahead_is_byte_identical() {
        use crate::challenger::Challenger;
        let log_n = 15;
        let cfg = prover_config_for(log_n, 6, LigeritoProfile::Fast).unwrap();

        let mut rng = crate::challenger::RandomChallenger::new(0x10_0CA_4EAD);
        let f: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let b: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();

        // The combine pass's entry accumulation: round-0 message + round-1
        // coefficients in one sweep (also pins the fused y against the plain
        // message pass). The true inner product is the sumcheck target.
        let (first, y, la) = round_msg_eval_and_lookahead(&f, &b);
        let (first_plain, y_plain) = round_msg_and_eval_blocked(&f, &b, 1);
        assert_eq!(first, first_plain);
        assert_eq!(y, y_plain);
        let target = y;

        let log_msg_cols_0 = log_n - cfg.initial_k;
        let ntt = AdditiveNttF128::standard(log_msg_cols_0 + cfg.log_inv_rates[0]);
        let wtns = ligero_commit(
            &f,
            log_msg_cols_0,
            cfg.initial_k,
            cfg.log_inv_rates[0],
            &ntt,
            cfg.merkle_hash,
        );

        let prove = |cfg: &ProverConfig, la: Option<FoldLookahead>| -> LigeritoProof {
            let mut ch = crate::challenger::FsChallenger::new(b"la-test");
            recursive_prover_with_basis_precomputed_round0(
                cfg,
                f.clone(),
                b.clone(),
                target,
                &wtns.mat,
                &wtns.tree,
                (first.u_0, first.u_2),
                la,
                &mut ch,
            )
        };
        let with_la = prove(&cfg, Some(la));
        let without = prove(&cfg, None);
        assert_eq!(with_la, without, "lookahead skip must not move a byte");

        // And the proof is a real one.
        let v_cfg = verifier_config_for(log_n, 6, LigeritoProfile::Fast).unwrap();
        let cap = wtns.cap(v_cfg.l0_cap_depth()).to_vec();
        let mut v_ch = crate::challenger::FsChallenger::new(b"la-test");
        assert!(extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &with_la,
            log_n,
            target,
            &cap,
            1 << cfg.initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        ));

        // Two L0 OODs → two lookahead corrections.
        let mut cfg2 = cfg.clone();
        cfg2.ood_samples[0] = 2;
        assert_eq!(
            prove(&cfg2, Some(la)),
            prove(&cfg2, None),
            "lookahead must survive repeated OOD β-glues"
        );
    }

    /// `recursive_prover_with_basis` +
    /// `extension::recursive_verifier_with_basis_succinct` roundtrip.
    /// Single-claim case (`b = eq(z, ·)`, `target = poly(z)`) — must
    /// round-trip cleanly.
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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
        let mut v_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let ok = extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_cap(&v_cfg),
            1usize << initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        );
        assert!(ok, "basis-based verifier rejected valid proof");
    }

    /// The sampler draws with replacement: exactly `count` positions, in
    /// range, challenger-deterministic, from ONE batched squeeze — and at the
    /// shipped L0 shape it really does repeat, so the duplicate handling below
    /// is on the live path rather than a theoretical branch.
    #[test]
    fn sample_queries_is_with_replacement_from_one_batched_squeeze() {
        use crate::challenger::Challenger;
        let block_len = 1usize << 13;
        let count = udr_queries(1); // 243 — the shipped L0 count at rate 1/2

        let sched = stratified::LevelSchedule::decompose(count, 13);
        let mut ch = crate::challenger::FsChallenger::new(b"sample-queries-test");
        let qs = sample_queries(&mut ch, block_len, count, &sched);

        // Exactly `count` draws. The old sampler's draw count was data-
        // dependent (it redrew on a repeat); this one is not, which is the
        // whole point for the circuit.
        assert_eq!(qs.len(), count);
        assert!(qs.iter().all(|&q| q < block_len));

        // Challenger-deterministic.
        let mut ch2 = crate::challenger::FsChallenger::new(b"sample-queries-test");
        assert_eq!(sample_queries(&mut ch2, block_len, count, &sched), qs);

        // One `sample_f128_vec` squeeze plus a low-bit mask — nothing else.
        // This pins the transcript shape the FS chain table has to replay.
        let mut ch3 = crate::challenger::FsChallenger::new(b"sample-queries-test");
        let raw = ch3.sample_f128_vec(count);
        let expected: Vec<usize> = sched
            .query_strata()
            .zip(&raw)
            .map(|((c, stratum), v)| {
                let lo_bits = 13 - c;
                (stratum << lo_bits) | ((v.lo as usize) & ((1usize << lo_bits) - 1))
            })
            .collect();
        assert_eq!(qs, expected);

        // 243 draws into 2^13 has a birthday expectation of ~3.6 collisions.
        let mut uniq = qs.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert!(
            uniq.len() < qs.len(),
            "expected repeats among 243 draws into 2^13, got {} unique",
            uniq.len()
        );
    }

    /// A repeated query position must carry a single row. Only one row per
    /// position reaches the Merkle check, but every slot feeds
    /// `induce_sumcheck_enforced_sum` — so without this check a prover could
    /// slip an unchecked row into the induced basis through the second slot.
    #[test]
    fn verify_level_opens_rejects_disagreeing_rows_at_a_repeated_position() {
        use crate::challenger::Challenger;
        let (log_msg, log_interleaved, log_inv_rate) = (4usize, 2usize, 2usize);
        let num_interleaved = 1usize << log_interleaved;
        let poly_len = (1usize << log_msg) * num_interleaved;

        let mut ch = crate::challenger::RandomChallenger::new(0x0DDD_1CE5);
        let poly: Vec<F128> = (0..poly_len).map(|_| ch.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );

        // A multiset in sample order with a deliberate repeat — the shape
        // `sample_queries` produces (unsorted, duplicates kept). Under
        // per-query capped paths there is no dedup and no explicit
        // agreement check: each slot's row is Merkle-bound by its OWN path
        // against the cap, so two DIFFERENT rows at one position cannot
        // both verify — the tree has one leaf there. The statement this
        // test pins is unchanged; the mechanism moved from an explicit
        // comparison to per-path binding.
        // 5 queries = summands [2, 0]: four depth-2 strata plus one whole-
        // block draw — the depth-0 query REPEATS position 9 across summands
        // (within one summand strata are disjoint, so repeats only ever
        // happen across summands).
        let queries = vec![9usize, 17, 33, 49, 9];
        let sched = stratified::LevelSchedule::decompose(
            queries.len(),
            w.block_len.trailing_zeros() as usize,
        );
        let c = sched.cap_depth();
        let cap = w.cap(c).to_vec();
        let honest: Vec<Vec<F128>> = queries.iter().map(|&q| w.row(q).to_vec()).collect();
        let proof = merkle_paths_for(&w.tree, w.block_len, &queries, &sched);

        assert!(
            verify_level_opens(
                &cap,
                w.block_len,
                &queries,
                &honest,
                num_interleaved,
                &proof,
                HashKind::Sha256,
                &sched
            ),
            "honest unsorted opening with a repeat must verify"
        );

        // Slot 4 is the second occurrence of position 9. Tamper it: its own
        // path check fails (position 9's leaf is fixed by the cap).
        let mut forged = honest.clone();
        forged[4][0] += F128::ONE;
        assert!(
            !verify_level_opens(
                &cap,
                w.block_len,
                &queries,
                &forged,
                num_interleaved,
                &proof,
                HashKind::Sha256,
                &sched
            ),
            "disagreeing rows at a repeated position must be rejected"
        );

        // Control: the same tamper at a non-repeated position is equally
        // caught — every slot is bound the same way.
        let mut tampered = honest.clone();
        tampered[1][0] += F128::ONE;
        assert!(
            !verify_level_opens(
                &cap,
                w.block_len,
                &queries,
                &tampered,
                num_interleaved,
                &proof,
                HashKind::Sha256,
                &sched
            ),
            "a tampered row at a unique position must fail its path check"
        );

        // Shape checks: a truncated or extended flat path vector, and a cap
        // of the wrong size, reject before any hashing.
        let mut short = proof.clone();
        short.pop();
        assert!(!verify_level_opens(
            &cap,
            w.block_len,
            &queries,
            &honest,
            num_interleaved,
            &short,
            HashKind::Sha256,
            &sched
        ));
        let mut long = proof.clone();
        long.push([0u8; 32]);
        assert!(!verify_level_opens(
            &cap,
            w.block_len,
            &queries,
            &honest,
            num_interleaved,
            &long,
            HashKind::Sha256,
            &sched
        ));
        let wrong_cap = w.cap(c + 1).to_vec();
        assert!(!verify_level_opens(
            &wrong_cap,
            w.block_len,
            &queries,
            &honest,
            num_interleaved,
            &proof,
            HashKind::Sha256,
            &sched
        ));

        // Wrong-position binding: swap two slots' path segments (rows
        // unchanged) — each row now folds against the other's siblings.
        let path_len = w.block_len.trailing_zeros() as usize - c;
        let mut swapped = proof.clone();
        for t in 0..path_len {
            swapped.swap(t, path_len + t); // slot 0 <-> slot 1
        }
        assert!(!verify_level_opens(
            &cap,
            w.block_len,
            &queries,
            &honest,
            num_interleaved,
            &swapped,
            HashKind::Sha256,
            &sched
        ));
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
        let wtns = ligero_commit(&yr, log_msg_cols, 0, log_inv_rate, &ntt, HashKind::Sha256);

        // Distinct query positions. The protocol samples with replacement, so
        // distinctness is not required here — it just keeps this fixture's
        // expected values easy to reason about.
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

    /// The F256 succinct verifier accepts a proof when its residual-basis
    /// callback evaluates the same materialized basis used by the prover.
    #[test]
    fn recursive_verifier_with_basis_succinct_accepts_dense_basis() {
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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let mut v_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let ok = extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_cap(&v_cfg),
            1usize << v_cfg.initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        );
        assert!(ok, "F256 succinct verifier must accept");
    }

    /// Build a matching (ProverConfig, VerifierConfig) pair with explicit
    /// OOD samples for the OOD-path tests below.
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
        let mut dim = log_n - initial_k + 1;
        for &k in ks {
            recursive_log_msg_cols.push(dim - k);
            dim = dim - k + 1;
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
            claim_batch_grinding_bits: vec![3; r + 1],
            consistency_batch_grinding_bits: vec![4; r + 1],
            ood_samples: ood_samples.clone(),
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![3; r + 1],
            consistency_batch_grinding_bits: vec![4; r + 1],
            ood_samples,
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
        (p, v)
    }

    /// End-to-end OOD binding under F256: a JohnsonOod-shaped config (one
    /// extra OOD at L0 and two at L1/L2) round-trips through the production
    /// succinct verifier. Tampering with OOD values or either F128 batching
    /// nonce family rejects.
    #[test]
    fn ligerito_ood_and_fold_grinding_roundtrip_and_tamper() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 2;
        let ks = [2usize, 2, 2, 2];
        // Exercise enough code switches to match the production recursion
        // depth: one point at L0 and two at every split commitment.
        let (p_cfg, v_cfg) = ood_test_configs(
            log_n,
            initial_k,
            &ks,
            vec![1, 2, 2, 2, 2],
            vec![0, 0, 0, 0, 0],
        );

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
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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
        assert_eq!(proof.ood_values.len(), 9, "1 + 4·2 explicit OOD samples");
        assert!(proof.fold_grinding_nonces.is_empty());
        assert_eq!(proof.claim_batch_grinding_nonces.len(), 9 + 5);
        assert_eq!(proof.consistency_batch_grinding_nonces.len(), 5);

        let verify = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            extension::recursive_verifier_with_basis_succinct(
                &v_cfg,
                proof,
                log_n,
                target,
                &initial_cap(&v_cfg),
                1usize << v_cfg.initial_k,
                |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
                &mut ch,
            )
        };

        assert!(verify(&proof), "F256 verifier must accept OOD proof");

        // Tamper every OOD value in turn → both verifiers reject. Iterating
        // all positions certifies L0 and every explicit recursive point,
        // rather than only the first flattened proof entry.
        for idx in 0..proof.ood_values.len() {
            let mut bad_ood = proof.clone();
            bad_ood.ood_values[idx] += F128::ONE;
            assert!(!verify(&bad_ood), "must reject tampered OOD value {idx}");
        }

        let mut missing_ood = proof.clone();
        missing_ood.ood_values.pop();
        assert!(!verify(&missing_ood), "must reject a missing OOD value");

        let mut extra_ood = proof.clone();
        extra_ood.ood_values.push(F128::ZERO);
        assert!(!verify(&extra_ood), "must reject an extra OOD value");

        let mut missing_query_nonce = proof.clone();
        missing_query_nonce.grinding_nonces.pop();
        assert!(
            !verify(&missing_query_nonce),
            "must reject a missing query nonce"
        );

        let mut extra_query_nonce = proof.clone();
        extra_query_nonce.grinding_nonces.push(0);
        assert!(
            !verify(&extra_query_nonce),
            "must reject a trailing query nonce"
        );

        for (name, mutate) in [
            ("claim batching", |p: &mut LigeritoProof| {
                p.claim_batch_grinding_nonces[0] ^= 0xDEAD_BEEF
            }),
            ("consistency batching", |p: &mut LigeritoProof| {
                p.consistency_batch_grinding_nonces[0] ^= 0xDEAD_BEEF
            }),
        ] as [(&str, fn(&mut LigeritoProof)); 2]
        {
            let mut bad = proof.clone();
            mutate(&mut bad);
            assert!(!verify(&bad), "must reject tampered {name} nonce");
        }

        for (name, mutate) in [
            ("missing claim-batching", |p: &mut LigeritoProof| {
                p.claim_batch_grinding_nonces.pop();
            }),
            ("extra claim-batching", |p: &mut LigeritoProof| {
                p.claim_batch_grinding_nonces.push(0)
            }),
            ("missing consistency-batching", |p: &mut LigeritoProof| {
                p.consistency_batch_grinding_nonces.pop();
            }),
            ("extra consistency-batching", |p: &mut LigeritoProof| {
                p.consistency_batch_grinding_nonces.push(0)
            }),
        ] as [(&str, fn(&mut LigeritoProof)); 4]
        {
            let mut bad = proof.clone();
            mutate(&mut bad);
            assert!(!verify(&bad), "must reject {name} nonce vector");
        }
    }

    /// A real embedded profile config (m=22 fast = JohnsonOod) drives a full
    /// prover→verifier round-trip through the basis opening path. This is the
    /// production shape: OOD samples and the F256 split ladder come straight from the
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
        assert!(p_cfg.fold_grinding_bits.iter().all(|&g| g == 0));
        assert!(p_cfg.recursive_ks.iter().all(|&k| k == 4));

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
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            p_cfg.merkle_hash,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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

        let verify = |candidate: &LigeritoProof| {
            let mut v_ch = crate::challenger::FsChallenger::new(b"m22-fast");
            extension::recursive_verifier_with_basis_succinct(
                &v_cfg,
                candidate,
                log_n,
                target,
                &initial_cap(&v_cfg),
                1usize << initial_k,
                |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
                &mut v_ch,
            )
        };
        assert!(verify(&proof), "m22 fast profile proof must verify");

        let mut bad = proof.clone();
        bad.sumcheck_transcript_f256[0].u_0.c0.lo ^= 1;
        assert!(!verify(&bad), "mutated F256 transcript limb must reject");

        let mut bad = proof.clone();
        bad.sumcheck_transcript_f256[0].u_0.c1.lo ^= 1;
        assert!(!verify(&bad), "mutated extension coefficient must reject");

        let mut bad = proof.clone();
        bad.ood_values[0].lo ^= 1;
        assert!(!verify(&bad), "mutated base-field OOD answer must reject");

        let mut bad = proof.clone();
        bad.final_proof.yr[0].lo ^= 1;
        assert!(!verify(&bad), "mutated split residual word must reject");

        let mut bad = proof.clone();
        bad.final_proof.opened_rows[0][0].lo ^= 1;
        assert!(!verify(&bad), "mutated coordinate row opening must reject");
    }

    /// End-to-end under SHA-256 (the non-default hash): the same recursion,
    /// every Merkle commitment (L0 and each recursive level) built and
    /// checked with the other hash. Also pins the failure mode of a hash
    /// mismatch — a verifier configured for the wrong hash must reject,
    /// since the roots commit to the hash.
    #[test]
    fn ligerito_m22_roundtrip_under_sha256() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let mut v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");
        // The embedded configs all declare blake3; override to exercise the
        // other arm of the option end to end.
        assert_eq!(p_cfg.merkle_hash, HashKind::Blake3);
        p_cfg.merkle_hash = HashKind::Sha256;
        v_cfg.merkle_hash = HashKind::Sha256;

        let mut rng = crate::challenger::RandomChallenger::new(0xB1A5_E300);
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
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            p_cfg.merkle_hash,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            extension::recursive_verifier_with_basis_succinct(
                &v_cfg,
                &proof,
                log_n,
                target,
                &initial_cap(&v_cfg),
                1usize << initial_k,
                |ris, residual_log| {
                    extension::evaluate_dense_at_residual(&b, ris, residual_log)
                },
                &mut v_ch,
            ),
            "sha256 Merkle proof must verify"
        );

        // Same proof, verifier configured for BLAKE3 → every opening's
        // recomputed root disagrees, so it must reject.
        let mut wrong_cfg = v_cfg.clone();
        wrong_cfg.merkle_hash = HashKind::Blake3;
        let mut w_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            !extension::recursive_verifier_with_basis_succinct(
                &wrong_cfg,
                &proof,
                log_n,
                target,
                &initial_cap(&v_cfg),
                1usize << initial_k,
                |ris, residual_log| {
                    extension::evaluate_dense_at_residual(&b, ris, residual_log)
                },
                &mut w_ch
            ),
            "a sha256-configured verifier must reject a blake3 proof"
        );
    }

    /// The Merkle hash and the Fiat-Shamir transcript hash are independent
    /// options: all four combinations must prove and verify. Also pins the
    /// failure mode of a transcript-hash mismatch, the FS analogue of the
    /// Merkle mismatch checked above.
    #[test]
    fn ligerito_m22_roundtrip_over_hash_matrix() {
        use crate::challenger::Challenger;
        const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];
        let log_n = 22usize - crate::pcs::LOG_PACKING;
        let initial_k = 6;

        for merkle_hash in KINDS {
            for fs_hash in KINDS {
                let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                let mut v_cfg =
                    verifier_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                p_cfg.merkle_hash = merkle_hash;
                v_cfg.merkle_hash = merkle_hash;

                let mut rng = crate::challenger::RandomChallenger::new(0x4A11_0000);
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
                let wtns_0 =
                    ligero_commit(&poly, log_msg_cols_0, initial_k, 1, &ntt_0, merkle_hash);
                let initial_cap =
                    |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

                let mut p_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                let proof = recursive_prover_with_basis(
                    &p_cfg,
                    poly,
                    b.clone(),
                    target,
                    &wtns_0.mat,
                    &wtns_0.tree,
                    &mut p_ch,
                );

                let mut v_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                assert!(
                    extension::recursive_verifier_with_basis_succinct(
                        &v_cfg,
                        &proof,
                        log_n,
                        target,
                        &initial_cap(&v_cfg),
                        1usize << initial_k,
                        |ris, residual_log| {
                            extension::evaluate_dense_at_residual(&b, ris, residual_log)
                        },
                        &mut v_ch
                    ),
                    "merkle={merkle_hash} fs={fs_hash} must verify"
                );

                // Verifier on the other transcript hash: challenges diverge
                // from the first sample, so it must reject.
                let other_fs = match fs_hash {
                    HashKind::Sha256 => HashKind::Blake3,
                    HashKind::Blake3 => HashKind::Sha256,
                };
                let mut w_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", other_fs);
                assert!(
                    !extension::recursive_verifier_with_basis_succinct(
                        &v_cfg,
                        &proof,
                        log_n,
                        target,
                        &initial_cap(&v_cfg),
                        1usize << initial_k,
                        |ris, residual_log| {
                            extension::evaluate_dense_at_residual(&b, ris, residual_log)
                        },
                        &mut w_ch
                    ),
                    "merkle={merkle_hash}: an {other_fs} transcript must reject an {fs_hash} proof"
                );
            }
        }
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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_cap =
            |cfg: &VerifierConfig| -> Vec<Hash> { wtns_0.cap(cfg.l0_cap_depth()).to_vec() };

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
            recursive_log_msg_cols: vec![log_n - initial_k - (k_0 - 1)],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
        let mut v_ch = crate::challenger::FsChallenger::new(b"batched");
        let ok = extension::recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_cap(&v_cfg),
            1usize << initial_k,
            |ris, residual_log| extension::evaluate_dense_at_residual(&b, ris, residual_log),
            &mut v_ch,
        );
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

        // Path 1: built-in L0 commit.
        let mut p_ch = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_a = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        // Path 2: build L0 externally via ligero_commit, then call _with_l0.
        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let mut wtns_0_external = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
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
        assert_eq!(proof_a.initial_cap, proof_b.initial_cap);
        assert_eq!(proof_a.recursive_caps, proof_b.recursive_caps);
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();
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
            claim_batch_grinding_bits: vec![0; 2],
            consistency_batch_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: HashKind::Sha256,
            stratified: vec![],
        }
        .with_default_stratified();

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
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
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
        let w2 = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.tree.last(), w2.tree.last());
    }
}
