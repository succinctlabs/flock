//! Zerocheck PIOP: prove a(y) · b(y) ⊕ c(y) = 0 for all y ∈ {0,1}^m.
//!
//! Inputs are three bit vectors of length 2^m. Output is an evaluation claim
//! on the multilinear extensions â, b̂, ĉ at the protocol-derived point.
//!
//! Protocol shape (m = log_n, k_skip = [`K_SKIP`] = 6):
//!   1. Verifier samples `r ∈ F_{2^128}^m` (the zerocheck challenge).
//!   2. Prover sends `P^{AB}(λ)` and `P^C(λ)` for λ ∈ Λ, |Λ| = 2^k_skip.
//!   3. Verifier samples `z ∈ F_{2^128}` (univariate-skip fold point).
//!   4. For each of the `m - k_skip` multilinear rounds, prover sends
//!      `(P_r(1), P_r(∞))` and verifier samples `ρ_r`.
//!   5. Prover sends final MLE evaluations `(â, b̂, ĉ)` at the resulting point.
//!
//! Both `prove` and `verify` are wired end-to-end. The prove→verify roundtrip
//! is tested on honest witnesses; verify also rejects byte-mutated proofs and
//! shape-corrupted ones.

use crate::challenger::Challenger;
use crate::field::{F8, F128};
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use serde::{Deserialize, Serialize};

// The AG-skip prover half (round-1 kernel drivers, friendly-Horner tail,
// r1 nonce grind) is only reachable from aarch64-gated entry points; the
// verifier half is cross-arch. Silence the resulting dead-code cascade on
// non-aarch64 lint legs at the module level — aarch64 keeps full detection.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub mod ag_skip;
pub mod multilinear;
pub mod univariate_skip;
pub mod univariate_skip_optimized;

use multilinear::{
    UniSkipFoldTable, fold_and_compute_round_pair_into, fold_and_round_pair_sparse_into,
    fold_in_place_pair, interpolate_at_z_combined, interpolate_at_z_on_lambda, round_pair_naive,
    uni_skip_fold_and_round_pair_optimized_packed_padded,
};
use univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

/// Number of variables folded in round 1 via the additive-NTT univariate skip.
/// |Λ| = 2^K_SKIP = 64 elements; the round-1 prover message is two length-64
/// vectors of F128.
pub const K_SKIP: usize = 6;

/// Fiat--Shamir grinding policy for this zerocheck.
///
/// Zerocheck has three independently sampled challenge families:
///
/// 1. the initial point used to turn a Boolean-cube identity into an
///    eq-weighted claim;
/// 2. the univariate-skip point `z`; and
/// 3. one challenge for every quadratic multilinear sumcheck round.
///
/// [`Self::per_challenge_128`] chooses enough leading-zero PoW bits that an
/// error term of the form `degree / |F_{2^128}|`, after a prover's trial and
/// error over the challenge, is *strictly* below `2^-128`.  In particular,
/// `degree = 2` needs two bits, not one: `2 / 2 = 1` would only meet the
/// bound, not beat it.
///
/// The policy is intentionally local to zerocheck. Other sumcheck families
/// have different degrees and proof formats, so they define separate
/// schedules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZerocheckGrinding {
    enabled: bool,
}

impl ZerocheckGrinding {
    /// Legacy/default behaviour: no PoW operations and no nonce payload in
    /// the transcript.
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Grind every zerocheck challenge whose soundness is being used.
    pub const fn per_challenge_128() -> Self {
        Self { enabled: true }
    }

    /// Number of leading-zero bits which strictly turns
    /// `numerator / 2^128` into a value below `2^-128`.
    ///
    /// Equivalently this is `floor(log2(numerator)) + 1`.  Keeping the
    /// strict inequality explicit prevents an accidental one-bit
    /// under-provisioning at powers of two.
    const fn bits_for_numerator(numerator: usize) -> u32 {
        debug_assert!(numerator > 0);
        usize::BITS - numerator.leading_zeros()
    }

    /// PoW before sampling the initial eq-weighted identity point.  A
    /// nonzero multilinear polynomial in `m` variables has total degree at
    /// most `m`, so Schwartz--Zippel contributes at most `m / |F|` here.
    pub const fn initial_bits(self, m: usize) -> Option<u32> {
        if self.enabled {
            Some(Self::bits_for_numerator(m))
        } else {
            None
        }
    }

    /// PoW before sampling the univariate-skip point.  The combined round-1
    /// polynomial has degree `< 2^(K_SKIP + 1)`, hence degree at most
    /// `2^(K_SKIP + 1) - 1`.
    pub const fn skip_bits(self) -> Option<u32> {
        if self.enabled {
            Some(Self::bits_for_numerator((1usize << (K_SKIP + 1)) - 1))
        } else {
            None
        }
    }

    /// Explicit PoW bits on the AG-skip zerocheck's FUSED `r₁` nonce
    /// ([`ag_skip::sample_r1_prover_pow`]): ALL `bits_for(474) = 9` bits
    /// required ([`ag_skip::R1_ZERO_BOUND`]) are explicit — the recursion
    /// circuit binds the decode with RELAXED canonicity (any fiber point
    /// over the XOF-derived `x`), which returns the sampler's 5 flattening
    /// bits to the prover, so they are repaid in the PoW target. `None`
    /// under a disabled schedule (the direct route's plain single-attempt
    /// nonce, which makes no 128-bit claim).
    pub const fn ag_r1_bits(self) -> Option<u32> {
        if self.enabled {
            Some(ag_skip::R1_POW_BITS)
        } else {
            None
        }
    }

    /// PoW before a standard degree-two tail-round challenge.
    pub const fn multilinear_round_bits(self) -> Option<u32> {
        if self.enabled {
            Some(Self::bits_for_numerator(2))
        } else {
            None
        }
    }

    /// Number of nonce words carried by one proof at domain dimension `m`.
    pub const fn nonce_count(self, m: usize) -> usize {
        if self.enabled {
            // initial point, skip point, and one nonce per tail round
            2 + m - K_SKIP
        } else {
            0
        }
    }
}

/// Sparse-support gate for round 2 and the tail: the support-proportional
/// kernels engage while `live · SPARSE_TAIL_GATE ≤ n`. Set to 16 when the
/// sparse tail was a sequential scalar walk; after its parallelization
/// (dense-kernel structure) the crossover moved — measured on the capacity
/// sweeps at both the small and the real-m30 load, the sparse path runs at
/// per-element parity with the dense fold (25% utilization matches the
/// full-utilization zerocheck), so the gate engages at half utilization.
/// Full utilization itself stays dense (live · 2 > n): it is the anchor
/// configuration and the dense kernels are the calibrated choice there.
pub const SPARSE_TAIL_GATE: usize = 1;

/// [`SPARSE_TAIL_GATE`] with an env override (`FLOCK_SPARSE_GATE`) — a
/// tuning knob for A/B experiments; the constant above is the default.
/// Value-identical either way (the sparse kernels drop only zero terms).
fn sparse_tail_gate() -> usize {
    static GATE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("FLOCK_SPARSE_GATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(SPARSE_TAIL_GATE)
    })
}

/// One run of identically-shaped blocks inside a [`PaddingSpec`] run-list.
///
/// A run is `n_blocks` consecutive blocks of `2^k_log` bits each; inside each
/// block, bits `[0, useful_bits_per_block)` carry real data and bits
/// `[useful_bits_per_block, 2^k_log)` are zero padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddingRun {
    pub k_log: usize,
    pub useful_bits_per_block: usize,
    pub n_blocks: usize,
}

impl PaddingRun {
    /// Address-space extent of the run in bits (= `n_blocks · 2^k_log`).
    pub fn extent_bits(&self) -> usize {
        self.n_blocks << self.k_log
    }
}

/// Witness padding descriptor for URM / fold work-skipping.
///
/// The witness is described by an ordered **run-list**: the [`PaddingRun`]s
/// are laid out back-to-back from address 0, and everything after the last
/// run (up to the instance's `2^m` domain) is an implicit all-zero gap.
/// URM/fold contributions from a chunk of all-zero bits are themselves zero,
/// so kernels may skip any chunk the spec marks as padding or gap and produce
/// byte-identical output — provided those bits are honestly zero.
///
/// Single-table callers build **single-run** specs (one run tiling the whole
/// domain: [`PaddingSpec::dense`], [`PaddingSpec::uniform`], and
/// `BlockR1cs::padding_spec`); the hot kernels detect that case via
/// [`PaddingSpec::as_single_run`] and take exactly the pre-run-list code
/// path. Multi-run specs — the count-derived slot schedules of the
/// multi-table design (`docs/multi-table-design.tex` §5.2, the union prove
/// path) — go through general run-list paths that, since M6, skip dead
/// regions with cost proportional to the declared support
/// ([`Self::useful_block_intervals`] drives the interval-based kernels).
///
/// Use [`PaddingSpec::dense`] when the witness has no padding holes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddingSpec {
    runs: Vec<PaddingRun>,
}

impl PaddingSpec {
    /// "No padding": every bit of the witness is treated as useful. Equivalent
    /// to the legacy URM path with no skipping.
    pub fn dense(m: usize) -> Self {
        Self::uniform(m, 1usize << m, 1)
    }

    /// Single-run spec: `n_blocks` blocks of `2^k_log` bits, each with a
    /// `useful_bits_per_block` useful prefix. With `n_blocks = 2^(m − k_log)`
    /// this is exactly the pre-run-list `PaddingSpec`.
    pub fn uniform(k_log: usize, useful_bits_per_block: usize, n_blocks: usize) -> Self {
        Self::from_runs(vec![PaddingRun {
            k_log,
            useful_bits_per_block,
            n_blocks,
        }])
    }

    /// General run-list constructor. Runs with `n_blocks = 0` cover no address
    /// space and are dropped (canonical form, so `as_single_run` is reliable).
    pub fn from_runs(runs: Vec<PaddingRun>) -> Self {
        for run in &runs {
            assert!(
                run.useful_bits_per_block <= 1usize << run.k_log,
                "useful_bits_per_block {} exceeds block size 2^{}",
                run.useful_bits_per_block,
                run.k_log
            );
        }
        Self {
            runs: runs.into_iter().filter(|r| r.n_blocks > 0).collect(),
        }
    }

    /// The runs, in address order.
    pub fn runs(&self) -> &[PaddingRun] {
        &self.runs
    }

    /// The single run when the list has exactly one — the hot kernels' fast
    /// path. The fast path treats the run as tiling the entire domain
    /// periodically (it ignores `n_blocks`), which matches the pre-run-list
    /// kernels bit-for-bit; a single run with a trailing gap is still handled
    /// correctly because the gap must be honestly zero, like all padding.
    pub fn as_single_run(&self) -> Option<PaddingRun> {
        match self.runs.as_slice() {
            [run] => Some(*run),
            _ => None,
        }
    }

    /// Total extent covered by the runs, in bits. The instance domain `2^m`
    /// may be larger; the difference is the implicit trailing zero gap.
    pub fn covered_bits(&self) -> usize {
        self.runs.iter().map(|r| r.extent_bits()).sum()
    }

    /// [`Self::useful_intervals`] coarsened to `2^log2_block`-bit blocks and
    /// merged: block `x` is listed iff bits `[x·2^log2_block,
    /// (x+1)·2^log2_block)` intersect a useful interval. This is the live set
    /// of a table whose entries each aggregate one block of witness bits
    /// (e.g. the post-URM tables at `log2_block = k_skip`, or packed words at
    /// `log2_block = 7`): outside it the honest table is identically zero.
    pub fn useful_block_intervals(&self, log2_block: usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (s, e) in self.useful_intervals() {
            let (s2, e2) = (s >> log2_block, e.div_ceil(1usize << log2_block));
            match out.last_mut() {
                Some((_, prev_e)) if *prev_e >= s2 => *prev_e = (*prev_e).max(e2),
                _ => out.push((s2, e2)),
            }
        }
        out
    }

    /// Per-block coverage over `n_blocks` blocks of `2^log2_block` bits —
    /// the gating map for block-grained kernels (the AG round 1 and fold,
    /// whose natural tile is the 8192-bit code block). `Dead` blocks may be
    /// skipped outright (their honest contribution is zero), `Full` blocks
    /// read in place, and `Partial` blocks carry their block-local useful
    /// bit ranges so a kernel can cleanse them into a zeroed scratch block
    /// ([`cleanse_block`]) and never read a declared-dead bit — the
    /// exactness `PooledDirty` requires.
    pub fn block_coverage(&self, log2_block: usize, n_blocks: usize) -> Vec<BlockCoverage> {
        let block_bits = 1usize << log2_block;
        let mut out = vec![BlockCoverage::Dead; n_blocks];
        for (s, e) in self.useful_intervals() {
            let mut blk = s >> log2_block;
            while blk < n_blocks && (blk << log2_block) < e {
                let b0 = blk << log2_block;
                let (cs, ce) = (s.max(b0), e.min(b0 + block_bits));
                let piece = (cs - b0, ce - b0);
                out[blk] = match std::mem::replace(&mut out[blk], BlockCoverage::Dead) {
                    _ if piece == (0, block_bits) => BlockCoverage::Full,
                    BlockCoverage::Dead => BlockCoverage::Partial(vec![piece]),
                    BlockCoverage::Partial(mut v) => {
                        // Intervals are sorted and merged, so pieces arrive
                        // in order and never touch (a touching pair would
                        // have merged upstream).
                        v.push(piece);
                        BlockCoverage::Partial(v)
                    }
                    BlockCoverage::Full => unreachable!("a full block admits no second piece"),
                };
                blk += 1;
            }
        }
        out
    }

    /// Sorted, merged list of useful bit intervals `[start, end)` — the
    /// semantic content of the spec (everything outside is declared zero).
    /// Consumed by the general (multi-run) kernel paths and by tests; cost is
    /// O(total blocks), fine off the single-run hot path.
    pub fn useful_intervals(&self) -> Vec<(usize, usize)> {
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        let mut offset = 0usize;
        for run in &self.runs {
            let block_bits = 1usize << run.k_log;
            if run.useful_bits_per_block > 0 {
                for blk in 0..run.n_blocks {
                    let start = offset + blk * block_bits;
                    let end = start + run.useful_bits_per_block;
                    match intervals.last_mut() {
                        Some((_, prev_end)) if *prev_end == start => *prev_end = end,
                        _ => intervals.push((start, end)),
                    }
                }
            }
            offset += run.extent_bits();
        }
        intervals
    }
}

/// One block's standing in a [`PaddingSpec::block_coverage`] map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockCoverage {
    /// No useful bit — skippable outright (honest contribution zero).
    Dead,
    /// Every bit useful — read in place.
    Full,
    /// The block-local useful bit ranges `[start, end)`, sorted and
    /// non-touching — cleanse into zeroed scratch before reading.
    Partial(Vec<(usize, usize)>),
}

/// Copy one block's useful bits — block-local bit `ranges` — from `src` at
/// byte offset `base_byte` into `dst` (zeroed here first). Edge bits of a
/// non-byte-aligned range are masked, so a declared-dead neighbor bit never
/// leaks through: after this, `dst` is the block a fully honest prover
/// would have had, whatever garbage the pooled source carries.
pub fn cleanse_block(src: &[u8], base_byte: usize, ranges: &[(usize, usize)], dst: &mut [u8]) {
    dst.fill(0);
    for &(s, e) in ranges {
        let (sb, eb) = (s / 8, e.div_ceil(8));
        for i in sb..eb {
            let mut mask = 0xFFu8;
            if i == s / 8 {
                mask &= 0xFFu8 << (s % 8);
            }
            if e % 8 != 0 && i == e / 8 {
                mask &= !(0xFFu8 << (e % 8));
            }
            dst[i] |= src[base_byte + i] & mask;
        }
    }
}

// ---------------------------------------------------------------------------
// Public types: claim, proof, error.
// ---------------------------------------------------------------------------

/// Evaluation claims on the multilinear extensions of a, b, c. **Note that
/// `a_eval`/`b_eval` and `c_eval` are claimed at *different points*** —
/// extract_c separates C from the AB sumcheck:
///
/// - `a_eval`, `b_eval` are at `(z, mlv_challenges)` — the AB sumcheck binds
///   the rest variables one at a time to fresh `ρ_r` challenges.
/// - `c_eval` is at `(z, r_rest)` — C is linear, so its eq-weighted sum
///   collapses immediately to an MLE evaluation at the original eq weights;
///   no per-round folding needed. Here `r_rest = r[K_SKIP..m]` from the
///   zerocheck challenge.
///
/// The downstream caller (R1CS prover + PCS) opens each commitment at its
/// own claim point. Two openings for a, b at the same point; one for c at
/// a different point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaim {
    /// Univariate-skip challenge sampled after round 1 (binds the K_SKIP
    /// skip variables).
    pub z: F128,
    /// AB sumcheck bind challenges, one per multilinear round; length = `m - K_SKIP`.
    pub mlv_challenges: Vec<F128>,
    /// Eq weights for the rest variables = the zerocheck challenge restricted
    /// to `r[K_SKIP..m]`. This is the *rest part of the c-claim's point*.
    /// Length = `m - K_SKIP`.
    pub r_rest: Vec<F128>,
    /// `â(z, mlv_challenges)`.
    pub a_eval: F128,
    /// `b̂(z, mlv_challenges)`.
    pub b_eval: F128,
    /// `ĉ(z, r_rest)` — at a *different point* than a_eval, b_eval.
    pub c_eval: F128,
}

/// All round messages the prover sends, in order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckProof {
    /// Round 1 (univariate skip): `P^{AB}(λ)` for λ ∈ Λ, length 2^K_SKIP.
    pub round1_ab: Vec<F128>,
    /// Round 1 (extract_c): `P^C(λ)` for λ ∈ Λ, length 2^K_SKIP. Sent separately
    /// from `round1_ab` so the verifier can evaluate the C-claim immediately
    /// and skip the C-column in all subsequent rounds.
    pub round1_c: Vec<F128>,
    /// Multilinear sumcheck rounds: each entry is `(P_r(1), P_r(∞))` via the
    /// Karatsuba ∞-trick. Length = `m - K_SKIP`.
    pub multilinear_rounds: Vec<(F128, F128)>,
    /// Final MLE evaluations sent at the end of the protocol.
    pub final_a_eval: F128,
    pub final_b_eval: F128,
    pub final_c_eval: F128,
    /// PoW nonces in transcript order: initial eq point, skip point, then
    /// one nonce per multilinear round.  Empty under
    /// [`ZerocheckGrinding::disabled`].
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
}

/// Reasons the verifier may reject a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `log_n` doesn't satisfy `log_n >= K_SKIP`.
    LogNTooSmall { log_n: usize, k_skip: usize },
    /// Round-1 messages have the wrong length (expected `2^K_SKIP`).
    BadRound1Length { expected: usize, got: usize },
    /// Wrong number of multilinear-round messages (expected `log_n - K_SKIP`).
    BadMultilinearRoundsLength { expected: usize, got: usize },
    /// The supplied nonce vector does not match the configured grinding
    /// schedule.  This is checked before replaying the transcript so a
    /// malformed proof cannot shift nonce-to-challenge alignment.
    BadGrindingNonceCount { expected: usize, got: usize },
    /// A nonce does not satisfy the PoW at the transcript position where its
    /// corresponding challenge is sampled.
    InvalidGrindingNonce { which: &'static str },
    /// `proof.final_c_eval` doesn't match the verifier's reconstruction
    /// `C_s · interpolate_at_z_on_lambda(round1_c, k_skip, z)`. Catches
    /// dishonesty in the round-1 C message or in the final c-eval claim.
    CEvalMismatch,
    /// The AB sumcheck final consistency check failed: the inner running
    /// claim after all rounds should equal `final_a_eval · final_b_eval`.
    /// Any inconsistency in `round1_ab`, in a multilinear round's
    /// `(P_r(1), P_r(∞))`, or in `final_a_eval` / `final_b_eval` propagates
    /// to this check.
    SumcheckFinalFailed,
}

// ---------------------------------------------------------------------------
// API: prove / verify.
// ---------------------------------------------------------------------------

/// Prove that `a(y) · b(y) ⊕ c(y) = 0` for all `y ∈ {0,1}^m`.
///
/// Inputs are LSB-first bit-packed byte vectors (each of length `2^m / 8`).
/// `m ≥ K_SKIP + N_INNER` (= 13). `challenger` supplies all verifier
/// randomness; the prover absorbs each of its messages into the challenger
/// before sampling the next challenge so the verifier (using the same
/// challenger implementation in lockstep) derives identical challenges.
///
/// Returns:
///   - the [`ZerocheckProof`] (raw round messages), and
///   - the [`ZerocheckClaim`] the higher-level caller will pass to its PCS.
pub fn prove_packed<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded_with_grinding(
        a_packed,
        b_packed,
        c_packed,
        m,
        &PaddingSpec::dense(m),
        ZerocheckGrinding::disabled(),
        challenger,
    )
}

/// [`prove_packed`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_packed_with_grinding<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    grinding: ZerocheckGrinding,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded_with_grinding(
        a_packed,
        b_packed,
        c_packed,
        m,
        &PaddingSpec::dense(m),
        grinding,
        challenger,
    )
}

/// Same as [`prove_packed`] but lets the caller declare a run-list padding
/// pattern so URM can skip work for chunks that fall entirely in zero
/// padding (or in the trailing gap after the last run). Output is
/// byte-identical to the dense path when the padding bits are honestly zero.
pub fn prove_packed_padded<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded_with_grinding(
        a_packed,
        b_packed,
        c_packed,
        m,
        padding,
        ZerocheckGrinding::disabled(),
        challenger,
    )
}

/// [`prove_packed_padded`] with an explicit Fiat--Shamir grinding policy.
pub fn prove_packed_padded_with_grinding<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    grinding: ZerocheckGrinding,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    let (proof, claim, _) = prove_packed_padded_inner(
        a_packed, b_packed, c_packed, m, padding, grinding, false, challenger,
    );
    (proof, claim)
}

/// Variant of [`prove_packed_padded`] that ALSO returns the canonical
/// `s_hat_v_c` produced by the fused two-bank round-1 kernel. The downstream
/// PCS open uses this to skip `fold_1b_rows` for the c-claim — see
/// [`crate::pcs::ring_switch::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`].
///
/// Wire output `(ZerocheckProof, ZerocheckClaim)` is byte-identical to
/// [`prove_packed_padded`].
pub fn prove_packed_padded_capture_s_hat_v_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    prove_packed_padded_capture_s_hat_v_c_with_grinding(
        a_packed,
        b_packed,
        c_packed,
        m,
        padding,
        ZerocheckGrinding::disabled(),
        challenger,
    )
}

/// [`prove_packed_padded_capture_s_hat_v_c`] with an explicit grinding
/// policy.  The returned `s_hat_v_c` is unchanged; only the Fiat--Shamir
/// transcript and nonce payload differ when grinding is enabled.
pub fn prove_packed_padded_capture_s_hat_v_c_with_grinding<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    grinding: ZerocheckGrinding,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        a_packed, b_packed, c_packed, m, padding, grinding, true, challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_packed_padded_inner<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    grinding: ZerocheckGrinding,
    capture_s_hat_v_c: bool,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Option<Vec<F128>>) {
    let k_skip = K_SKIP;
    const N_INNER: usize = 7; // 3 small + 4 medium fixed-constant eq dims
    assert!(
        m >= k_skip + N_INNER,
        "prove requires m >= k_skip + N_INNER (= {})",
        k_skip + N_INNER
    );
    let expected_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected_bytes);
    assert_eq!(b_packed.len(), expected_bytes);
    assert_eq!(c_packed.len(), expected_bytes);
    let n_mlv = m - k_skip;
    let mut grinding_nonces = Vec::with_capacity(grinding.nonce_count(m));

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- 1. Sample r (with protocol-fixed constants in the inner 7 dims) ----
    //
    // r layout:
    //   r[0..k_skip]                — sampled (used by verifier for the
    //                                  final check at S; not by the URM)
    //   r[k_skip..k_skip+3]         — protocol small-eq constants φ_8(0xF7..)
    //   r[k_skip+3..k_skip+7]       — protocol medium-eq constants β_i
    //   r[k_skip+7..m]              — sampled (the "outer" eq weights for
    //                                  the URM and multilinear rounds)
    let r_skip = if let Some(bits) = grinding.initial_bits(m) {
        let (nonce, r_skip) = challenger.grind_pow_and_sample_f128_vec(bits, k_skip);
        grinding_nonces.push(nonce);
        r_skip
    } else {
        challenger.sample_f128_vec(k_skip)
    };
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- 3. Round 1: URM (extract_c, parallel) ----
    //
    // The optimized URM drops a `C_s = φ_8(0x1C)` scalar from its accumulators
    // (a prover-side optimization tied to the small-eq trick — see the
    // C_s factor analysis in `univariate_skip_optimized`). The wire format
    // must be in "naive" convention so the verifier doesn't need to know
    // about this internal optimization; we restore the C_s factor here.
    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let t_round1 = std::time::Instant::now();
    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let (round1_ab_opt, round1_c_opt, s_hat_v_c) = if capture_s_hat_v_c {
        let (ab, c, s) =
            crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                a_packed,
                b_packed,
                c_packed,
                m,
                k_skip,
                &r,
                &inv_table,
                padding,
            );
        (ab, c, Some(s))
    } else {
        let (ab, c) = round1_shift_reduce_extract_c_packed_padded(
            a_packed, b_packed, c_packed, m, k_skip, &r, &inv_table, padding,
        );
        (ab, c, None)
    };
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms",
            t_round1.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- 4. Observe round-1 message, sample z (URM fold point) ----
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = if let Some(bits) = grinding.skip_bits() {
        let (nonce, z) = challenger.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        z
    } else {
        challenger.sample_f128()
    };

    // ---- 5. c_eval = ĉ(z, r_rest) via interpolation of round1_c at z ----
    //
    // round1_c (now in naive convention) carries `P^C(λ) = Σ_x eq(r_rest, x) · ĉ(λ, x)`
    // as its 2^k_skip evaluations on Λ. Interpolating to λ=z gives
    // `ĉ(z, r_rest)` directly (the eq-weighted sum collapses to the MLE
    // evaluation because ĉ is linear). This is **the c-claim** — at point
    // `(z, r_rest)`, *not* `(z, ρ-values)`. ~64 F128 muls + Lagrange weights.
    let final_c_eval = interpolate_at_z_on_lambda(&round1_c, k_skip, z);

    // ---- 6. Round 2: fused fold + first multilinear message ----
    //
    // Convention A wrapping: pass `mlv_arg[0] = ONE` so the function's output
    // `mlv_arg[0] · G(1)` becomes the bare `G(1)` we send on the wire. The
    // verifier samples ρ_1 after observing this message.
    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    // Support-proportional prover (M6): under a multi-run count-derived spec
    // the post-URM tables are zero outside the declared support (the live
    // interval list). While that support is sparse (live·16 ≤ n), round 2 and
    // the tail rounds fold/evaluate over the live intervals only — every
    // skipped term carries an `a·b` factor of zero, so all messages and folded
    // values are byte-identical to the dense path.
    //
    // The buffers hold the LIVE SPAN, not the padded domain: dead positions get
    // no storage at all, so both the fold cost and the footprint are
    // count-derived and the phase leaves the capacity axis entirely. When the
    // tail leaves the sparse path — the live fraction crosses the gate, or the
    // domain drops below the fused threshold and the naive kernels need global
    // indexing — `expand_to_dense` scatters the live span back into a full
    // padded buffer once. See [`multilinear::LiveLayout`].
    let sparse_from_round2 = padding.as_single_run().is_none() && {
        let list = padding.useful_block_intervals(k_skip);
        let live_elems: usize = list.iter().map(|&(s, e)| e - s).sum();
        let n_out = 1usize << n_mlv;
        n_out >= 8 && live_elems * sparse_tail_gate() <= n_out
    };
    // `store` is the compaction map for the tail buffers: `Some` means they
    // hold ONLY the live span, `None` means the full padded domain. `domain`
    // is the logical multilinear size and halves every round regardless —
    // under compaction it is no longer `a_mlv.len()`.
    let mut domain = 1usize << n_mlv;
    let (mut a_mlv, mut b_mlv, msg_1, msg_inf, mut store) = if sparse_from_round2 {
        let (a, b, m1, mi, st) = multilinear::uni_skip_fold_and_round_pair_runs_sparse(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        );
        (a, b, m1, mi, Some(st))
    } else {
        let (a, b, m1, mi) = uni_skip_fold_and_round_pair_optimized_packed_padded(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        );
        (a, b, m1, mi, None)
    };

    if zc_timing {
        eprintln!(
            "[zc-timing] round2 fused fold: {:.2} ms",
            t_round2.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_tail = std::time::Instant::now();
    let mut multilinear_msgs = Vec::with_capacity(n_mlv);
    multilinear_msgs.push((msg_1, msg_inf));
    challenger.observe_f128(msg_1);
    challenger.observe_f128(msg_inf);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    let rho = if let Some(bits) = grinding.multilinear_round_bits() {
        let (nonce, rho) = challenger.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        rho
    } else {
        challenger.sample_f128()
    };
    mlv_rhos.push(rho);

    // ---- 7. Rounds 3..(n_mlv + 1) — AB only (c is done) ----
    //
    // Iter i: fold (a, b) at ρ_{i+1}, compute round (i+3) message, sample
    // ρ_{i+2}. Use the fused parallel path while log_n ≥ 10; below that the
    // SplitEqGhash inner can't form lo_size ≥ 2, so we fall back to
    // fold_in_place_pair + round_pair_naive.
    //
    // Ping-pong scratch buffers for the fused path: each fused round folds
    // (a_mlv, b_mlv) of size N into size N/2. Rather than allocating — and,
    // worse, `munmap`-ing, which is single-threaded and caps the tail's
    // parallel speedup — a fresh 64 MB buffer per round, we alternate between
    // two persistent buffers. Scratch capacity = N/2 (the largest fused
    // output); only needed when the first round is actually fused.
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        (
            crate::scratch::take_f128(n_in / 2),
            crate::scratch::take_f128(n_in / 2),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // `sparse_dirty` tracks whether the current buffers' dead regions hold
    // unwritten scratch. Under compaction the dead regions are not stored at
    // all, so leaving the sparse path means EXPANDING (scatter + zero-fill)
    // rather than zeroing in place.
    let mut sparse_dirty = sparse_from_round2;

    for i in 0..(n_mlv - 1) {
        let rho_prev = mlv_rhos[i];
        // From the LOGICAL domain, not the buffer: under live-span storage
        // `a_mlv.len()` is the compacted length and need not be a power of two.
        let log_n_before = domain.trailing_zeros() as usize;

        // r_next for the next round's message: length log_n_before - 1.
        // r_next[0] = ONE (Convention A factor); r_next[1..] are the eq
        // weights for the remaining variables = r[k_skip + i + 2..m].
        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        // The sparse path also requires the fused domain (>= 1024): below it
        // the naive kernels index globally, so compaction must be undone.
        let use_sparse = store
            .as_ref()
            .is_some_and(|st| domain >= 1024 && st.len() * sparse_tail_gate() <= domain);
        if !use_sparse
            && let Some(st) = store.take()
            && sparse_dirty
        {
            // Back to global indexing: scatter the live span into a full
            // padded buffer and zero the rest.
            let a_full = multilinear::expand_to_dense(&a_mlv, &st, domain);
            let b_full = multilinear::expand_to_dense(&b_mlv, &st, domain);
            crate::scratch::give_f128(std::mem::replace(&mut a_mlv, a_full));
            crate::scratch::give_f128(std::mem::replace(&mut b_mlv, b_full));
            // The ping-pong scratch shrank toward the compacted cap while
            // the tail ran sparse; the dense fold below slices
            // `a_nxt[..domain/2]`, so the scratch must re-grow with the
            // buffers (the recorded gate=4 panic: range end 1024 out of
            // range for slice of length 678). A fragmented near-full spec
            // can force this exit at `domain >= 1024` even at gate = 1 —
            // interval ends round `st.len()` outward past the domain.
            if a_nxt.len() < domain / 2 {
                crate::scratch::give_f128(a_nxt);
                crate::scratch::give_f128(b_nxt);
                a_nxt = crate::scratch::take_f128(domain / 2);
                b_nxt = crate::scratch::take_f128(domain / 2);
            }
            sparse_dirty = false;
        }

        let (m1, mi) = if use_sparse {
            let st = store.as_ref().expect("use_sparse implies store");
            // Output storage is bounded by the input's: shrinking pairs can
            // only round outward by one slot per interval end.
            let cap = st.len() + 2 * st.intervals().len() + 2;
            if a_nxt.len() < cap {
                crate::scratch::give_f128(a_nxt);
                crate::scratch::give_f128(b_nxt);
                a_nxt = crate::scratch::take_f128(cap);
                b_nxt = crate::scratch::take_f128(cap);
            }
            let (m1, mi, store_out) = fold_and_round_pair_sparse_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..cap],
                &mut b_nxt[..cap],
                rho_prev,
                &r_next,
                st,
                domain,
            );
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(store_out.len());
            b_mlv.truncate(store_out.len());
            store = Some(store_out);
            sparse_dirty = true;
            (m1, mi)
        } else if log_n_before >= 10 {
            let half = a_mlv.len() / 2;
            let (m1, mi) = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            // Swap current <-> scratch, then shrink the new current to the
            // folded size. The old (larger) buffer becomes scratch; we only
            // ever write its leading `half` slots next round, so its stale
            // length is harmless.
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            (m1, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };

        domain /= 2;
        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        let rho = if let Some(bits) = grinding.multilinear_round_bits() {
            let (nonce, rho) = challenger.grind_pow_and_sample_f128(bits);
            grinding_nonces.push(nonce);
            rho
        } else {
            challenger.sample_f128()
        };
        mlv_rhos.push(rho);
    }
    debug_assert!(
        store.is_none(),
        "the tail must leave live-span storage before the final binding \
         (domain drops below the fused threshold, forcing expansion)"
    );

    // ---- 8. Final binding at ρ_{n_mlv} (the last challenge) ----
    let rho_last = *mlv_rhos.last().expect("at least one ρ sampled");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    debug_assert_eq!(a_mlv.len(), 1);
    debug_assert_eq!(b_mlv.len(), 1);

    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];

    // ---- Fiat–Shamir: bind the final â, b̂ claims into the transcript ----
    //
    // These two claims are reduced downstream by lincheck via a *single*
    // random-linear-combination check with coefficient α (`target = α·v_a + v_b`,
    // see `lincheck`). That batching is only sound if α is sampled *after*
    // (v_a, v_b) are committed to the transcript — otherwise a prover that knows
    // α can pick (v_a, v_b) to satisfy the one batched equation while violating
    // the individual checks. So observe them here, before any later challenge
    // (the next one drawn is lincheck's α). `final_c_eval` needs no observe — the
    // verifier recomputes it from the already-absorbed `round1_c`/`z` and rejects
    // on mismatch (see `verify`), so it is already transcript-bound.
    challenger.observe_f128(final_a_eval);
    challenger.observe_f128(final_b_eval);

    // Recycle the four tail buffers (the two len-1 survivors still own their
    // full round-2 capacity) for the next phase/prove.
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);

    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        );
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();

    let proof = ZerocheckProof {
        round1_ab,
        round1_c,
        multilinear_rounds: multilinear_msgs,
        final_a_eval,
        final_b_eval,
        final_c_eval,
        grinding_nonces,
    };
    let claim = ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    };
    (proof, claim, s_hat_v_c)
}

/// Verify a zerocheck proof for an instance over `{0,1}^log_n`.
///
/// Walks the challenger in lockstep with the prover, samples the same
/// challenges, and checks every round's consistency equation.
///
/// On accept: returns the [`ZerocheckClaim`] the caller must check against
/// its PCS opening of `â`, `b̂`, `ĉ`.
/// On reject: returns a [`VerifyError`] indicating which check failed.
pub fn verify<C: Challenger>(
    log_n: usize,
    proof: &ZerocheckProof,
    challenger: &mut C,
) -> Result<ZerocheckClaim, VerifyError> {
    verify_with_grinding(log_n, proof, ZerocheckGrinding::disabled(), challenger)
}

/// [`verify`] with an explicit Fiat--Shamir grinding policy.  The policy is
/// a verifier parameter, not prover-controlled proof metadata: it is part of
/// the security configuration agreed before verification starts.
pub fn verify_with_grinding<C: Challenger>(
    log_n: usize,
    proof: &ZerocheckProof,
    grinding: ZerocheckGrinding,
    challenger: &mut C,
) -> Result<ZerocheckClaim, VerifyError> {
    let m = log_n;
    let k_skip = K_SKIP;
    const N_INNER: usize = 7;

    if m < k_skip + N_INNER {
        return Err(VerifyError::LogNTooSmall { log_n: m, k_skip });
    }
    let n_mlv = m - k_skip;
    let ell = 1usize << k_skip;

    // ---- Shape checks ----
    if proof.round1_ab.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_ab.len(),
        });
    }
    if proof.round1_c.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_c.len(),
        });
    }
    if proof.multilinear_rounds.len() != n_mlv {
        return Err(VerifyError::BadMultilinearRoundsLength {
            expected: n_mlv,
            got: proof.multilinear_rounds.len(),
        });
    }
    if proof.grinding_nonces.len() != grinding.nonce_count(m) {
        return Err(VerifyError::BadGrindingNonceCount {
            expected: grinding.nonce_count(m),
            got: proof.grinding_nonces.len(),
        });
    }

    challenger.observe_label(b"flock-zerocheck-v0");
    let mut nonce_idx = 0usize;

    // ---- Re-derive r (in lockstep with prove_packed) ----
    let r_skip = if let Some(bits) = grinding.initial_bits(m) {
        let r_skip = challenger
            .verify_pow_and_sample_f128_vec(proof.grinding_nonces[nonce_idx], bits, k_skip)
            .ok_or(VerifyError::InvalidGrindingNonce { which: "initial" })?;
        nonce_idx += 1;
        r_skip
    } else {
        challenger.sample_f128_vec(k_skip)
    };
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- Observe round-1 messages, sample z ----
    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);
    let z = if let Some(bits) = grinding.skip_bits() {
        let z = challenger
            .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
            .ok_or(VerifyError::InvalidGrindingNonce { which: "skip" })?;
        nonce_idx += 1;
        z
    } else {
        challenger.sample_f128()
    };

    // ---- Reconstruct ĉ(z, r_rest) from round1_c ----
    //
    // P^C has degree < 2^k_skip in λ (C is linear, summed against eq); ell
    // evaluations on Λ uniquely interpolate to z. round1_c is in naive
    // convention (the prover restored the C_s factor before sending), so
    // `ĉ(z, r_rest) = P^C(z)` directly.
    let computed_c_eval = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    if computed_c_eval != proof.final_c_eval {
        return Err(VerifyError::CEvalMismatch);
    }

    // ---- Reconstruct the initial AB running claim ----
    //
    // P^{AB}(z) requires the polynomial in λ of degree < 2·ell to be evaluated
    // at z. The prover sent only ell evaluations on Λ — not enough on its own.
    // The verifier uses the **zerocheck assumption** `P^{AB}(λ) + P^C(λ) = 0`
    // for `λ ∈ S`. Together with the ell Λ-evaluations of the combined
    // polynomial, that's 2·ell evaluations — enough to interpolate the
    // combined polynomial at z. Then `P^{AB}(z) = P^{combined}(z) − P^C(z)`,
    // which in char-2 is `P^{combined}(z) + P^C(z)`.
    //
    // If the prover's witness is dishonest the S-zero assumption fails, the
    // reconstructed c_0 is wrong, and the running-claim chain ends at a value
    // inconsistent with `â · b̂`. We catch that at the final sumcheck check.
    let combined_at_lambda: Vec<F128> = proof
        .round1_ab
        .iter()
        .zip(&proof.round1_c)
        .map(|(x, y)| *x + *y)
        .collect();
    let combined_at_z = interpolate_at_z_combined(&combined_at_lambda, k_skip, z);
    // `P^C(z)` was already computed above as `computed_c_eval` — same function,
    // same arguments. Recomputing it cost a second Λ-interpolation: under the
    // textbook weights that was 8,256 constraints, 4.5% of a BLAKE3 boolean
    // verify's entire arithmetic, for a value already in hand (measured,
    // `benches/verifier_mul_count.rs`). Sub-millisecond natively, which is why
    // it went unnoticed.
    let mut c_running = combined_at_z + computed_c_eval;

    // ---- Multilinear sumcheck chain ----
    //
    // The propagated running claim is the *inner* polynomial value G(ρ),
    // not the full per-round polynomial P(ρ) = eq(r_eq, ρ) · G(ρ). The eq
    // factor for the just-bound variable is absorbed by the next round's
    // consistency check via the identity
    //   G_{r-1}(ρ_{r-1}) = (1 + r_eq_r) · G_r(0) + r_eq_r · G_r(1).
    //
    // Round r (0-indexed i = r − 2) binds the i-th rest variable with eq weight
    // r[k_skip + i]. The prover sends `(G(1), G(∞))` (Convention A — no
    // factor). Verifier:
    //   1. reconstruct G(0) from consistency `c_running = (1+r_eq)·G(0) + r_eq·G(1)`,
    //   2. observe message, sample ρ_i,
    //   3. update `c_running ← G(ρ_i)`,
    //      where `G(X) = G(0)·(1+X) + G(1)·X + G(∞)·X·(X+1)` (char-2 quadratic
    //      interpolation through G(0), G(1), G(∞)).
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    for (i, &(msg_1, msg_inf)) in proof.multilinear_rounds.iter().enumerate() {
        let r_eq = r[k_skip + i];
        let one_plus_r_eq = F128::ONE + r_eq;

        let g1 = msg_1;
        let g_inf = msg_inf;
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        challenger.observe_f128(msg_1);
        challenger.observe_f128(msg_inf);
        let rho = if let Some(bits) = grinding.multilinear_round_bits() {
            let rho = challenger
                .verify_pow_and_sample_f128(proof.grinding_nonces[nonce_idx], bits)
                .ok_or(VerifyError::InvalidGrindingNonce {
                    which: "multilinear",
                })?;
            nonce_idx += 1;
            rho
        } else {
            challenger.sample_f128()
        };
        mlv_rhos.push(rho);

        let one_plus_rho = F128::ONE + rho;
        // G(ρ) = G(0)·(1+ρ) + G(1)·ρ + G(∞)·ρ·(1+ρ).
        c_running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());

    // ---- AB sumcheck final consistency ----
    //
    // After all variables are bound, the inner running claim is just the
    // polynomial without the eq weighting:
    //   G_final(ρ_all) = â(z, ρ) · b̂(z, ρ) = final_a_eval · final_b_eval.
    // (The eq factors were absorbed round-by-round into the consistency checks,
    // never accumulating into the running claim.)
    let r_rest: Vec<F128> = r[k_skip..].to_vec();
    let expected_final = proof.final_a_eval * proof.final_b_eval;
    if c_running != expected_final {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    // ---- Fiat–Shamir: bind the final â, b̂ claims (mirrors `prove_packed_padded_inner`) ----
    //
    // Must observe at the same transcript position as the prover, before the
    // next challenge (lincheck's α) is drawn, so the α-batched reduction of
    // these two claims is sound. `final_c_eval` is already bound via the
    // recompute-and-compare above, so it is not observed.
    challenger.observe_f128(proof.final_a_eval);
    challenger.observe_f128(proof.final_b_eval);

    Ok(ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: proof.final_a_eval,
        b_eval: proof.final_b_eval,
        c_eval: proof.final_c_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    use crate::test_rng::Rng;

    /// Pack three Boolean vectors into the (a_packed, b_packed, c_packed)
    /// shape that `prove_packed` consumes.
    fn pack_abc(a: &[bool], b: &[bool], c: &[bool]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use univariate_skip::pack_bits;
        (pack_bits(a), pack_bits(b), pack_bits(c))
    }

    /// `prove` runs end-to-end at the smallest valid m (= k_skip + N_INNER = 13)
    /// without panicking, and produces output of the right shape.
    ///
    /// We can't yet check the proof is *accepted* (verify is a stub), but the
    /// structural sanity here catches:
    ///   - mismatched challenger observe/sample sequence
    ///   - wrong slice lengths in r / mlv_arg / r_next at any round
    ///   - any unreachable assert in the underlying functions
    #[test]
    fn prove_runs_end_to_end() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Honest witness: c = a AND b, so a·b ⊕ c = 0 on the hypercube.
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut challenger = FsChallenger::new(b"flock-test-v0");
            let (proof, claim) = prove_packed(&a_p, &b_p, &c_p, m, &mut challenger);

            // Shape checks.
            assert_eq!(proof.round1_ab.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.round1_c.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.multilinear_rounds.len(), m - K_SKIP, "m={m}");
            assert_eq!(claim.mlv_challenges.len(), m - K_SKIP, "m={m}");

            // Claim's eval fields agree with the proof's final evals.
            assert_eq!(claim.a_eval, proof.final_a_eval, "m={m}");
            assert_eq!(claim.b_eval, proof.final_b_eval, "m={m}");
            assert_eq!(claim.c_eval, proof.final_c_eval, "m={m}");
        }
    }

    /// **Prove→verify roundtrip**: an honest proof verifies cleanly, and the
    /// claim returned by `verify` is byte-for-byte equal to the claim returned
    /// by `prove`.
    #[test]
    fn prove_verify_roundtrip_honest() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(1000 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &proof, &mut ch_verify);
            let claim_v = result.unwrap_or_else(|e| panic!("verify rejected at m={m}: {e:?}"));

            assert_eq!(claim_p, claim_v, "claim mismatch at m={m}");
        }
    }

    /// The 128-bit-per-error grinding schedule is carried on the proof,
    /// checked before every challenge it protects, and recorded as ordinary
    /// `Pow` operations for the recursion transcript tape.
    #[test]
    fn per_challenge_grinding_roundtrip_and_tape() {
        use crate::transcript_record::{RecordingChallenger, TranscriptOp};

        let m = 13;
        let mut rng = Rng::new(0x1280_0001);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let grinding = ZerocheckGrinding::per_challenge_128();

        let mut ch_prove = FsChallenger::new(b"flock-zc-grinding-v0");
        let (proof, claim_p) =
            prove_packed_with_grinding(&a_p, &b_p, &c_p, m, grinding, &mut ch_prove);
        assert_eq!(proof.grinding_nonces.len(), grinding.nonce_count(m));
        assert_eq!(grinding.initial_bits(m), Some(4));
        assert_eq!(grinding.skip_bits(), Some(7));
        assert_eq!(grinding.multilinear_round_bits(), Some(2));

        let mut rec = RecordingChallenger::new(FsChallenger::new(b"flock-zc-grinding-v0"));
        let claim_v = verify_with_grinding(m, &proof, grinding, &mut rec)
            .expect("grinded honest proof must verify");
        assert_eq!(claim_p, claim_v);

        let pow_bits: Vec<u32> = rec
            .shape()
            .ops()
            .iter()
            .filter_map(|op| match op {
                TranscriptOp::Pow { bits } | TranscriptOp::LegacyPow { bits } => Some(*bits),
                _ => None,
            })
            .collect();
        let mut expected = vec![4, 7];
        expected.extend(std::iter::repeat_n(2, m - K_SKIP));
        assert_eq!(
            pow_bits, expected,
            "one PoW immediately precedes each protected challenge"
        );

        let mut missing = proof.clone();
        missing.grinding_nonces.pop();
        let mut ch_bad = FsChallenger::new(b"flock-zc-grinding-v0");
        assert!(matches!(
            verify_with_grinding(m, &missing, grinding, &mut ch_bad),
            Err(VerifyError::BadGrindingNonceCount { .. })
        ));
    }

    /// **Verify rejects byte-mutated proofs.** Walk each component of the
    /// proof and flip one F128 entry; the verifier must return an `Err`
    /// (rather than panicking or silently accepting).
    #[test]
    fn verify_rejects_mutations() {
        let m = 14;
        let mut rng = Rng::new(5050);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let _seed: u64 = 0xDEAD_BEEF;
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Each closure returns a mutated copy; verify must reject all of them.
        let mutations: Vec<(&str, Box<dyn Fn(&ZerocheckProof) -> ZerocheckProof>)> = vec![
            (
                "round1_ab[0] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_ab[0].lo ^= 1;
                    q
                }),
            ),
            (
                "round1_c[5] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_c[5].lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[0].0 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.multilinear_rounds[0].0.lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[2].1 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    let last = q.multilinear_rounds.len() / 2;
                    q.multilinear_rounds[last].1.hi ^= 1;
                    q
                }),
            ),
            (
                "final_a_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_a_eval.lo ^= 1;
                    q
                }),
            ),
            (
                "final_c_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_c_eval.hi ^= 1;
                    q
                }),
            ),
        ];

        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &bad, &mut ch);
            assert!(
                result.is_err(),
                "verify accepted mutated proof ({label}) — should have rejected"
            );
        }
    }

    /// Shape rejections: too-short round1, wrong number of multilinear rounds.
    #[test]
    fn verify_rejects_shape_errors() {
        let m = 14;
        let mut rng = Rng::new(606);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Truncate round1_ab.
        let mut bad = proof.clone();
        bad.round1_ab.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadRound1Length { .. })
        ));

        // Truncate multilinear rounds.
        let mut bad = proof.clone();
        bad.multilinear_rounds.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadMultilinearRoundsLength { .. })
        ));

        // log_n too small.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(K_SKIP + 6, &proof, &mut ch),
            Err(VerifyError::LogNTooSmall { .. })
        ));
    }

    /// AUDIT: a FALSE statement (c ≠ a·b at some hypercube point) must be
    /// rejected, even though the prover follows the honest algorithm on its
    /// (dishonest) witness.
    #[test]
    fn audit_false_statement_rejected() {
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(7777 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Correct c, then corrupt ONE bit so a·b ⊕ c ≠ 0 somewhere.
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            c[3] = !c[3];

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "verify ACCEPTED a false statement at m={m}: {res:?}"
            );
        }
    }

    /// AUDIT: flipping any round's `msg_inf` (the degree-2 / ∞ coefficient)
    /// must be rejected. `msg_inf` is observed into the transcript, so the
    /// tamper both reshuffles subsequent ρ challenges and breaks the
    /// running-claim chain — either way the final check fails.
    #[test]
    fn audit_round_msg_inf_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(424242);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // For each round, flip msg_inf to a different value. Because msg_inf
        // is observed into the transcript, this reshuffles subsequent rho's;
        // a sound verifier should reject (overwhelming probability).
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].1 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &bad, &mut ch);
            assert!(res.is_err(), "msg_inf tamper at round {idx} ACCEPTED");
        }
    }

    /// AUDIT: the LAST round's `msg_inf` must be constrained — a common
    /// off-by-one is to leave the final round's leading coefficient unchecked.
    /// Kept separate from the all-rounds loop above so a regression here points
    /// straight at the final-round binding.
    #[test]
    fn audit_last_round_inf_constrained() {
        let m = 13;
        let mut rng = Rng::new(98765);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let last = proof.multilinear_rounds.len() - 1;
        let mut bad = proof.clone();
        bad.multilinear_rounds[last].1 += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch).is_err(),
            "last-round msg_inf unconstrained"
        );
    }

    /// AUDIT (Fiat–Shamir binding of the final â, b̂ claims). Regression test
    /// for the gap where `final_a_eval`/`final_b_eval` were not observed into
    /// the transcript.
    ///
    /// Downstream, lincheck reduces these two claims via a *single* random-
    /// linear-combination check (`target = α·v_a + v_b`). That batching is only
    /// sound if α is sampled *after* the claims are bound to the transcript —
    /// otherwise a prover that already knows α can pick (v_a, v_b) to satisfy
    /// the one batched equation while violating the individual ties.
    ///
    /// A *product-preserving* tamper `(â, b̂) → (â·t, b̂·t⁻¹)` leaves the
    /// zerocheck's own final check `c_running == â·b̂` satisfied, so `verify`
    /// still returns `Ok` — the zerocheck alone is blind to it. The defense is
    /// that both claims are now observed last in the transcript, so the next
    /// challenge (the slot lincheck draws α from) must diverge from the honest
    /// run. This assertion FAILS before the observe was added (identical
    /// post-state) and passes now.
    #[test]
    fn audit_final_ab_claims_bound_to_transcript() {
        let m = 14;
        let mut rng = Rng::new(0xF1A7_5A11);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Honest verify, then capture the next challenge the transcript feeds
        // downstream — this is exactly the slot lincheck samples α from.
        let mut ch_honest = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &proof, &mut ch_honest).is_ok(),
            "honest verify rejected"
        );
        let alpha_honest = ch_honest.sample_f128();

        // Product-preserving tamper: â' = â·t, b̂' = b̂·t⁻¹ ⇒ â'·b̂' = â·b̂, so the
        // zerocheck's `c_running == â·b̂` check still holds for the tampered pair.
        let t = F128 {
            lo: 0x0123_4567_89ab_cdef,
            hi: 0xfedc_ba98_7654_3210,
        };
        assert!(t != F128::ZERO && t != F128::ONE, "t must be nontrivial");
        let mut bad = proof.clone();
        bad.final_a_eval *= t;
        bad.final_b_eval *= t.inv();
        assert_ne!(bad.final_a_eval, proof.final_a_eval, "tamper must change â");
        assert_ne!(bad.final_b_eval, proof.final_b_eval, "tamper must change b̂");
        assert_eq!(
            bad.final_a_eval * bad.final_b_eval,
            proof.final_a_eval * proof.final_b_eval,
            "tamper must preserve the product",
        );

        // The zerocheck's own checks are blind to a product-preserving tamper:
        // verify still ACCEPTS. This is precisely the gap the FS binding closes —
        // the tamper is caught only because the claims now move the transcript.
        let mut ch_tampered = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch_tampered).is_ok(),
            "product-preserving tamper rejected by zerocheck's own checks (unexpected)",
        );
        let alpha_tampered = ch_tampered.sample_f128();

        // The fix: observing â, b̂ makes the downstream challenge depend on them,
        // so lincheck's α (and everything after) diverges and rejects the
        // tampered pair. Before the fix these challenges were equal.
        assert_ne!(
            alpha_honest, alpha_tampered,
            "final â/b̂ claims are NOT bound into the transcript: a product-preserving \
             tamper leaves the downstream challenge unchanged, breaking lincheck's \
             α-batched reduction of (v_a, v_b)",
        );
    }

    /// AUDIT: many random false witnesses must all be rejected. Stronger than a
    /// single corruption — exercises the full prove→verify path on statements
    /// that are false at varying numbers of hypercube points.
    #[test]
    fn audit_many_false_statements_rejected() {
        let m = 13;
        for seed in 0..20u64 {
            let mut rng = Rng::new(0xBADC0DE ^ seed);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            // Flip a random number of bits (1..=4).
            let nflip = 1 + (rng.next_u64() as usize % 4);
            for _ in 0..nflip {
                let idx = rng.next_u64() as usize % c.len();
                c[idx] = !c[idx];
            }
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "false statement (seed={seed}) ACCEPTED: {res:?}"
            );
        }
    }

    /// AUDIT: tamper msg_1 in each round; must reject.
    #[test]
    fn audit_round_msg_1_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(31415);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].0 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            assert!(
                verify(m, &bad, &mut ch).is_err(),
                "msg_1 tamper round {idx} ACCEPTED"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Run-list PaddingSpec.
    // -----------------------------------------------------------------------

    /// The block-coverage map (the AG kernels' gating grid) classifies
    /// every 8192-bit block correctly — full runs, bit-granular partial
    /// tails, dead gaps, runs straddling the grid, mid-block starts, and
    /// multiple pieces per block — and [`cleanse_block`] reproduces exactly
    /// the honest block (dead bits zero, edge bits masked) from a dirty
    /// source.
    #[test]
    fn block_coverage_and_cleanse() {
        let spec = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 1 << 13,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 3001,
                n_blocks: 1,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 0,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 14,
                useful_bits_per_block: 9000,
                n_blocks: 1,
            },
        ]);
        let cov = spec.block_coverage(13, 8);
        assert_eq!(
            cov,
            vec![
                BlockCoverage::Full,
                BlockCoverage::Full,
                BlockCoverage::Partial(vec![(0, 3001)]),
                BlockCoverage::Dead,
                BlockCoverage::Dead,
                BlockCoverage::Full,
                BlockCoverage::Partial(vec![(0, 9000 - 8192)]),
                BlockCoverage::Dead,
            ]
        );
        // Sub-grid runs: a dead prefix pushes an interval to a mid-block
        // start, and two disjoint intervals land in ONE grid block.
        let sub = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 1000,
                n_blocks: 1,
            },
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 2000,
                n_blocks: 1,
            },
        ]);
        assert_eq!(
            sub.block_coverage(13, 2),
            vec![
                BlockCoverage::Partial(vec![(0, 1000), (4096, 6096)]),
                BlockCoverage::Dead,
            ]
        );

        // Cleanse: a garbage source, block-local ranges — the output holds
        // the source's bits exactly on the ranges and zero elsewhere.
        let src: Vec<u8> = (0..2048u32).map(|i| (i * 37 + 11) as u8).collect();
        let base = 1024usize;
        let ranges = [(0usize, 1001usize), (4099usize, 6096usize), (8003, 8190)];
        let mut dst = [0xEEu8; 1024]; // pre-dirty: cleanse must fully own it
        cleanse_block(&src, base, &ranges, &mut dst);
        for bit in 0..8192usize {
            let useful = ranges.iter().any(|&(s, e)| bit >= s && bit < e);
            let got = (dst[bit / 8] >> (bit % 8)) & 1;
            let want = if useful {
                (src[base + bit / 8] >> (bit % 8)) & 1
            } else {
                0
            };
            assert_eq!(got, want, "bit {bit} (useful: {useful})");
        }
    }

    /// Run-list construction/accessor sanity: canonical forms, extents,
    /// single-run detection, and useful-interval merging.
    #[test]
    fn padding_spec_run_list_accessors() {
        // dense(m) is a single run tiling the domain.
        let dense = PaddingSpec::dense(5);
        assert_eq!(
            dense.as_single_run(),
            Some(PaddingRun {
                k_log: 5,
                useful_bits_per_block: 32,
                n_blocks: 1
            })
        );
        assert_eq!(dense.covered_bits(), 32);
        assert_eq!(dense.useful_intervals(), vec![(0, 32)]);

        // uniform: one interval per block; partial useful prefixes don't merge.
        let uni = PaddingSpec::uniform(4, 10, 3);
        assert_eq!(uni.covered_bits(), 48);
        assert_eq!(uni.useful_intervals(), vec![(0, 10), (16, 26), (32, 42)]);
        assert!(uni.as_single_run().is_some());

        // Fully-useful blocks merge into one interval, across run boundaries
        // too when the next run starts where the previous one's data ends.
        let multi = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 4,
                useful_bits_per_block: 16,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 3,
                useful_bits_per_block: 4,
                n_blocks: 1,
            },
        ]);
        assert!(multi.as_single_run().is_none());
        assert_eq!(multi.covered_bits(), 40);
        assert_eq!(multi.useful_intervals(), vec![(0, 36)]);

        // Zero-block runs are dropped (canonical form), so a list that
        // degenerates to one real run still takes the single-run fast path;
        // zero-useful runs cover address space but contribute no intervals.
        let canon = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 4,
                useful_bits_per_block: 16,
                n_blocks: 0,
            },
            PaddingRun {
                k_log: 3,
                useful_bits_per_block: 0,
                n_blocks: 2,
            },
        ]);
        assert_eq!(canon.runs().len(), 1);
        assert!(canon.as_single_run().is_some());
        assert_eq!(canon.covered_bits(), 16);
        assert_eq!(canon.useful_intervals(), Vec::<(usize, usize)>::new());
    }

    /// A run whose useful prefix exceeds its block size is malformed.
    #[test]
    #[should_panic(expected = "exceeds block size")]
    fn padding_spec_rejects_oversized_useful_prefix() {
        let _ = PaddingSpec::uniform(4, 17, 1);
    }

    /// Zero every bit outside the spec's useful intervals (honest padding).
    fn zero_outside_useful(spec: &PaddingSpec, bits: &mut [bool]) {
        let mut useful = vec![false; bits.len()];
        for (s, e) in spec.useful_intervals() {
            useful[s..e].fill(true);
        }
        for (b, u) in bits.iter_mut().zip(&useful) {
            if !*u {
                *b = false;
            }
        }
    }

    /// **Single-run spec is byte-identical to the dense prover** (same proof,
    /// same claim, same transcript position) on an honestly padded witness —
    /// the run-list generalization must not perturb today's wire format.
    /// Covers the BLAKE3 shape (k_log=14, useful=15409) over several blocks.
    #[test]
    fn prove_padded_single_run_matches_dense() {
        let (m, k_log, useful_bits) = (17usize, 14usize, 15_409usize);
        let padding = PaddingSpec::uniform(k_log, useful_bits, 1 << (m - k_log));

        let mut rng = Rng::new(0x5111_C1E4);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_dense);

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded) =
            prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        // Transcript position: the next challenge either prover's caller
        // would draw (lincheck's α slot) must agree.
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );
    }

    /// **Multi-run spec is byte-identical to the dense prover** through the
    /// general kernel paths (full-length b_med_counts table in round 1,
    /// per-pair skip table in round 2), including the `capture_s_hat_v_c`
    /// variant. The spec has two runs of different block shapes plus an
    /// implicit trailing gap — the shape of a multi-table slot schedule.
    #[test]
    fn prove_padded_multi_run_matches_dense() {
        let m = 15usize;
        // Two runs (2×2^13 + 1×2^12 = 20480 bits) + a 12288-bit trailing gap.
        let padding = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 5_000,
                n_blocks: 2,
            },
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 3_000,
                n_blocks: 1,
            },
        ]);
        assert!(padding.as_single_run().is_none(), "must exercise multi-run");
        assert!(padding.covered_bits() < 1 << m, "must exercise the gap");

        let mut rng = Rng::new(0x0417_1157);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense, s_hat_v_dense) = prove_packed_padded_capture_s_hat_v_c(
            &a_p,
            &b_p,
            &c_p,
            m,
            &PaddingSpec::dense(m),
            &mut ch_dense,
        );

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded, s_hat_v_padded) =
            prove_packed_padded_capture_s_hat_v_c(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        assert_eq!(s_hat_v_dense, s_hat_v_padded, "s_hat_v_c mismatch");
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );

        // And the multi-run proof still verifies.
        let mut ch_verify = FsChallenger::new(b"flock-test-v0");
        verify(m, &proof_padded, &mut ch_verify).expect("multi-run proof must verify");
    }

    /// **Sparse multi-run spec is byte-identical to the dense prover** through
    /// the M6 support-proportional tail (`fold_and_round_pair_sparse_into`):
    /// the support here is ~1% of the domain, so the tail's sparse rounds
    /// genuinely run (unlike `prove_padded_multi_run_matches_dense`, whose
    /// support is too dense to trigger them), including the mid-tail
    /// switch-back to the dense kernels (zeroing dead scratch) once the live
    /// fraction crosses the threshold.
    #[test]
    fn prove_padded_sparse_multi_run_matches_dense() {
        let m = 16usize;
        // Two count-derived-shaped runs: blocks of 2^13 bits with a 256-bit
        // declared prefix (n_t = 2 rows of 128), then a gap-shaped zero run,
        // then a smaller block shape — plus the implicit trailing gap.
        let padding = PaddingSpec::from_runs(vec![
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 256,
                n_blocks: 3,
            },
            PaddingRun {
                k_log: 13,
                useful_bits_per_block: 0,
                n_blocks: 1,
            },
            PaddingRun {
                k_log: 12,
                useful_bits_per_block: 128,
                n_blocks: 2,
            },
        ]);
        assert!(padding.as_single_run().is_none(), "must exercise multi-run");
        let live = padding.useful_block_intervals(K_SKIP);
        let live_elems: usize = live.iter().map(|&(s, e)| e - s).sum();
        assert!(
            live_elems * 16 <= 1usize << (m - K_SKIP),
            "spec must be sparse enough to drive the sparse tail"
        );

        let mut rng = Rng::new(0x0616_5A9D);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        zero_outside_useful(&padding, &mut a);
        zero_outside_useful(&padding, &mut b);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_dense = FsChallenger::new(b"flock-test-v0");
        let (proof_dense, claim_dense, s_hat_v_dense) = prove_packed_padded_capture_s_hat_v_c(
            &a_p,
            &b_p,
            &c_p,
            m,
            &PaddingSpec::dense(m),
            &mut ch_dense,
        );

        let mut ch_padded = FsChallenger::new(b"flock-test-v0");
        let (proof_padded, claim_padded, s_hat_v_padded) =
            prove_packed_padded_capture_s_hat_v_c(&a_p, &b_p, &c_p, m, &padding, &mut ch_padded);

        assert_eq!(proof_dense, proof_padded, "proof mismatch");
        assert_eq!(claim_dense, claim_padded, "claim mismatch");
        assert_eq!(s_hat_v_dense, s_hat_v_padded, "s_hat_v_c mismatch");
        assert_eq!(
            ch_dense.sample_f128(),
            ch_padded.sample_f128(),
            "post-proof transcript state diverged"
        );

        let mut ch_verify = FsChallenger::new(b"flock-test-v0");
        verify(m, &proof_padded, &mut ch_verify).expect("sparse multi-run proof must verify");
    }

    /// The support-proportional dispatch is byte-identical to the dense
    /// prover on FRAGMENTED slot schedules — the shapes that separate "tasks
    /// derived from the live intervals" from "tasks derived from the domain".
    /// Each case targets one way the interval-derived dispatch could differ
    /// from a whole-domain scan:
    ///
    /// - *fragmented*: many small intervals separated by dead gaps, so tasks
    ///   coalesce several pieces and their output spans cover gaps that must
    ///   stay unwritten.
    /// - *unaligned*: `useful_bits_per_block` is not a multiple of the pair
    ///   window (2^(k_skip+1)), so consecutive intervals round to pair
    ///   intervals that TOUCH — they must merge, or the shared boundary pair
    ///   is folded (and accumulated into the message) twice.
    /// - *wide*: live pairs exceed one task's budget in both round 2 and the
    ///   tail, exercising the multi-task output carve and the message
    ///   regrouping across tasks.
    #[test]
    fn prove_sparse_fragmented_multi_run_matches_dense() {
        let cases: [(&str, usize, PaddingSpec); 3] = [
            (
                "fragmented",
                16,
                PaddingSpec::from_runs(
                    (0..8)
                        .flat_map(|_| {
                            [
                                PaddingRun {
                                    k_log: 9,
                                    useful_bits_per_block: 128,
                                    n_blocks: 6,
                                },
                                PaddingRun {
                                    k_log: 9,
                                    useful_bits_per_block: 0,
                                    n_blocks: 2,
                                },
                            ]
                        })
                        .collect(),
                ),
            ),
            (
                "unaligned",
                16,
                PaddingSpec::from_runs(vec![
                    PaddingRun {
                        k_log: 10,
                        useful_bits_per_block: 200,
                        n_blocks: 20,
                    },
                    PaddingRun {
                        k_log: 8,
                        useful_bits_per_block: 129,
                        n_blocks: 40,
                    },
                ]),
            ),
            (
                "wide",
                24,
                PaddingSpec::from_runs(vec![
                    PaddingRun {
                        k_log: 12,
                        useful_bits_per_block: 3000,
                        n_blocks: 2048,
                    },
                    PaddingRun {
                        k_log: 12,
                        useful_bits_per_block: 0,
                        n_blocks: 512,
                    },
                    PaddingRun {
                        k_log: 11,
                        useful_bits_per_block: 1500,
                        n_blocks: 2048,
                    },
                ]),
            ),
        ];

        for (name, m, padding) in cases {
            assert!(
                padding.as_single_run().is_none(),
                "{name}: must exercise multi-run"
            );
            let live = padding.useful_block_intervals(K_SKIP);
            let live_elems: usize = live.iter().map(|&(s, e)| e - s).sum();
            assert!(
                live.len() > 4 && live_elems <= 1usize << (m - K_SKIP),
                "{name}: must be fragmented and drive the sparse path \
                 ({} intervals, {live_elems} live)",
                live.len(),
            );
            if name == "wide" {
                // Keep this case doing its job: it only covers the multi-task
                // carve while its live work exceeds one task's budget
                // (`LIVE_PAIRS_PER_TASK` in multilinear.rs, 2^16 pairs).
                let live_pairs: usize = padding
                    .useful_block_intervals(K_SKIP + 1)
                    .iter()
                    .map(|&(s, e)| e - s)
                    .sum();
                assert!(
                    live_pairs > 1 << 16,
                    "{name}: {live_pairs} live pairs no longer forces a \
                     multi-task round-2 dispatch"
                );
            }

            let mut rng = Rng::new(0x_5B10_2C4E ^ m as u64);
            let mut a = rng.bits(1 << m);
            let mut b = rng.bits(1 << m);
            zero_outside_useful(&padding, &mut a);
            zero_outside_useful(&padding, &mut b);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            let mut ch_dense = FsChallenger::new(b"flock-test-v0");
            let (proof_dense, claim_dense, s_hat_v_dense) = prove_packed_padded_capture_s_hat_v_c(
                &a_p,
                &b_p,
                &c_p,
                m,
                &PaddingSpec::dense(m),
                &mut ch_dense,
            );

            let mut ch_padded = FsChallenger::new(b"flock-test-v0");
            let (proof_padded, claim_padded, s_hat_v_padded) =
                prove_packed_padded_capture_s_hat_v_c(
                    &a_p,
                    &b_p,
                    &c_p,
                    m,
                    &padding,
                    &mut ch_padded,
                );

            assert_eq!(proof_dense, proof_padded, "{name}: proof mismatch");
            assert_eq!(claim_dense, claim_padded, "{name}: claim mismatch");
            assert_eq!(s_hat_v_dense, s_hat_v_padded, "{name}: s_hat_v_c mismatch");

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            verify(m, &proof_padded, &mut ch_verify)
                .unwrap_or_else(|e| panic!("{name}: proof must verify: {e:?}"));
        }
    }

    /// Determinism: same witness + same challenger seed → same proof.
    #[test]
    fn prove_deterministic() {
        let m = 14;
        let mut rng = Rng::new(99);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch1 = FsChallenger::new(b"flock-test-v0");
        let mut ch2 = FsChallenger::new(b"flock-test-v0");
        let (proof1, claim1) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch1);
        let (proof2, claim2) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch2);

        assert_eq!(proof1.round1_ab, proof2.round1_ab);
        assert_eq!(proof1.round1_c, proof2.round1_c);
        assert_eq!(proof1.multilinear_rounds, proof2.multilinear_rounds);
        assert_eq!(proof1.final_a_eval, proof2.final_a_eval);
        assert_eq!(proof1.final_b_eval, proof2.final_b_eval);
        assert_eq!(proof1.final_c_eval, proof2.final_c_eval);
        assert_eq!(claim1.z, claim2.z);
        assert_eq!(claim1.mlv_challenges, claim2.mlv_challenges);
    }
}
