//! Stratified query schedules: the binary decomposition of a level's query
//! count into power-of-two summands (see `docs/stratified-queries.tex`).
//!
//! Write `q = Σ_i 2^{c_i}` (distinct powers, descending). Summand `i` draws
//! one query uniformly inside each of the `2^{c_i}` depth-`c_i` subtrees of
//! the level's Merkle tree, so every query's stratum — and hence the tree
//! node its opening path terminates at — is a protocol constant. Soundness
//! is exactly the `(1−γ)^q` the unstratified sampler charges: the miss
//! probability factorizes over summands (independent draws), each summand's
//! factor is bounded by AM–GM over its equal-size strata, and the uniform
//! placement maximizes every factor simultaneously, so no slack is lost.
//!
//! **Authority.** A schedule is computed from `q` ONCE, at config-build time
//! ([`LevelSchedule::decompose`] via `with_default_stratified`), stored on
//! the prover/verifier configs, and consumed verbatim from there. It is
//! never derived at proof time, and in particular never from a proof's own
//! shape: the allocation is part of the statement, and a verifier that
//! adapted to the proof's geometry would accept a weaker, prover-chosen
//! allocation. Anything parsed from a proof is *checked* against the stored
//! schedule.
//!
//! The soundness-critical invariant a hand-edited schedule must keep (and
//! [`LevelSchedule::validate`] enforces): every summand covers the whole
//! block with equal-size strata and queries each stratum exactly once —
//! i.e. the schedule is exactly a list of stratum depths. What breaks
//! without it is quantified in the memory/docs: strata with zero queries
//! are catastrophic (the adversary hides all disagreement there), and any
//! per-summand density skew degrades the exponent below `q`.

/// One summand of a level's stratified schedule, in resolved form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Summand {
    /// Stratum depth `c`: the summand queries each of the `2^c` depth-`c`
    /// subtrees once.
    pub depth: usize,
    /// Query count `2^c`.
    pub count: usize,
    /// Squeeze bits per query: the low `d − c` index bits are sampled, the
    /// top `c` ARE the stratum. NOT the opening path length — paths
    /// truncate at the schedule's cap depth ([`LevelSchedule::cap_depth`]),
    /// which for shallower summands is deeper than their stratum.
    pub squeeze_bits: usize,
}

/// The stratified query schedule of one commit level: the depths of its
/// power-of-two summands, descending. Everything else (counts, per-query
/// bit widths, the cap depth) derives from these plus `log_block_len`.
///
/// Canonical form ([`Self::decompose`], enforced by [`Self::validate`]):
/// depths are non-increasing, strictly decreasing below `log_block_len`,
/// and repeats are allowed only *at* `log_block_len` (a full deterministic
/// sweep of the leaf layer; only reachable when `q > block_len`, which no
/// shipped ladder produces — [`derive_ladder_shape_tuned`] keeps `block_len ≥ q`
/// as a proof-size convention).
///
/// [`derive_ladder_shape_tuned`]: super::ligerito
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelSchedule {
    /// `d`: log₂ of the level's committed block length (codeword positions).
    pub log_block_len: usize,
    /// Summand stratum depths `c_1 ≥ c_2 ≥ …` (see canonical form above).
    pub summand_depths: Vec<usize>,
}

impl LevelSchedule {
    /// The canonical schedule for `queries` draws from a `2^log_block_len`
    /// block: greedily take the largest power-of-two summand that fits,
    /// clamped at the leaf layer. For `queries ≤ block_len` this is exactly
    /// the binary representation of `queries`; it puts the cap at
    /// `floor(lg q)` — the deepest any summand can sit — which is what
    /// every truncated path's length `d − c_1` and the absorbed cap's size
    /// trade against.
    pub fn decompose(queries: usize, log_block_len: usize) -> Self {
        let mut summand_depths = Vec::new();
        let mut rem = queries;
        while rem > 0 {
            let c = (usize::BITS as usize - 1 - rem.leading_zeros() as usize).min(log_block_len);
            summand_depths.push(c);
            rem -= 1usize << c;
        }
        Self {
            log_block_len,
            summand_depths,
        }
    }

    /// Total queries `Σ 2^{c_i}`.
    pub fn queries(&self) -> usize {
        self.summand_depths.iter().map(|&c| 1usize << c).sum()
    }

    /// Cap depth = the top summand's stratum depth: the absorbed commitment
    /// is the `2^{c_1}` nodes at depth `c_1`; deeper summands don't exist
    /// (depths only decrease), shallower summands' terminals derive from
    /// the cap via the shared upper tree. Zero queries degenerate to a
    /// classic root commitment.
    pub fn cap_depth(&self) -> usize {
        self.summand_depths.first().copied().unwrap_or(0)
    }

    /// The summands in schedule order, resolved to (depth, count, sibs).
    pub fn summands(&self) -> impl Iterator<Item = Summand> + '_ {
        let d = self.log_block_len;
        self.summand_depths.iter().map(move |&c| Summand {
            depth: c,
            count: 1usize << c,
            squeeze_bits: d - c,
        })
    }

    /// Per-query strata in the canonical sample order Phase 1's sampler
    /// draws in: summands in schedule order, strata `0..2^c` within each.
    /// Yields `(stratum_depth, stratum_index)` per query.
    pub fn query_strata(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.summand_depths
            .iter()
            .flat_map(|&c| (0..1usize << c).map(move |s| (c, s)))
    }

    /// Total opening cost `q·(d − c_1)` in path siblings: every path
    /// truncates at the cap layer, since all nodes above it are folds of
    /// the absorbed cap — a shallower summand's remaining `c_1 − c_j`
    /// levels are verifier-derivable, never proof bytes. NOT the squeeze
    /// accounting: a query still draws its own `d − c_j` low index bits
    /// ([`Summand::squeeze_bits`]).
    pub fn total_path_siblings(&self) -> usize {
        self.queries() * (self.log_block_len - self.cap_depth())
    }

    /// Check the soundness-critical canonical form against the query count
    /// the security schedule owes: non-increasing depths, within the tree,
    /// repeats only at the leaf layer, summing to exactly `expected_queries`.
    pub fn validate(&self, expected_queries: usize) -> Result<(), String> {
        for &c in &self.summand_depths {
            if c > self.log_block_len {
                return Err(format!(
                    "stratified: summand depth {c} exceeds block log {}",
                    self.log_block_len
                ));
            }
        }
        for w in self.summand_depths.windows(2) {
            if w[1] > w[0] {
                return Err(format!(
                    "stratified: summand depths not non-increasing ({} after {})",
                    w[1], w[0]
                ));
            }
            if w[1] == w[0] && w[0] != self.log_block_len {
                return Err(format!(
                    "stratified: repeated summand depth {} below the leaf layer — \
                     merge to depth {} (saves path siblings, same bound)",
                    w[0],
                    w[0] + 1
                ));
            }
        }
        let got = self.queries();
        if got != expected_queries {
            return Err(format!(
                "stratified: schedule sums to {got} queries, security schedule owes {expected_queries}"
            ));
        }
        Ok(())
    }
}

/// Per-level block logs from the config's declared ladder shape: level 0 is
/// `initial_log_msg_cols + log_inv_rates[0]`; level `ℓ ≥ 1` is
/// `recursive_log_msg_cols[ℓ−1] + log_inv_rates[ℓ]` (the block that
/// `queries[ℓ]` opens is the one level `ℓ` committed).
pub fn level_block_logs(
    initial_log_msg_cols: usize,
    recursive_log_msg_cols: &[usize],
    log_inv_rates: &[usize],
) -> Vec<usize> {
    assert_eq!(
        recursive_log_msg_cols.len() + 1,
        log_inv_rates.len(),
        "stratified: ladder shape mismatch (recursive levels vs rates)"
    );
    let mut out = Vec::with_capacity(log_inv_rates.len());
    out.push(initial_log_msg_cols + log_inv_rates[0]);
    for (l, &cols) in recursive_log_msg_cols.iter().enumerate() {
        out.push(cols + log_inv_rates[l + 1]);
    }
    out
}

/// The canonical per-level schedules for a config's query counts:
/// `decompose(queries[ℓ], block_log[ℓ])` for every level.
pub fn schedules(queries: &[usize], block_logs: &[usize]) -> Vec<LevelSchedule> {
    assert_eq!(
        queries.len(),
        block_logs.len(),
        "stratified: query levels vs block levels mismatch"
    );
    queries
        .iter()
        .zip(block_logs)
        .map(|(&q, &d)| LevelSchedule::decompose(q, d))
        .collect()
}

/// Validate stored per-level schedules against the query counts and ladder
/// shape they must serve. This is the load-time check: schedules may be
/// customized (e.g. a split top summand), but never structurally unsound
/// and never out of step with the security schedule's counts.
pub fn validate_schedules(
    stored: &[LevelSchedule],
    queries: &[usize],
    block_logs: &[usize],
) -> Result<(), String> {
    if stored.len() != queries.len() {
        return Err(format!(
            "stratified: {} schedules for {} query levels",
            stored.len(),
            queries.len()
        ));
    }
    for (l, (sched, (&q, &d))) in stored
        .iter()
        .zip(queries.iter().zip(block_logs))
        .enumerate()
    {
        if sched.log_block_len != d {
            return Err(format!(
                "stratified: level {l} schedule block log {} != ladder block log {d}",
                sched.log_block_len
            ));
        }
        sched
            .validate(q)
            .map_err(|e| format!("stratified: level {l}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        merkle::cap_depth,
        pcs::{
            ligerito::{default_config, default_verifier_config},
            stratified::{LevelSchedule, level_block_logs, schedules, validate_schedules},
        },
    };

    #[test]
    fn decompose_is_binary_representation() {
        // Slim L0 shape: q = 90 = 64 + 16 + 8 + 2, d = 19.
        let s = LevelSchedule::decompose(90, 19);
        assert_eq!(s.summand_depths, vec![6, 4, 3, 1]);
        assert_eq!(s.queries(), 90);
        assert_eq!(s.cap_depth(), 6);
        // Paths truncate at the cap: 90·(19 − 6) = 1,170 — not the
        // per-summand 64·13 + 16·15 + 8·16 + 2·18 = 1,236, whose extra 66
        // siblings the absorbed cap already determines.
        assert_eq!(s.total_path_siblings(), 1170);
        assert!(s.validate(90).is_ok());
    }

    #[test]
    fn decompose_powers_and_edges() {
        assert_eq!(LevelSchedule::decompose(128, 19).summand_depths, vec![7]);
        assert_eq!(LevelSchedule::decompose(96, 19).summand_depths, vec![6, 5]);
        assert_eq!(LevelSchedule::decompose(1, 19).summand_depths, vec![0]);
        let zero = LevelSchedule::decompose(0, 19);
        assert!(zero.summand_depths.is_empty());
        assert_eq!(zero.cap_depth(), 0);
        assert!(zero.validate(0).is_ok());
    }

    #[test]
    fn decompose_clamps_at_leaf_layer() {
        // q > block_len: full deterministic sweeps at depth d, then the
        // binary representation of the remainder. Unreachable from shipped
        // ladders (block_len ≥ q convention) but the math stays total: the
        // per-summand AM–GM bound applies to a depth-d summand verbatim.
        let s = LevelSchedule::decompose(300, 8);
        assert_eq!(s.summand_depths, vec![8, 5, 3, 2]);
        assert_eq!(s.queries(), 300);
        assert!(s.validate(300).is_ok());

        let s = LevelSchedule::decompose(600, 8);
        assert_eq!(s.summand_depths, vec![8, 8, 6, 4, 3]);
        assert_eq!(s.queries(), 600);
        assert!(s.validate(600).is_ok());
    }

    #[test]
    fn query_strata_order_and_widths() {
        let s = LevelSchedule::decompose(11, 5); // 8 + 2 + 1
        let strata: Vec<(usize, usize)> = s.query_strata().collect();
        assert_eq!(strata.len(), 11);
        assert_eq!(&strata[..3], &[(3, 0), (3, 1), (3, 2)]);
        assert_eq!(&strata[8..], &[(1, 0), (1, 1), (0, 0)]);
        let bits: Vec<usize> = s.summands().map(|m| m.squeeze_bits).collect();
        assert_eq!(bits, vec![2, 4, 5]);
    }

    #[test]
    fn validate_rejects_unsound_schedules() {
        // Zero-query strata / wrong totals: the schedule owes exactly q.
        let s = LevelSchedule::decompose(90, 19);
        assert!(s.validate(96).is_err());

        // Increasing depths.
        let bad = LevelSchedule {
            log_block_len: 19,
            summand_depths: vec![4, 6],
        };
        assert!(bad.validate(80).is_err());

        // Repeat below the leaf layer: mergeable, refuse the non-canonical
        // (and path-row-wasteful) form.
        let bad = LevelSchedule {
            log_block_len: 19,
            summand_depths: vec![5, 5],
        };
        assert!(bad.validate(64).is_err());

        // Depth beyond the tree.
        let bad = LevelSchedule {
            log_block_len: 4,
            summand_depths: vec![5],
        };
        assert!(bad.validate(32).is_err());
    }

    #[test]
    fn config_constructors_attach_valid_schedules() {
        // Every ProverConfig/VerifierConfig construction site ends in
        // .with_default_stratified() — pin that the canonical constructor
        // yields stored schedules that pass the load-time authority check.
        let p = default_config(20, 4, 2).unwrap();
        assert_eq!(p.stratified.len(), p.queries.len());
        p.validate_stratified().unwrap();
        for (sched, &q) in p.stratified.iter().zip(&p.queries) {
            assert_eq!(sched.queries(), q);
            // Stratified cap = floor(lg q): never deeper than the old
            // ceil(lg q) cap, shallower exactly when q isn't a power of two.
            assert!(sched.cap_depth() <= cap_depth(q, sched.log_block_len));
        }
        let v = default_verifier_config(20, 4, 2).unwrap();
        v.validate_stratified().unwrap();
        assert_eq!(
            p.stratified, v.stratified,
            "prover and verifier must agree on the allocation"
        );
    }

    #[test]
    fn schedules_follow_the_ladder() {
        let block_logs = level_block_logs(14, &[9, 6], &[2, 3, 4]);
        assert_eq!(block_logs, vec![16, 12, 10]);
        let queries = vec![148, 121, 110];
        let scheds = schedules(&queries, &block_logs);
        assert_eq!(scheds.len(), 3);
        for (sched, &q) in scheds.iter().zip(&queries) {
            assert!(sched.validate(q).is_ok());
        }
        assert!(validate_schedules(&scheds, &queries, &block_logs).is_ok());
        // A tampered count is caught.
        assert!(validate_schedules(&scheds, &[148, 121, 111], &block_logs).is_err());
        // A tampered block log is caught.
        assert!(validate_schedules(&scheds, &queries, &[16, 12, 9]).is_err());
    }
}
