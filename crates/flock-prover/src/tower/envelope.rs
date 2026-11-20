use std::{
    env::var,
    sync::atomic::{AtomicUsize, Ordering},
};

use flock_core::{
    aggregate::Accumulator, circuit::builder::SlotId, matrix_fold::MatrixClaim,
    pcs::ligerito::embedded_initial_k_or_default,
};

use crate::tower::{
    AssistLayerGate, BitSpreadGate, BitSpreadTable, Blake3Gate, CollapsedSlots, F128,
    FamilyTransposeTileGate, GateType, HashKind, HornerStepGate, LeafEvalGate, LigeritoProfile,
    MacGate, MergedRoundGate, PcsParams, PowMaskGate, PrefixGate, SLOT_WORDS, ShapeBuilder,
    SpineGate, SquareChainGate, SwapGate, TransposeDotGate, UnionInstance, Wire, ZcRoundGate,
    affine_walk::AffineEnvLayouts, config::ENVELOPE_FLOOR_M, cw,
};

/// Family H's Frobenius squaring chains ([`SquareChainGate`]).
pub(super) const KEY_SQUARE_CHAIN: usize = 890;
/// Family H's Horner steps ([`HornerStepGate`]).
pub(super) const KEY_HORNER_STEP: usize = 891;
/// Family H's bit-transpose dot, one row per transposed word
/// ([`TransposeDotGate`]).
pub(super) const KEY_TRANSPOSE_DOT: usize = 892;

/// Wall 2's registry-geometry constants at the settled envelope (slim,
/// m* = 29): the UNION of the leaf-outer's and the node's type sets, at the
/// envelope maxima. Measured at the m29 fixed point (envelope_registry_diff
/// + the tower census, 2026-08-06):
///
/// - `spread_w` 21 covers the m33 FAST chain leaf's L0 depth; the m29 Slim
///   outer ladder needs 19 and leaves the high outputs unread.
/// - The affine inner verifier rides five shape-free gates (keys 900..=904,
///   [`AffineEnvLayouts`]), independent of lanes and residual length.
/// - `nu` 14: each of the two independent child verifiers has its own
///   identical BLAKE slot, and every physical slot stays below 2^14 rows.
/// - The table types' IO words plus one public slot take more than 256 cell
///   slots once family H's square chain, Horner step and transpose dot
///   join the affine inner's gates, so the cell address needs 9 bits and
///   `mu = nu + 9 = 23` tower-wide.
///
/// A ladder that drifts off these constants surfaces as a NEW slot at
/// emission time and hence a registry-digest mismatch — the failure is
/// loud, never silent.
pub(super) struct EnvShape {
    pub(super) nu: usize,
    pub(super) spread_w: usize,
    pub(super) pf_w: usize,
    /// Historical counts* oracle values. Shipped envelope proofs use
    /// unconditional free counts, so these values no longer pad rows or
    /// determine the circuit digest; `counts_el` remains the canonical
    /// element-slot key list and retains the old cap census for
    /// comparison.
    pub(super) counts_el: [(usize, usize); 11],
    /// publics* — the ONE public-segment length every envelope outer pads
    /// to (published zeros appended after all real publics). The child's
    /// publics count is what a PARENT's walk consumes — H(publics) chain
    /// rows and the recombination's 8-lane folds both scale with it — so
    /// one count is what makes the L1 walk (leaf children) and the L2
    /// walk (node children) row-identical. The last [`ENV_APP_WORDS`] of
    /// them are the APPLICATION BLOCK (see [`env_app_base`]).
    publics: usize,
    /// lanes* — the pinned committed lane count (see [`outer_lanes`]): the
    /// one aggregate of a child's layout that stays circuit structure
    /// under FREE COUNTS.
    pub(super) lanes: usize,
    /// The ACC_CHAIN block's width: the leaf registry's lane. A tower fixes
    /// it once for every level ([`ENV_ACC_CHAIN_WORDS`] for the chain
    /// tower; [`lane_block_words`] of its lane for the others).
    pub(super) acc_chain_words: usize,
    /// The affine inner's slot layouts ([`AffineEnvLayouts::of`]): the
    /// union of the leaf's and the envelope outer's, so every outer of one
    /// tower declares the same types; set by the builders through
    /// [`EnvShape::with_affine`].
    pub(super) affine: AffineEnvLayouts,
}

impl EnvShape {
    /// This envelope with the affine layouts of a tower whose leaf commits
    /// under `leaf` and whose outers prove under `outer_profile`.
    pub(super) fn with_affine(mut self, leaf: &PcsParams, outer_profile: LigeritoProfile) -> Self {
        self.affine = AffineEnvLayouts::of(&[leaf, &self.outer_params(outer_profile)]);
        self
    }

    /// The envelope outer's commitment parameters: the m* floor at the
    /// profile's own interleave, `lanes*` lanes.
    pub(super) fn outer_params(&self, profile: LigeritoProfile) -> PcsParams {
        PcsParams {
            m: ENVELOPE_FLOOR_M,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: embedded_initial_k_or_default(ENVELOPE_FLOOR_M, profile),
            profile,
            num_lanes: Some(self.lanes),
            merkle_hash: HashKind::Blake3,
        }
    }
}

/// The APPLICATION STATEMENT's width in the envelope's public segment:
/// the hash-chain span `(h_start, h_end)` — four packed words each —
/// plus the SPAN COUNTER `g^{n_blocks}` (word 8; Ron's call, 2026-09-07:
/// the number of hashes is part of the statement) and the COUNTER BASE
/// `C = g^{blocks_per_leaf}` (word 9). Ten 128-bit words.
///
/// The base word is the counter's GROUNDING: a fixed public is only
/// natively checked at the ROOT (`check_public` — below the root the
/// walks treat child publics as witness wires), so every parent
/// copy-constrains each child's word 9 to its own baked C, the chain
/// ends at the root's native check, and every word 8 is then bound by
/// PROVEN relations — `C^k` via mac rows at the FL, the children's
/// product via a mac row at every node.
pub(super) const ENV_APP_WORDS: usize = 10;

/// The span counter's base: the polynomial `x` in F128, certified a
/// GENERATOR of the multiplicative group by `span_generator_full_order`
/// — so two spans collide only at a difference of `2^128 - 1`, which no
/// tower reaches. The count composes MULTIPLICATIVELY child-to-parent
/// (char-2 kills integer field addition): an FL raises the GROUNDED
/// base word to its arity via mac rows, and every node multiplies its
/// children's counter words with one MAC row — see [`ENV_APP_WORDS`] for
/// the grounding chain.
pub(super) const SPAN_G: F128 = F128 { lo: 2, hi: 0 };

/// `SPAN_G^e` by square-and-multiply — the native side of the span
/// counter: the statement leg holds the published word against
/// `span_count_word(statement.n_blocks)`.
pub(super) fn span_count_word(e: u128) -> F128 {
    let mut acc = F128::ONE;
    let mut base = SPAN_G;
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        base = base * base;
        e >>= 1;
    }
    acc
}

/// THE GENERATOR CERTIFICATE: `SPAN_G` has full order `2^128 - 1` — for
/// every prime factor `p`, `g^{(2^128 - 1)/p} != 1`. The factorization is
/// `(2^64 - 1)(2^64 + 1)` = 3 · 5 · 17 · 257 · 641 · 65537 · 6700417 ·
/// 274177 · 67280421310721. Plus the homomorphism the composition rests
/// on: `g^{a+b} = g^a · g^b`.
#[test]
fn span_generator_full_order() {
    const GROUP_ORDER: u128 = u128::MAX; // 2^128 - 1
    const PRIMES: [u128; 9] = [3, 5, 17, 257, 641, 65537, 6700417, 274177, 67280421310721];
    assert_eq!(
        PRIMES.iter().product::<u128>(),
        GROUP_ORDER,
        "factorization"
    );
    for p in PRIMES {
        assert_ne!(
            span_count_word(GROUP_ORDER / p),
            F128::ONE,
            "SPAN_G's order is divisible by {p}"
        );
    }
    assert_eq!(span_count_word(GROUP_ORDER), F128::ONE, "Lagrange sanity");
    assert_eq!(
        span_count_word(1 << 20) * span_count_word((1 << 20) + 12345),
        span_count_word((1 << 21) + 12345),
        "the counter composes multiplicatively"
    );
}

/// Steady-repetition override: how many EXTRA times a builder re-runs its
/// ONLINE phases (tapes + walk + witgen + prove + verify) over the
/// once-built shape, collecting one [`Online`] record per iteration. The
/// bench sets this per stage so a 5-run median costs ONE ~3-5 s setup
/// instead of five — the per-shape setup was ~96% of the bench's wall
/// clock. `usize::MAX` = unset (the `TOWER_STEADY` env knob applies).
pub(super) static STEADY_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

pub(super) fn steady_reps() -> usize {
    let ov = STEADY_OVERRIDE.load(Ordering::Relaxed);
    if ov != usize::MAX {
        return ov;
    }
    var("TOWER_STEADY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The INHERITABLE ACCUMULATOR blocks. An outer publishes the accumulator
/// claims its parent will fold as PRIORS, and the parent connects to them
/// WIRE-TO-WIRE (`child_pub_w[base + ..]`) — so, exactly like the app
/// block, they have to sit where live public usage cannot move them.
/// Otherwise a first-level child and an internal child expose their claims
/// at different indices and no single parent circuit can read both.
///
/// TWO blocks, keyed by REGISTRY ROLE rather than by which fold produced
/// them — that is the distinction a parent actually cares about: MAIN
/// carries the ENVELOPE-registry claims (an internal node's own 2→1 fold),
/// CHAIN the LOWER-registry ones (a first-level node's chain fold; an
/// internal node's chain LANE). An outer with no claims of a role fills
/// that block with zeros. Each block is `[claims | zero padding]`, so a
/// shorter shape — a dev-size chain, a fold with fewer groups — rides the
/// same layout and a reader simply stops at its own group widths.
pub(super) const ENV_ACC_CHAIN_WORDS: usize = 160;

/// The public segment's BODY: everything before the tail blocks. Fixed for
/// every envelope, so a tower whose lane needs a wider ACC_CHAIN block
/// grows `publics*` by exactly the difference. The largest live body is the
/// leanXMSS first-level node over two 992-signature leaves at 2^15-row
/// slots: 3301 words (2026-11-10, affine inner). The m33 chain FL uses 1977
/// and the internal nodes of both towers about 2520. 3382 leaves 81 words
/// of margin and makes the chain tower's `publics*` 4544 words, 71 KiB, a
/// whole number of the parent's 1 KiB H(publics) chunks.
const ENV_BODY_WORDS: usize = 3382;

/// The ACC_CHAIN width a lane accumulator needs: per folded group its
/// published `[live | rho_col | rho_row | value]`, boolean A and B per group,
/// the linear-identity groups, then sigma, then jagged.
pub(super) fn lane_block_words(acc: &Accumulator) -> usize {
    let w = |c: &MatrixClaim| 2 + c.col.point.len() + c.row.point.len();
    acc.per_type.iter().map(|(a, b)| w(a) + w(b)).sum::<usize>()
        + acc.per_linear.iter().map(w).sum::<usize>()
        + acc.sigma.iter().map(|(_, c)| w(c)).sum::<usize>()
        + acc.jagged.iter().map(|(_, c)| w(c)).sum::<usize>()
}
// A live internal node publishes 852 words in both towers (m33 chain and
// leanXMSS, 2026-11-10): family H's three element types (square chain,
// Horner step, transpose dot) each add their A and B claims. 44 words of
// margin. The block is fixed so that the same layout carries the
// accumulator at every level.
pub(super) const ENV_ACC_MAIN_WORDS: usize = 896;

/// THE PASSENGER (wall 3): one sigma-shaped and one jagged-shaped entry,
/// same layout as the ACC_MAIN keyed slots. A spine node's node-slot
/// inherits an entry keyed by its child's OWN child — which matches at
/// every steady level but once, at the first steady node over a base
/// node. That single ORPHAN cannot fold (its key names a circuit no slot
/// of this node names), so it rides here, re-published child to parent by
/// a gated copy, until the root discharges it against the base circuit's
/// own tables. Zeros when empty, which is every node but that one and its
/// ancestors.
pub(super) const ENV_PASS_WORDS: usize = 96;

/// FREE COUNTS ARE UNCONDITIONAL (the count win shipped, 2026-08-09): under
/// the envelope, children declare their own per-type row counts — the
/// heights reach a parent only as folded claims on the jagged layout,
/// discharged at the root — and only the LANE COUNT stays pinned
/// (`EnvShape::lanes`). The former count-padding switches are retired;
/// `counts_el` remains only as the key list + historical census.
pub(super) fn outer_lanes(union: &UnionInstance, log_batch_size: usize) -> Option<usize> {
    let content = union.commit_lanes(log_batch_size);
    let env = envelope_shape();
    let c = content.unwrap_or(1usize << log_batch_size);
    assert!(
        c <= env.lanes,
        "content lanes {c} exceed the lane pin {}",
        env.lanes
    );
    Some(env.lanes)
}

pub(super) fn env_app_base(env: &EnvShape) -> usize {
    env.publics - ENV_APP_WORDS
}

pub(super) fn env_pass_base(env: &EnvShape) -> usize {
    env_app_base(env) - ENV_PASS_WORDS
}

pub(super) fn env_acc_main_base(env: &EnvShape) -> usize {
    env_pass_base(env) - ENV_ACC_MAIN_WORDS
}

pub(super) fn env_acc_chain_base(env: &EnvShape) -> usize {
    env_acc_main_base(env) - env.acc_chain_words
}

/// The reserved tail blocks an envelope outer hands to
/// [`pad_envelope_counts`] — published after the padding, each zero-filled
/// to its fixed width. Everything empty is the leaf/node outer's case.
#[derive(Default)]
pub(super) struct EnvTail<'w> {
    /// Envelope-registry accumulator claims: this outer's own 2→1 fold.
    pub(super) acc_main: &'w [Wire],
    /// Lower-registry accumulator claims: the FL's chain fold, or an
    /// internal node's chain LANE.
    pub(super) acc_chain: &'w [Wire],
    /// The PASSENGER: entries this node could not fold and did not drop.
    pub(super) pass: &'w [Wire],
    /// The application statement.
    pub(super) app: &'w [Wire],
}

/// The fixed envelope shape — always on, no override: the registry
/// convergence below is pinned to m* = 29's measured geometry.
pub(super) fn envelope_shape() -> EnvShape {
    envelope_shape_with_lane(ENV_ACC_CHAIN_WORDS)
}

/// [`envelope_shape`] with an ACC_CHAIN block of `acc_chain_words`: the
/// same registry and body, the public segment longer by the difference.
pub(super) fn envelope_shape_with_lane(acc_chain_words: usize) -> EnvShape {
    EnvShape {
        // The two-child envelope fits at 14 with each independent child
        // verifier on its own identical BLAKE slot. Every physical slot
        // remains below 2^14 rows, and 512 cell slots give mu = 14 + 9 =
        // 23 (measured 2026-11-10).
        nu: 14,
        // 21 = the m33 FAST chain leaf's L0 depth (log_msg_cols 20 +
        // log_inv_rate 1), which the first-level node walks. Smaller
        // leaves (m32 needs 20) and the m29 slim outer ladder (19) leave
        // the top outputs unread.
        spread_w: 21,
        // Six variants: pl = Σ_{levels above} fold count, so the deepest is
        // the m32 FAST chain ladder's level-0 (six levels, 5×3 folds above
        // it) — the m29 slim outer ladder's five stop at 12 and ride the
        // rest at count 0.
        pf_w: 8,
        // Iterated at the padded envelope 2026-08-06 (probe + tower
        // census, elementwise max of leaf/node usage). Only b3, le8, pf8
        // and mac are content-geometry-sensitive; everything else hits its
        // cap exactly (registry-shaped).
        // The 4th entry is the fused PoW-mask slot (one row per grinding
        // site). It is a historical oracle cap only: free counts are
        // unconditional, and the strict-Slim m29 spine exercises the live
        // count and fixed envelope layout end to end.
        // BLAKE is the only boolean family whose live count exceeds 2^14.
        // The two independent child regions use identical slots while the
        // shipped free-count path records each actual prefix.
        counts_el: [
            (600, 49000), // mac — the nu* driver; watch the 2^15 ceiling
            (602, 8000),  // fold/recombination MACs, split from verifier arithmetic
            (500, 1000),  // zcr
            // mrs — 1000, was 900: wall 3's steady spine node runs the
            // extra keyed slot's rounds (measured 949 live).
            (400, 1000),
            (0, 9000),    // spine
            (601, 300),   // assist
            (8, 4200),    // leaf-eval 8-lane
            (318, 15000), // prefix w 8
            // Census only, like every count here (free counts never pad to
            // it). Square chains: family H's 16 Horner chains plus ĝ's
            // forward chains, which scale with the child's geometry: 410
            // live at the production first-level node (2 × 992 signatures),
            // 340 at the 4-signature test leaf's (2026-11-05; 160 predated
            // ĝ's chains). Horner: 256 steps per child region.
            (KEY_SQUARE_CHAIN, 410),
            (KEY_HORNER_STEP, 512),
            // The transpose dot: 128 rows per claim, two claims per child.
            (KEY_TRANSPOSE_DOT, 512),
        ],
        // The body plus the four tail blocks (see [`ENV_BODY_WORDS`]).
        publics: ENV_BODY_WORDS
            + acc_chain_words
            + ENV_ACC_MAIN_WORDS
            + ENV_PASS_WORDS
            + ENV_APP_WORDS,
        // The committed lane count — the ONE piece of a child's layout that
        // stays circuit structure (the parent hashes `num_lanes`-word
        // leaves), so it is pinned while everything count-shaped rides the
        // jagged claims. The largest member is the leanXMSS first-level
        // node over two 992-signature leaves: 24.15 lanes of 2^17 words
        // (2026-11-10, affine inner; 30.32 under the F256 inner). The m33
        // chain FL is 21.86 and the internal nodes of both towers about 18.
        // 25 to 28 words all hash as 7 BLAKE3 blocks per opened row; 26
        // keeps a lane of margin. It stays below `2^initial_k = 32`, so
        // children remain lane-major, and a bigger content fails loudly in
        // `outer_lanes`.
        lanes: 26,
        acc_chain_words,
        affine: AffineEnvLayouts::default(),
    }
}

/// Find-or-create a slot under this file's keyed-cache scheme
/// (0 spine / 8 leaf-eval / 400 mrs / 500 zcr / 600 mac / 601 assist /
/// 602 fold-mac / 890 square chain / 891 Horner step / 892 transpose dot /
/// 310+w prefix / 900..=904 the affine inner). Every element-slot
/// declaration on the
/// recursion path routes through this, so the envelope can pre-seed the
/// cache (fixing the declaration order registry-wide) while the
/// off-envelope path creates on first use, in the historical order,
/// byte-identically.
pub(super) fn slot_cached<G>(
    sb: &mut ShapeBuilder,
    cache: &mut Vec<(usize, SlotId)>,
    key: usize,
    mk: impl FnOnce() -> G,
) -> SlotId
where
    G: GateType + Send + Sync + 'static,
    G::Row: Send + 'static,
    G::Hint: 'static,
{
    match cache.iter().find(|&&(k, _)| k == key) {
        Some(&(_, s)) => s,
        None => {
            let s = sb.slot(mk());
            cache.push((key, s));
            s
        }
    }
}

/// Declare the envelope's 32 table types in one canonical order.
/// `Registry::new` sorts class-major then k_log-descending with a
/// STABLE sort, so the declaration order here fixes every same-k_log
/// tie-break — the leaf-outer and node registries become the same sorted
/// type list, which together with nu* is registry-digest equality. Returns
/// the six boolean slots; every element type pre-seeds `cache` under the keyed
/// scheme so both builders' demand sites hit the cache instead of
/// declaring. The order is the node's historical one.
pub(super) fn declare_envelope_slots(
    sb: &mut ShapeBuilder,
    nu: usize,
    cache: &mut Vec<(usize, SlotId)>,
    env: &EnvShape,
) -> CollapsedSlots {
    debug_assert_eq!(nu, env.nu, "the envelope declares at nu*");
    let q = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        b3_alt: Some(sb.slot(Blake3Gate { nu })),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(env.spread_w),
            nu,
        }),
        pow: sb.slot(PowMaskGate { nu }),
        family: Some(sb.slot(FamilyTransposeTileGate { nu })),
    };
    slot_cached(sb, cache, 600, MacGate::new);
    slot_cached(sb, cache, 602, MacGate::new);
    slot_cached(sb, cache, 500, ZcRoundGate::new);
    slot_cached(sb, cache, 400, MergedRoundGate::new);
    slot_cached(sb, cache, 0, SpineGate::new);
    slot_cached(sb, cache, 601, AssistLayerGate::new);
    slot_cached(sb, cache, 8, || LeafEvalGate::new(8));
    slot_cached(sb, cache, 310 + env.pf_w, || PrefixGate::new(env.pf_w));
    // The affine inner emits into the envelope's eq-prefix slot at
    // `310 + affine_gates::PREFIX` (`AffineSlots::from_cache`), so the two
    // widths must agree.
    assert!(
        env.affine.is_empty() || env.pf_w == crate::tower::affine_gates::PREFIX,
        "envelope pf_w {} != the affine eq-prefix width {}",
        env.pf_w,
        crate::tower::affine_gates::PREFIX
    );
    env.affine.declare(sb, cache);
    slot_cached(sb, cache, KEY_SQUARE_CHAIN, SquareChainGate::new);
    slot_cached(sb, cache, KEY_HORNER_STEP, HornerStepGate::new);
    slot_cached(sb, cache, KEY_TRANSPOSE_DOT, TransposeDotGate::new);
    q
}

/// Every envelope slot's count rounds up to a multiple of `2^8` rows
/// ([`pad_envelope_counts`]): the alignment the opening's multipoint assist
/// needs for its closed form, at a few percent of area (the first-level
/// node over two 992-signature leaves: 30.47 → 30.96 lanes, measured before
/// family H by Horner).
pub(super) const ENV_ROW_ALIGN_LOG: usize = 8;

/// Pad every envelope slot's declared count up to counts* (the counts pin:
/// one declared-count vector for every envelope outer, so the union content
/// a parent walks is level-independent). Call once per builder, AFTER all
/// emission, immediately before `finish()`.
///
/// A padding row is a REAL GATE with all-`zw` inputs (and a zero hint for
/// the hinted swap slot), so every mechanism sees an ordinary row by
/// construction: the boolean witness generators set the const bit the
/// lincheck's count binding demands (all-zero rows fail exactly there —
/// found the hard way), the element rows come out all-zero (the builder
/// tables are homogeneous), and the wiring covers the cells with genuine
/// gather-claimed gates. The outputs are deliberately unconsumed.
pub(super) fn pad_envelope_counts(
    sb: &mut ShapeBuilder,
    q: &CollapsedSlots,
    cache: &[(usize, SlotId)],
    env: &EnvShape,
    zw: Wire,
    hints: &mut Vec<[u32; SLOT_WORDS]>,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    tail: &EnvTail,
) {
    // FREE COUNTS ARE THE DEFAULT (the count win): the ROW padding is
    // skipped — children declare their own counts, min-one-row keeps every
    // type live, and the heights reach a parent only as jagged claims.
    // The tail blocks and the public segment still pad, so the layout a
    // parent reads is unchanged. The historical caps remain in `counts_el`
    // as the slot-declaration key list, but row-count padding is retired.
    let mut report: Vec<String> = Vec::new();
    let mut over: Vec<String> = Vec::new();
    let mut pad = |sb: &mut ShapeBuilder,
                   hints: &mut Vec<[u32; SLOT_WORDS]>,
                   over: &mut Vec<String>,
                   name: &str,
                   s: SlotId,
                   target: usize,
                   hinted: bool,
                   fixed_inputs: Option<&[Wire]>| {
        let live = sb.rows_in_slot(s);
        report.push(format!("{name} {live}/{target}"));
        if live > target {
            over.push(format!("{name} {live} > {target}"));
            return;
        }
        let ins = fixed_inputs
            .map(<[Wire]>::to_vec)
            .unwrap_or_else(|| vec![zw; sb.slot_inputs(s)]);
        assert_eq!(
            ins.len(),
            sb.slot_inputs(s),
            "padding input arity for {name}"
        );
        for _ in live..target {
            if hinted {
                hints.push([0u32; SLOT_WORDS]);
                sb.gate_hinted(s, &ins);
            } else {
                sb.gate(s, &ins);
            }
        }
    };
    // In free-count mode a live slot's target IS its live count, while an
    // empty declared slot pads only to ONE ROW — never to the old cap.
    // That is the whole pin the run structure needs: `assist_boundaries`
    // merges columns only when they are EMPTY, so every non-empty column is
    // a singleton run and the run count is registry-derived EXCEPT through
    // the predicate `n_t > 0`. Keep every type non-empty and the counts
    // become pure values.
    // COUNT ALIGNMENT: every slot rounds up to a multiple of
    // `2^ENV_ROW_ALIGN_LOG` rows (real padding gates, as above), so every
    // committed column height is too and the dense stack is aligned at that
    // granularity — the multipoint assist's closed form over sub-columns
    // instead of a 2^m weight. No dummy row is committed: the padding rows
    // are live, declared gates.
    let align = 1usize << ENV_ROW_ALIGN_LOG;
    let floor1 = |sb: &ShapeBuilder, s| {
        sb.rows_in_slot(s)
            .max(1)
            .next_multiple_of(align)
            .min(1usize << env.nu)
    };
    let t_b3 = floor1(sb, q.b3);
    let b3_alt = q.b3_alt.expect("the envelope declares two BLAKE slots");
    let t_b3_alt = floor1(sb, b3_alt);
    let t_swap = floor1(sb, q.swap);
    let t_spread = floor1(sb, q.spread);
    let t_pow = floor1(sb, q.pow);
    pad(sb, hints, &mut over, "b3", q.b3, t_b3, false, None);
    pad(sb, hints, &mut over, "b3b", b3_alt, t_b3_alt, false, None);
    pad(sb, hints, &mut over, "swap", q.swap, t_swap, true, None);
    pad(
        sb, hints, &mut over, "spread", q.spread, t_spread, false, None,
    );
    let pow_check = cw(sb, vals, consts, F128::new(0, 1u64 << 63));
    let pow_inputs = [zw, zw, zw, pow_check];
    pad(
        sb,
        hints,
        &mut over,
        "pow",
        q.pow,
        t_pow,
        false,
        Some(&pow_inputs),
    );
    let family = q.family.expect("the envelope declares family H");
    let t_family = floor1(sb, family);
    pad(
        sb, hints, &mut over, "family", family, t_family, false, None,
    );
    let affine_keys = env.affine.keys();
    let keys = env
        .counts_el
        .iter()
        .map(|&(k, c)| (k, c))
        .chain(affine_keys.iter().map(|&k| (k, 0)));
    for (key, count) in keys {
        let &(_, s) = cache
            .iter()
            .find(|&&(k, _)| k == key)
            .unwrap_or_else(|| panic!("envelope slot key {key} missing from the cache"));
        let _ = count;
        let target = floor1(sb, s);
        pad(
            sb,
            hints,
            &mut over,
            &format!("el{key}"),
            s,
            target,
            false,
            None,
        );
    }
    // A slot the emission demanded but the envelope never declared: the
    // keyed cache created it on the fly, so this builder's registry carries
    // a type the other envelope outers do not — the digest diverges and
    // nothing else here would say so. Name it (the key IS the parameter:
    // 100 + pl for a residual variant, 310 + w for a prefix width).
    let stray: Vec<usize> = cache
        .iter()
        .map(|&(k, _)| k)
        .filter(|k| !env.counts_el.iter().any(|&(c, _)| c == *k) && !affine_keys.contains(k))
        .collect();
    assert!(
        stray.is_empty(),
        "off-envelope slot keys {stray:?} — the emission needs types counts* does not declare"
    );
    // publics* (wall 4): the public segment pads to ONE length with
    // published zeros, appended after every real public — tail publics
    // shift no recorded block base, and a parent's walk (H(publics)
    // rows, recombination folds) sees the same segment length at every
    // level.
    // The TAIL blocks — the inheritable accumulator claims, then the
    // application statement — are published AFTER the padding, so each sits
    // at a constant of the envelope rather than at a function of this
    // outer's live usage. That is what lets a parent read a child's claims
    // and statement at ONE index whatever kind of child it walks. A block
    // this outer has no content for is zeros, built exactly as the padding
    // is.
    let body =
        env.publics - env.acc_chain_words - ENV_ACC_MAIN_WORDS - ENV_PASS_WORDS - ENV_APP_WORDS;
    debug_assert_eq!(body, ENV_BODY_WORDS);
    let live_pub = sb.public_len();
    report.push(format!("publics {live_pub}/{body}"));
    if live_pub > body {
        over.push(format!("publics {live_pub} > {body}"));
    } else {
        for _ in live_pub..body {
            vals.push(F128::ZERO);
            sb.public_input();
        }
        for (name, w, width) in [
            ("acc_chain", tail.acc_chain, env.acc_chain_words),
            ("acc_main", tail.acc_main, ENV_ACC_MAIN_WORDS),
            ("pass", tail.pass, ENV_PASS_WORDS),
            ("app", tail.app, ENV_APP_WORDS),
        ] {
            report.push(format!("{name} {}/{width}", w.len()));
            if w.len() > width {
                over.push(format!("{name} {} > {width}", w.len()));
                continue;
            }
            for &x in w {
                sb.publish(x);
            }
            for _ in w.len()..width {
                vals.push(F128::ZERO);
                sb.public_input();
            }
        }
    }
    // The live/target census: target is the live count (or one
    // schema-preserving dummy row).
    println!("  [envelope rows live/target] {}", report.join(" | "));
    // Overshoot is a real failure: the public segment or a tail block
    // outgrew the envelope's fixed layout.
    assert!(over.is_empty(), "counts* overshoot: {}", over.join(", "));
}
