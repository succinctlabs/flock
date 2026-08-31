use super::*;
use flock_core::circuit::builder::SlotId;
use std::env::var;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// Wall 2's registry-geometry constants at the settled envelope (slim,
/// m* = 29): the UNION of the leaf-outer's and the node's type sets, at the
/// envelope maxima. Measured at the m29 fixed point (envelope_registry_diff
/// + the tower census, 2026-08-06):
///
/// - `spread_w` 20 covers the m32 FAST chain leaf's L0 depth; the m29 Slim
///   outer ladder needs 19 and leaves the high output unread.
/// - Extension-field residual work uses three reusable gates, independent
///   of prefix length (the base-field per-prefix `ResidualGate` family died
///   in the stage-3 registry diet).
/// - `nu` 14: each of the two independent child verifiers has its own
///   identical BLAKE slot, and the consolidated extension-field residual
///   gates keep every physical slot below 2^14 rows.
/// - 29 table types occupy 511 gate slots; with one public slot, the cell
///   address needs 9 bits and `mu = nu + 9 = 23` tower-wide.
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
    pub(super) counts_el: [(usize, usize); 15],
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
}

/// The APPLICATION STATEMENT's width in the envelope's public segment: the
/// hash-chain PoC's span `(h_start, h_end)`, eight 128-bit words.
pub(super) const ENV_APP_WORDS: usize = 8;

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
// F256 verification adds extension-field table claims to the shared
// registry. A live node uses 1,028 words; keep a fixed-shape margin so the
// same block carries the accumulator at every recursive level.
pub(super) const ENV_ACC_MAIN_WORDS: usize = 1152;

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
    env_acc_main_base(env) - ENV_ACC_CHAIN_WORDS
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
    EnvShape {
        // The two-child envelope fits at 14 after consolidating the F256
        // residual tables and assigning each independent child verifier to
        // its own identical BLAKE slot. Every physical slot remains below
        // 2^14 rows, while 512 cell slots give mu = 14 + 9 = 23.
        nu: 14,
        // 20 = the m32 FAST chain leaf's L0 depth (log_msg_cols 19 +
        // log_inv_rate 1), which the B-fast PoC's first-level node walks;
        // the m29 slim outer ladder needs only 19 and leaves the top
        // output unread.
        spread_w: 20,
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
            (700, 9000),  // extension-field Ligerito spine
            (701, 15000), // extension-field multiply-accumulate
            (601, 300),   // assist
            (8, 4200),    // leaf-eval 8-lane
            (808, 4200),  // extension-field leaf evaluation
            (318, 15000), // prefix w 8
            // The extension residual relation is decomposed into three
            // shared tables: one normalized-weight row and one accumulator
            // row per query, plus one three-factor prefix row per later
            // Ligerito level. These caps are the sums of the former six
            // per-prefix variants at the envelope maxima above.
            (880, 4150),   // normalized W_0..W_18 chain
            (881, 12690),  // three-factor extension prefix
            (882, 4150),   // eight-way residual accumulation
            (1008, 15000), // extension prefix w 8
        ],
        // Preserve the existing public body while enlarging ACC_MAIN.
        publics: 5684,
        // The committed lane count — the ONE piece of a child's layout that
        // stays circuit structure (the parent hashes `num_lanes`-word
        // leaves), so it is pinned while everything count-shaped rides the
        // jagged claims. 24 covers every envelope member's content-derived
        // count at min-one-row. F256 raises the largest live content to 25
        // lanes; 31 stays below `2^initial_k = 32`, so children remain
        // lane-major.
        lanes: 31,
    }
}

/// Find-or-create a slot under this file's keyed-cache scheme
/// (0 spine / 8 leaf-eval / 400 mrs / 500 zcr / 600 mac / 601 assist /
/// 602 fold-mac / 700 spine256 / 701 mac256 / 808 leaf-eval256 /
/// 880 resid-weights / 881 resid-prefix3 / 882 resid-acc /
/// 310+w prefix / 1000+w prefix256). Every element-slot declaration on the
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

/// Declare the envelope's 29 table types in one canonical order.
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
    slot_cached(sb, cache, 700, SpineGate256::new);
    slot_cached(sb, cache, 701, MacGate256::new);
    slot_cached(sb, cache, 601, AssistLayerGate::new);
    slot_cached(sb, cache, 8, || LeafEvalGate::new(8));
    slot_cached(sb, cache, 808, || LeafEvalGate256::new(8));
    slot_cached(sb, cache, 880, ResidualWeightsGate256::new);
    slot_cached(sb, cache, 881, ResidualPrefix3Gate256::new);
    slot_cached(sb, cache, 882, ResidualAccGate256::new);
    slot_cached(sb, cache, 310 + env.pf_w, || PrefixGate::new(env.pf_w));
    slot_cached(sb, cache, 1000 + env.pf_w, || PrefixGate256::new(env.pf_w));
    q
}

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
    let floor1 = |sb: &ShapeBuilder, s| sb.rows_in_slot(s).max(1);
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
    for &(key, count) in &env.counts_el {
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
        .filter(|k| !env.counts_el.iter().any(|&(c, _)| c == *k))
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
        env.publics - ENV_ACC_CHAIN_WORDS - ENV_ACC_MAIN_WORDS - ENV_PASS_WORDS - ENV_APP_WORDS;
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
            ("acc_chain", tail.acc_chain, ENV_ACC_CHAIN_WORDS),
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
