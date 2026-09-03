# AG-skip in the recursion tower — plan

Status: PLAN (2026-08-18, branch `ag-union`). Goal: the tower's provers run the
AG-skip boolean zerocheck (union-AG measured −21% prove at m32) and the
recursion circuit replays those proofs.

ENDGAME (Ron, 2026-08-18): the RS/φ8 univariate skip will EVENTUALLY be
deprecated and removed — AG becomes the only skip basis. Not immediate, but
it re-weights this plan: AG-everywhere (Phase C) is the critical path rather
than optional; the in-circuit AG lows (Phase D) are the *successor* of
`emit_lagrange_lows`, not an optional upgrade; migration knobs should be
flip-in-place + delete, not permanent parallel API; and Phase F lists the
removal blockers (x86 + CUDA AG kernels chief among them).

## Status + next-session entry point (updated 2026-08-18, evening 2)

DONE: **Phase A complete** (commit d6a0eb1 — circuit-AG entries
`R1csProofCircuitMergedAg` / `prove_fast_ligerito_union_circuit_ag` /
`verify_ligerito_union_circuit_ag{,_deferred}`; fused PoW predicate aligned
to the PowMask convention; `cross_class_circuit_ag_roundtrip` green).
**Phase B slice 1 done** (5a35c4e — `chain_leaf_ag_roundtrip`: the real
chain shape proves under AG, RS 17.6 vs AG 12.3 ms at m22 dev size).
**Phase B COMPLETE** (this branch): the tower's chain leaf proves under AG
on aarch64 and the whole pipeline replays it. What landed, all in
`crates/flock-prover/src/tower/` unless noted:

1. `MixedProof` flavor enum (`Rs | Ag`) with `wiring()/pcs_open()/element()`
   accessors; `build_chain_proof` proves AG behind the private
   `leaf_zc_ag()` switch — aarch64 only (the round-1 kernel), with
   `TOWER_LEAF_ZC=rs` as the A/B override. No public TowerConfig change.
2. `ChildTape::new` AG arm: flavored recording verify; anchor
   `flock-ag-skip-v1` (and the OTHER flavor's label asserted absent);
   round-1 pins ObserveSlice(158)/(64) vs `bp.ag.round1_ab/_c`; the
   locator walk's flavored head (ONE r_outer slice; r₁'s 5-op surface:
   point label + 2 seed squeezes + nonce label + ObserveBytes(4), recorded
   as `ZskipTapeRec::Ag { seed_ch, seed_fins, nonce_payload }`); the
   `.phi8()` pin replaced by the native point pin (seed rebuilt from the
   located chals, `decode_ag_point` under
   `pcs.zerocheck_grinding().ag_r1_bits()`, compared to
   `bool_assert.z_skip`).
3. Tier 0 landed as PUBLISH-THE-WHOLE-SURFACE: per AG child the FL
   publishes `[seed₂ (wire-connected), nonce (wire-connected to the
   ObserveBytes stream word), point₅ (advice), row-lows₆₄ (wire-connected
   to the fold's absorbed lows)]` — 72 publics/child — and
   `check_ag_skip_publics` re-derives point + lows natively (fused decode,
   budget bound, functional). `emit_lagrange_lows` + the λ-table publics
   are now emitted only for RS children. The envelope publics cap (5684)
   absorbed the delta — no envelope change needed at k=2.
4. c-point baked constants: flavored — `ag_skip::friendly_challenges()`
   for AG children (same 7-slot shape as the ghash set).
5. Tests: the WHOLE existing tower suite runs AG on aarch64 (leaf
   roundtrip+tampers, tape pins, region-alone, FL, internal, spine
   converges, e2e+lane) — the RS arm stays covered by the x86 CI arm and
   the `TOWER_LEAF_ZC=rs` knob; new `ag_skip_publics_checker_rejects_tampers`
   (seed/nonce/budget/point/low tampers all rejected).

Two bugs flushed en route (both latent, both would bite anything AG):
- `RecordingChallenger` did not forward `hash_kind()` — the trait default
  (SHA-256) silently diverged the AG nonce decode from the BLAKE3
  transcript during recording (flock-core/src/transcript_record.rs).
  Nothing had ever recorded an AG proof.
- The c-point r_outer wires used a naive 4-words-per-row squeeze map;
  a FUSED slice squeeze (the AG initial grind) reserves one word per row
  for the PoW predicate, so the map misaddressed. Now
  `squeeze_word_wire` (the exact map) everywhere.

m32 A/B (tower_online_bench, CHAIN_BLOCKS=262144, warm medians, same box
same session — `TOWER_LEAF_ZC=rs` vs default): leaf online 520.9 → 469.8 ms
(prove 475.8 → 427.7, −10.1%); amortised 762 → 688 ms/leaf = 344k → 381k
compressions/sec (+10.7%). The internal/spine arms flipped between the two
runs (wide bands on one arm — box noise); the leaf bands are tight, so the
leaf delta is the solid number. ATTRIBUTION RESOLVED (same day, evening 3): **the ~50 ms "gap" was not a
real AG cost — the reference was stale.** The recorded union numbers
(466.0/367.3, margin 98.7) came from a pressured-box session; re-run fresh
on a quiet box the same bench reads 375.3/319.1 (margin 56.2, 1.176×).
Three independent same-session probes then agree at m32:
- isolated zerocheck (ag_e2e_zerocheck): 161.7 → 101.4, margin 60.3
  (round-1 URM 2.29×, fold 1.24×, mlv tail 1.00× — the win is ALL round 1);
- union prove (blake3_rs_vs_ag): 375.3 → 319.1, margin 56.2;
- leaf PIOP∥wiring phase (PCS_TRACE): 172.9 → 119.8, margin 53.1 —
  commit and open are flavor-identical code and their apparent deltas
  (±10) are the cross-run noise floor.
So the flavor margin carries through the circuit path undiminished — the
wiring (μ = 22, 11 claims) hides inside BOTH flavors' PIOP spans — and
there is nothing leaf-specific to fix. WHY the stale margin was 99: the RS
round-1 URM is memory-bandwidth-bound and inflates disproportionately
under box pressure (466 → 375 = +24% vs AG's +15%) — the same mechanism as
the recorded a503066 swapping-box incident. LESSON: flavor A/B RATIOS are
box-state-dependent; trust same-process interleaved A/Bs (the three-arm
bench's warm-up discipline), never cross-session number reuse. The honest
quiet-box m32 zerocheck margin is ~53–60 ms ≈ 1.6× isolated, 1.18× on the
union prove.

**AG/RS OPTIMIZATION PARITY (Ron's call, same day): CLOSED** — the AG
zerocheck now has every RS-branch optimization (commits on this branch):
- Run-list round 1: `PaddingSpec::block_coverage` classifies the 8192-bit
  code-block grid Dead/Full/Partial; the SLP kernels needed NO surgery —
  they are position-independent additive sums, so full runs go in as
  (slice, eq-subrange) segments and Partial blocks are cleansed
  (`cleanse_block`, bit-masked edges) into zeroed scratch. No
  declared-dead bit is ever read.
- The fused fold is gated on the same map (`fold_and_first_round_padded`),
  and below the utilization gate emits LIVE-SPAN buffers
  (`fold_and_first_round_sparse` + 128-aligned `LiveLayout`) feeding RS's
  own support-proportional rounds (`fold_and_round_pair_sparse_into` —
  the friendly constants ride as ordinary r_next weights), with ONE
  expand-to-dense on exit resuming the AG lookahead tail mid-stream.
- Consequence: the `PooledDirty` witness election dropped its RS-only
  condition — the AG honest-zero forcing is GONE (the padding contract is
  now enforced by read-exactness, not by memset).
- Differential coverage: byte-identical proofs between dense-honest,
  padded-honest, and padded-DIRTY witnesses (deliberately inconsistent
  garbage in dead regions), both grinding schedules, at 62%/8-block and
  3-of-64-block utilization; plus a 200/256-row union roundtrip re-proving
  over pooled-dirty buffers byte-identically.
The direction of remaining asymmetry is now AG-ahead-of-RS (lookahead +
friendly-Horner tail), which is fine — RS is the deprecation target.
This also pre-pays Phase C's cost profile: the envelope outers' dead
boolean space (swap 12250/16384, spread 1060/16384, pow 4096/16384 at
nu* = 14) is now skipped, not scanned, under AG.

**Phase C COMPLETE (same day, evening 4)** — the envelope outers (FL /
internal / spine) prove under AG behind the private `outer_zc_ag()`
switch (`TOWER_OUTER_ZC=rs` A/B override). `LeafOuter.proof` is the
shared `MixedProof` (with `boolean_lincheck()` +
`verify_circuit{,_deferred}` dispatch methods); `RealTape`/`RealRegion`
carry the ChildTape/ChildRegion AG arms verbatim (anchor, 158/64 pins,
seed/nonce locator, point-pin decode, `ZskipWires`, friendly c-constants,
the exact `squeeze_word_wire` map — the RealRegion emitter had the same
naive-4-per-row bug the child emitter had); `build_node_outer_app` takes
the Tier-0 publish + `check_ag_skip_publics` per AG child, with the RS
λ machinery conditional. Pipeline green under AG-everywhere incl.
`chain_spine_converges` (one-digest property holds), all four flavor-knob
combos, Chain100+Chain128.

PERF LESSON EARNED EN ROUTE: the parity slice's segment-call round-1
driver COLLAPSED at the envelope — its per-column run structure is
19,920 full / 228 partial / 110,924 dead blocks at m30 (~15% live,
~450 segments), and per-segment rayon bridges made round 1 cost 28–43 ms
(2–3× the FULL dense scan). Fixed by
`round1_slp_packed_banks_fused_padded` — ONE parallel pass over the
live-block list (Full pairs keep the 2src c-transpose, Partials cleanse
inline; char-2 addition makes visit order irrelevant) — now 2.4–4.7 ms,
count-proportional parity. The segment driver survives only as the
bench-only unfused arm. `FLOCK_ZC_TIMING` now prints the AG phases.

m32 A/B (tower_online_bench, warm medians, same box): all-RS 721 ms/leaf
amortised (364k c/s) → all-AG **660 ms (397k c/s, +9.1%)**: leaf
494.6→442.1, FL 224.1→222.1, internal 224.7→214.2, spine 228.0→213.7.
NODE-DELTA CAVEAT (Ron's catch, corrected next day): nodes were ~200 ms
under RS on a prior good-box day, so the cross-run node totals above are
mostly BOX DRIFT — node cost is ~195–230 ms under BOTH flavors by box
state. The within-run evidence (flavor-identical commit/open lining up
across runs): the node's PIOP∥wiring joint reads 35–41 ms under AG vs
45–85 under RS — a real but small ~10 ms flavor win, often swallowed by
the wiring join and noise, exactly as the join-cap mechanism predicts.
The LEAF win (~50–60 ms, three same-process probes) is the arc's real
measured speed gain; phases C/D buy the nodes uniformity for RS removal
and the soundness posture, not speed.
Proof sizes: node 252.8→254.3 KiB (+1.5, the AG round-1 message).

**Phase D COMPLETE (same day, evening 5) — with a redesign on measured
evidence.** The plan's (b) `emit_ag_lows` is MEASURED-REJECTED: the base
functional's coordinate masks are ~48% dense (920 monomials, 28,464 XOR
terms, ~24.5k MAC rows/child even with four-Russians sharing —
`base_functional_circuit_census` in flock-core is the receipt), 20× the
plan's estimate and past the mac slot's cap for two children. What
landed instead ((a)+(c), Ron's call):
- `emit_ag_point_binding` at BOTH consumers: two BLAKE3 rows recompute
  `ns = H(seed‖nonce)` and its XOF block from the child's transcript
  wires (x binds by wire connect); a PowMask row enforces the fused
  target on `ns[16..32]`; and the point coordinates are constrained to a
  fiber point over x — the factored base fiber with `s` eliminated
  (`t²+t = x³+x`, `y²+uy = xut`, inverse-free) + the three
  denominator-cleared AS levels (D₀/D₁ guarded by advice inverses).
  ~110 mac rows + 2 b3 + 1 pow per AG child. (a) on-curve is subsumed —
  the fiber constraints ARE curve membership.
- CANONICITY RELAXED BY DESIGN: any of the ≤32 fiber points satisfies
  the rows; the sampler's 5 flattening bits return to the prover and are
  repaid by the all-explicit `R1_POW_BITS = 9` (was 4+5 credit; total
  unchanged; scan budget 2^24; the lincheck site keeps 3+5 — its decode
  stays canonical end-to-end).
- `check_ag_skip_publics` slims to TWO items: the NONCE RANGE (the
  PowMask row pins only the nonce word's high half, and Chain100 emits no
  row, so the range check stays native — the PR review caught its brief
  removal) and `lows == bf(point)`. The genus-95 sampler/DRBG/AS-solver
  leave the exit contract; `bf()` and the range check stay.
Whole pipeline green under everything (all knobs, Chain100+128), 687
workspace tests, x86 clean. m32 bench with the binding in place:
amortised 637 ms/leaf = 412k c/s (leaf 429.6, FL 209.2, internal 196.6,
spine 205.0) — the ~110 rows/child and the 9-bit grind cost nothing
measurable (the run's absolute node numbers reflect a friendly box; see
the node-delta caveat above — cost-NEUTRALITY is the claim, not a win).

NEXT (order per the endgame):
1. **Phase F long-lead kernels, start now**: the x86 AVX-512 AG round-1
   (RS removal otherwise kills x86 proving) and the CUDA AG round-1
   (GPU runs full RS zerocheck on-device; mlv tail carries over).
2. Phase E audit/docs — the audit ledger's AG rows move with the 9-bit
   split (guard tests already updated); the recursive-agreement section
   records the in-circuit decode + the one remaining checker item.
3. Phase F removal (locator arm deletion, fixture/proof-IO retirement,
   SkipPoint collapse). If the lows publics ever need to go fully
   in-circuit, the recorded path is a dedicated static-table route for
   the dense mask map, not per-term MACs.

Memory track: `recursion-track.md` (machine-local) mirrors this and adds
session gotchas. The two survey reports' full maps are summarized in the
section below; anything deeper re-derives quickly from the line refs.

## What the survey established (tower @ `ag-union` tip)

1. **The boolean zerocheck's arithmetic is NOT wired in-circuit today.** The
   tower binds a child's boolean zerocheck transcript-positionally (the FS
   chain rows force the observed proof words) and consumes only four surfaces:
   the `r_outer` squeeze words (→ c-claim point wires), the `z_skip` squeeze
   wire (→ `emit_lagrange_lows` → fold row-lows, tower.rs:11210 + call sites
   10201/16655), the finals `v_a, v_b` (→ lincheck-entry replay), and
   `z_partial`. The Λ-interpolation, round checks, and the ring-switch
   `claim_check` are never arithmetized. **Consequence: AG's 222-coordinate
   round-1 costs zero new in-circuit arithmetic under the current posture.**
2. **Proof shapes**: all three levels prove `R1csProofCircuitMerged` via
   `prove_fast_ligerito_union_circuit`; leaf = 1 boolean type, no element;
   FL/internal/spine = the envelope registry (6 boolean + 15 element types).
   Tapes record `verify_ligerito_union_circuit_deferred` under a
   `RecordingChallenger`. There is no AG circuit-flavor entry yet.
3. **The fold region is already flavor-generic**: `SkipPoint::weights(6)` is
   64-wide in both bases, so `MatrixClaim` widths, the fold tape shape, and
   `emit_weight` are unchanged. Only the row-lows *derivation* (today:
   `emit_lagrange_lows` from the one `zskip_w` wire, ~260 MAC rows) is
   φ8-specific.
4. **Capacity is not the constraint**: AG round-1 adds +94 observed words ≈
   +23.5 BLAKE3 compressions per child region (~0.25% of a region; b3 slots
   have ~7k rows headroom each). An in-circuit base-functional derivation
   would be ~1.0–1.5k MAC rows per child vs the mac slot's ~49k.
5. **The AG points have no transcript word.** RS gives one located squeeze
   wire; AG's `r₁` and the lincheck's fresh skip are decoded from
   (seed = 2 squeezes, nonce = ObserveBytes(4)) through a STANDALONE hash
   (outside the duplex sponge) — the existing `Op::Pow`/PowMask machinery
   never sees the fused PoW. Binding the point is the one real design
   decision.

## The design decision: how the recursion binds an AG point

**Tier 0 — checker-published (recommended start).** Per AG child, publish
(seed₂, nonce, point₅) in the outer proof's public segment; the fold row-lows
already ride the fold tape as observed words. The consumer's native checker
(the same tier as `check_fold_publics` / `AlphaRec`) re-derives
`evaluation_point_from_nonce_pow(seed, nonce) == point` (one hash + one
attempt, ~1 µs) and `base_evaluation_functional(point) == row-lows`, and the
`.phi8()`-style native pins become point pins. Zero new gates, zero new table
types (the registry-diet lesson: type COUNT costs ~20% node time), ~8 extra
publics/child (publics count 5684 moves — it is shape, not a pinned digest).
This is exactly the documented pre-in-circuit boundary posture the stale
comments at tower.rs:16571-16580 describe.

**Tier 1 — in-circuit upgrade (later, if the exit contract wants it).**
(a) on-curve predicate for the advice point (each Artin–Schreier equation
`z² + z = rhs` is one square + add; base equation a handful of mults);
(b) `emit_ag_lows`: base functional from point wires — x-powers ≤ 31,
4 y-powers, 8 z-monomials, per-push MACs, one denominator inverse via the
existing advice-inverse/zassert idiom (~1.0–1.5k MAC rows);
(c) standalone-hash binding of H(seed‖nonce) via the `emit_opening`-style
Blake3 rows + a PowMask row + XOF-stream binding of x/slot/choice bits.
Decode-canonicity (which AS root) either pinned via linear functionals
(Frobenius-chain idiom exists) or compensated with ~5 extra PoW bits.

## Phases

**Phase A — native entries + hygiene (small).**
- `R1csProofCircuitMergedAg` + `prove_fast_ligerito_union_circuit_ag`
  (aarch64) + `verify_ligerito_union_circuit_ag{,_deferred}` — thin over the
  already-flavored shared bodies (`prove_union_with_binding_zc`,
  `verify_union_piops`/`BooleanPiopRef`). Lift the boolean-only assert for the
  mixed/circuit AG arms (element region is flavor-independent; the AG flavor
  already forces honest-zero witness mode).
- **Align the fused PoW predicate with the PowMask convention** (MSB-first
  leading bits on the hash's serialized bytes, replacing low-LE-bits) NOW,
  while the AG transcript is unshipped — Tier 0 doesn't need it, Tier 1 does,
  and it's free today, frozen later.
- Circuit-AG roundtrip tests (leaf-shaped: blake3 circuit + wiring).

**Phase B — leaf-AG (the first payoff: workload −21% at m32).**
- `TowerConfig` grows the AG flavor for the leaf (decision: new variants
  `Chain128Ag`… vs a `leaf_zc` field).
- `build_chain_proof` proves circuit-AG; `ChildTape` gets the AG walk
  (anchor `flock-ag-skip-v1`; no r_skip slice; ObserveSlice(158)+(64);
  r₁ = Label + 2 Squeeze + Label + ObserveBytes(4); AG fresh-skip 5-op shape;
  round-1 pins typed to `AgProof`), tower.rs:12288-12300 / 11560-11582 /
  11494.
- `ChildRegion`: `zskip_w` → the Tier-0 surface (publish seed/nonce/point,
  skip `emit_lagrange_lows` for AG children, checker recomputes lows +
  point derivation); `.phi8()` pin at 12392 → point pin.
- c-point baked constants: the 7 ghash inner constants →
  `ag_skip::friendly_challenges()` (tower.rs:13504-13511).
- FL-over-AG-leaves roundtrip, shape-diff, capacity asserts, then
  tower_online_bench (expect leaf ~−100 ms at m32 ≈ +12-16% throughput).

**Phase C — outers-AG (envelope nodes).**
- Same items on `RealTape`/`RealRegion` (6211, 6859-6875, 6959, 8719,
  8222-8227, 10178-10212) + mixed-class-with-element AG entries.
- Envelope re-checks (nu* ≤ 14 per b3 slot, m* = 29 content, publics count),
  internal/spine digest equality, `chain_spine_converges` re-run.
- Expect node prove −10-15% (zerocheck share at m29).

**Phase D — Tier-1 upgrade** (in-circuit lows + on-curve + hash binding).
Under the deprecation endgame this is SCHEDULED, not optional: when RS is
removed, `emit_lagrange_lows` dies and `emit_ag_lows` is its mainline
replacement — going to Tier 0 permanently would be a posture REGRESSION
(the in-circuit lows were the recorded upgrade over the checker boundary).
Tier 0 remains the right FIRST landing; D closes the loop before removal.

**Phase E — audit + docs.** Extend the audit's recursive-agreement section:
AG rows are checker-tier obligations (like the fold publics), not PowMask
rows; friendly constants ≠ 1 is free in-circuit (baked constants); update
`docs/local/recursion-handoff.md` censuses and the memory track.

**Phase F — RS deprecation prerequisites** (the removal blockers, so they
can be scheduled early rather than discovered late):
1. **x86 AG round-1 kernel.** The AG prover's round-1 is aarch64-NEON SLP
   only; RS removal without an AVX-512 port kills x86 proving entirely
   (x86 VERIFY already works — `verify_ag`/`verify_with_grinding` are
   arch-independent).
2. **CUDA AG round-1 kernel.** The GPU prover runs the full RS zerocheck
   on-device (`cuda-ghash/zerocheck_round1/2/tail.cuh`, z_skip→lincheck
   hand-off resident); RS removal needs the AG twin (the mlv tail carries
   over — it is shape-identical — but round-1 over the genus-95 product
   code is a new kernel + vectors).
3. Recursion: delete the RS locator arms + `emit_lagrange_lows` + the φ8
   fused Pow+squeeze sites (keep the AG arms structurally parallel from
   Phase B so this is arm-deletion, not surgery).
4. Profiles/grinding: the RS rows of the audit schedule retire;
   `ZerocheckGrinding::skip_bits` / `LincheckGrinding::skip_bits` collapse
   to the AG accounting; the ungrinded direct route decides whether it
   gains the fused nonce or stays no-claim.
5. Transcript/fixture retirement: every RS byte pin (m6 merged fixtures,
   mixed-class pins, chain/Merkle/keccak3/sha2 relations — all currently
   RS), proof-IO version bump, and the parallel `*Ag` structs/entries
   renamed to primary as the RS structs are deleted.
6. `SkipPoint::Phi8` and `phi8()` die; `SkipPoint` may collapse back to a
   plain `EvaluationPoint` (claim types simplify).

## Open decisions (for Ron)

1. Tier 0 vs Tier 1 to start — recommend Tier 0 first, with Tier 1 (Phase D)
   scheduled before RS removal (see endgame note).
2. TowerConfig: under the endgame, do NOT grow the public config — keep
   Chain100/Chain128 and flip the flavor in place per phase (a private
   accessor / test-only knob during migration), so the eventual state has
   no residual flavor API to delete.
3. Proof struct: parallel `R1csProofCircuitMergedAg` during migration (no
   RS wire churn now); renamed to primary when the RS structs are deleted
   in Phase F. Structure all tower locator/region code as PARALLEL ARMS so
   RS removal is arm-deletion.
4. Scope order leaf-first — recommend yes; Phase C is on the deprecation
   critical path (no longer optional-until-measured).
5. PoW-convention alignment in Phase A — recommend yes (mandatory before
   the AG transcript becomes the only one).
6. Phase F sequencing: the x86 and CUDA AG round-1 kernels are the
   long-lead items — start them as soon as AG-everywhere (Phase C) is
   validated, independent of Phase D.
