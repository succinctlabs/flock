# Bloat-reduction plan

Status: **PHASE 1 EXECUTED 2026-08-27** on branch `bloat-phase1` (PR #37);
the execution record and every deviation are in `docs/bloat-ledger.md`.
Planned 2026-08-26. Baseline: 156,524 lines of
Rust across the workspace + 13,113 CUDA/C++; `main` at the post-stack merge
(#26/#32/#33/#10/#34 landed, PR queue empty, tower measured 672 ms/leaf
amortised all-AG).

Goal (Ron, 2026-08-26): a significant effort to reduce bloat — dead code,
duplication, and machinery that preceded multitable support and is obsoleted
by it.

## Where the bloat is

**A. Provably-dead public API.** The workspace is 0-warning, so private dead
code is largely gone — but `pub` items with no callers are invisible to
rustc, and review rounds kept finding them (deleted so far: the F128 basis
verifiers, `zero_dead_regions`, `BundleFlavor::Chain`, `HashKind`,
`GKR_PAR_DIAG`, dead timing fields, `env_free_counts`). Nobody has swept
systematically.

**B. Pre-multitable machinery.** The union transport obsoleted standalone
batch proving, but the direct/padded-commit path survives with these
consumers:
- the **GPU prover** (proves direct-shape blake3 — the biggest anchor),
- **keccak3** (padded commit; retirement raised 2026-08-15 and declined),
- the **Merkle-path shift protocol** (`merkle_path.rs`,
  `merkle_path_common.rs`, `merkle_glue.rs`, `merkle_r1cs.rs` ≈ 6k lines
  plus encoder slices),
- the AG direct entries (`prove_fast_ag`/`verify_ag` etc.), mostly for A/B
  benches.
Deletion here is gated on product decisions (below), not code analysis.

**C. RS/φ8 skip machinery.** Already scheduled for removal (AG endgame,
docs/ag-recursion-plan.md), blocked on the Phase F kernels (x86 AVX-512 +
CUDA AG round-1). Likely the single largest cut (rough guess 10k+ lines):
zerocheck RS arms, `univariate_skip_optimized` + kernels + deg4, φ8
friendly-challenge machinery, `MixedProof::Rs`, the tower's RS tape walks,
dual proof types, RS fixtures.

**D. Duplication.** Concrete evidence of cost: the same squeeze-map bug was
fixed twice (ChildRegion, then RealRegion), the LegacyPow omission existed in
two of four tape walkers, the consistency-bits fix landed in three transcript-
walk copies. Clusters: `tower.rs` at 20,304 lines (Child/Real walker pairs,
native replicas, gates duplicated with test files), the four encoders
drifting around `common.rs`, a per-test-file SplitMix64 Rng copied ~15×.

**E. Test/bench/doc sediment.** Rotted ignored fixtures (3 red union_element
MIXED fixtures; the element transcript-shape pin — fails on clean main),
concluded probes (`kappa7_probe`, `b3_width_audit`, `wiring_scaling`?),
21 bins (the `dump_*` ones are CUDA-CI-load-bearing; the rest unclassified),
design docs describing superseded protocols as current.

## Phases

**Phase 0 — the bloat ledger** (1 session, agent fan-out, no code changes).
Census agents per module produce `docs/bloat-ledger.md`: every pub item with
zero production callers; every ignored test classified (guard / rot /
concluded-probe); bins + benches classified (CI-load-bearing / useful /
dead); duplication clusters with line estimates. Each entry gets a verdict
and evidence (file:line, caller census). Same fan-out pattern as the PR #26
review.

**Phase 1 — uncontested deletions** (1–2 sessions). Everything the ledger
marks dead with no decision needed: dead pub items, concluded probes, rotted
fixtures (retire or re-pin, per fixture), dead bins, stale doc sections.
Batched commits; full suite + byte pins after each batch (the pins are what
make pruning safe here — a deletion that changes behavior fails loudly).

**Phase 2 — decommission decisions** (Ron's calls, then 1–2 sessions each):
0. **Profile consolidation — EXECUTED 2026-08-27** (ledger §C): the
   grind-free `Fast`/`Slim` are gone, `Fast128`/`Slim128` took their names,
   `Fast100`/`Slim100` frozen; 98 → 70 embedded TOMLs; proof-IO v22.
1. **keccak3** — retirement was declined 2026-08-15; does the reason hold?
   **EXECUTED 2026-08-27** (Ron): keccak and keccak3 both retired.
2. **Merkle-path shift product** — still a product, or a leftover now that
   the tower verifies Merkle in-circuit? Retiring ≈ 6k lines + removes a
   padded-commit consumer. **EXECUTED 2026-08-27** (Ron): shift protocol +
   monolithic Merkle block product retired; `merkle_glue` and the tower's
   test-oracle layout kept (ledger §C 2.2).
3. **Direct path / GPU** — the padded-commit direct path can't die while the
   GPU prover speaks it. Either port the GPU prover to the union transport
   (could fold into the Phase F CUDA-AG work as one "GPU catches up" effort)
   or keep direct GPU-only for an interim and delete its CPU-side non-GPU
   consumers. **Ron, 2026-08-27: wait — keep the CPU direct path until the
   GPU is ported to multitable + AG-skip.**

**Phase 3 — duplication consolidation** (careful, byte-pinned). Unify the
Child/Real tape-walker pairs; split tower.rs into a module tree (mechanical,
behavior-free); extract encoder commonalities; one shared test Rng. After
the deletions, so the surface being refactored is already smaller.

**Phase 4 — RS removal.** Gated on Phase F kernels; Phases 0–3 turn it into
executing a list (the ledger will have mapped every RS-only item).
Accelerator option (not recommended right after the GPU-port investment):
accept an AG-less x86/GPU interim.

## Guardrails against regrowth

- Probes and benches land with an expiry note (what question, when answered,
  delete on conclusion).
- `cargo-machete` (dependency dead weight) in CI.
- Periodic ledger refresh (cheap after the first).

## Sequencing

Phase 0 now → Phase 1 → Phase 2 decisions in parallel → Phase 3 → Phase F
kernels (orthogonal, interleave anywhere) → Phase 4.

## Verification methodology (all phases)

Full workspace suite + the m6 byte pins + the fold_lookahead oracles after
every batch; `cargo fmt --check` + clippy 1.98 + x86 cross-check before every
push (both bit CI in the past — see the toolchain notes in the memory
recursion-track). Transcript-affecting removals (proof-IO retirements) take
a version bump and re-pins, and the Blackwell CI revalidates the GPU side.
