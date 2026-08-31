# Bloat ledger

Status: **CENSUS COMPLETE 2026-08-27**, `main` @ `86d5fd5`. Phase 0 of
`docs/bloat-reduction-plan.md`. No code was changed to produce this document.

**PHASE 1 EXECUTED 2026-08-27** on branch `bloat-phase1` (nine commits,
+605 / −10,175 lines, **net −9,570**; PR #37): fixture re-pins, doc sediment, the §A dead-pub sweep,
the §E concluded probes, the §B/§F superseded benches + exclusive src, and
the genus95 prototypes. Every batch verified (full suite, fmt, clippy
`-D warnings`); x86 cross-check clean (17 pre-existing x86-only dead-fn
warnings in the NTT experiment arms are byte-identical to main — a future
cleanup candidate). Deviations from the written plan, with reasons:
- Census corrections found by the compiler: `AdditiveNttGf8::k`,
  `SparseEqTensor::len`, `BaseFunctional::len` were NOT dead (restored);
  `with_blake3_chunk_leaf` was test/bench-only, not zero-reference.
- Kept: the unfused `round1_slp_packed_banks` kernel (differential oracle),
  `is_empty` twins wherever a live `len` remains (clippy pairing).
  (`LOOKAHEAD_FRIENDLY`, `DISABLE_FRIENDLY_HORNER`, `NXT_ZEROFILL` were kept
  at first, then deleted 2026-08-27 on Ron's call once the PR #37 review
  showed no bench could flip them; the −1.5% friendly-lookahead finding is
  recorded in `ag_skip.rs` at the lookahead pass.)
- Second-order zero-caller pub API the sweep itself created, deleted
  2026-08-27 after the PR #37 review: `prewarm_prover_union`,
  `sparse_eq_from_parts`; the `cfg(test) pub` `prove/verify_merkle_path_shift`
  wrappers moved into `merkle_path.rs`'s `mod tests`.
- blake3's AG entries (`prove_fast_ag`, `prove_fast_union_ag`, …) kept —
  guarded API and the AG future; only the `*_timed` twins died. No
  rs_vs_ag arm was salvaged: correctness guards remain in blake3.rs and
  the tower benches carry AG e2e perf.
- The two rotted area-asserts in union_element now derive the b3 width
  from `USEFUL_BITS` instead of a hardcoded 93.
- Deferred: the orphan `dump_transpose_induce_vectors` (needs a Makefile
  run target validated on the Blackwell box — do it as a CUDA-CI
  follow-up); the §B judgment tier; the §E UNCLEAR probes.

Goal clarification (Ron, 2026-08-27, post-census): **flock's goal is to prove
recursively — the tower is the key product surface.** Verdicts on
tower-reachable code are framed accordingly (see PRODUCT-TRACK below); the
bloat campaign protects and hardens the recursion path, it does not trim it.

Method: twelve per-subsystem census agents (same fan-out as the PR #26 review)
swept every `pub` item, every `#[ignore]` test, every bin/bench/example, the
duplication clusters, and the doc tree; the highest-value claims were then
re-verified centrally by direct grep (tower/permutation reachability, the
Setup-type construction sites, the merkle_glue tower import, the orphan dump
bin, `bit_transpose_64bytes`, the dead singletons). One agent claim was
corrected during verification: `with_blake3_chunk_leaf` is test/bench-only,
not zero-reference dead.

Baseline: 156,524 Rust lines in `crates/` + 12,610 CUDA/C++ in `cuda-ghash/`.

## Headline numbers

| Bucket | Lines (est.) | Phase |
| --- | --- | --- |
| Uncontested deletions (dead pub, concluded probes, superseded benches, bench-only modules with no oracle role) | **~9,500–10,500** | 1 |
| Decision-gated: keccak retirement | ~2,650 (+1,040 bench) | 2.1 |
| Decision-gated: Merkle-path product | ~4,140 (+670 bench) | 2.2 |
| Decision-gated: direct/padded CPU entries | ~1,100 | 2.3 |
| Duplication consolidation (recommended clusters) | **~8,500** | 3 |
| RS/φ8 removal (now fully mapped) | **~10,000** | 4 |

Buckets overlap slightly (RS-tagged bench-only items are counted once, in the
earliest phase that can delete them). A campaign that lands Phases 1–4 in full
removes on the order of **35–40k lines**, ~25% of the Rust workspace.

## How to read the verdicts

- **DEAD** — zero references anywhere outside the definition. Delete in Phase 1.
- **TEST-ONLY / BENCH-ONLY** — reached only from tests, benches, bins, or
  examples. **This does not automatically mean delete.** Three legitimate
  sub-kinds recur and are labeled where they apply:
  - **API-ENTRY**: flock ships as a library; the hash-statement setups
    (`Blake3Setup`, `Sha256HybridSetup`, `KeccakSetup`), `proof_io`, and
    `CircuitBuilder` are product surface whose in-repo callers are necessarily
    benches/tests. Not deletion candidates on this evidence alone.
  - **ORACLE**: reference implementations kept so tests can check the
    optimized kernel differentially (e.g. `merkle_tree_sequential`, the
    genus95 `round1_raw*` family, `fold_1b_rows_naive`). Deleting one deletes
    the differential test with it — per-item judgment.
  - **PRODUCT-TRACK**: the recursion tower. **Flock's goal (Ron, 2026-08-27)
    is recursive proving — the tower is the key product surface, not an
    internal test.** Today `pub mod tower` (20,304 lines) is referenced by
    nothing outside `tower.rs` and is driven by its own `#[ignore]` test
    benches; the census records that as a **productionization gap** (public
    entry points, typed config instead of the `TOWER_CONFIG` env knob, a
    non-test caller, promoted tests), not as bloat. Nothing tower-reachable
    is a deletion candidate. The census facts recorded here — the 41-region
    map, the ~420 RS-only lines, the walker-duplication measurements — are
    the map for *hardening* the tower (Phase 3 module split, Phase 4 RS-arm
    removal). The k-ary arity paths (`build_fl_node_k` /
    `build_node_outer_app`, every current call site arity 2) and the
    `SpineIn::forge` adversarial branch are forward runway / adversarial
    coverage — the recursion track decides their fate during
    productionization, not this campaign.
- **RS-SCHEDULED** — exists only for the RS/univariate-skip/φ8 path. Phase 4.
- **DECISION-GATED** — dies with a Phase 2 product decision.

---

## A. Dead pub API (Phase 1, uncontested)

~60 items, ~600 lines directly + demotions. Every entry verified to have zero
non-doc references outside its definition.

**flock-core / pcs + merkle** (`pcs/ligerito.rs`, `pcs/commit.rs`, `merkle.rs`):
- `LigeritoProfile::parse` (ligerito.rs:144, ~12)
- `SumcheckProver::new_jit` (:4329, ~16), `::fold_blocked_jit` (:4345, ~21),
  `::f_len` (:4502) — the F128 JIT fold has no consumer (JIT lives in
  `extension::SumcheckProver256::first_fold_jit`)
- `PcsParams::log_leaf_f128_count` (commit.rs:223), `::l0_queries` (:275)
- `MERKLE_PCORES_ONLY` static (merkle.rs:528) — never written by any caller;
  the env var at :542 is the only live control

**flock-core / pcs.rs, jagged, ring_switch, matrix_fold**:
- `row_marginal` (matrix_fold.rs:668) — only the `col_marginal` twin is used
- `fold_1b_rows_multi` (ring_switch.rs:181) — callers all use `_padded`
- `RsEqInd::{is_empty, add_scaled_into, into_dense}` (ring_switch.rs:2289,
  2306, 2359), `SparseEqTensor::{len, is_empty}` (:1888, :1892)
- `verify_succinct` (ring_switch.rs:3043) — live path is `_with_grinding`
- `verify_frobenius_assist_deferred` (jagged.rs:2071),
  `verify_multipoint_twisted_deferred` (jagged.rs:4420)
- `pub use pack::unpack_witness` (pcs.rs:34) — `pack.rs:84` itself is only
  cfg(test)-called

**flock-core / zerocheck, ntt, genus95**:
- `ZerocheckGrinding::is_enabled` (zerocheck.rs:86)
- `eq_eval_binary_x` (multilinear.rs:281, ~25),
  `fold_packed_witness_at_z` (multilinear.rs:1167, ~23)
- `AdditiveNttGf8::k` (ntt.rs:136)
- `BaseFunctional::{len, is_empty}` (genus95/base_evaluator.rs:32, :38),
  `ProductFunctional::is_empty` (genus95/evaluator.rs:47)

**flock-core / circuit, element, lincheck**:
- `Circuit::fixed_public` accessor (circuit.rs:551; the field is live),
  `wire_cells` (circuit.rs:1247, ~16)
- `CircuitShape::{num_inputs, num_hints}` (builder.rs:807, :812)
- `SparseF128Matrix::nnz` (element_r1cs.rs:146)
- `lincheck::prove_padded` (lincheck.rs:1680, ~27),
  `lincheck/union::verify_union_timed` (:952, ~20 — re-exported at
  lincheck.rs:133, nothing consumes the re-export)

**flock-core / misc**:
- `verify_ligerito_timed` + `VerifyPhaseTimings` (verifier.rs:161, :50 — ~100
  lines, die together)
- `Accumulator::discharge_with_circuits` (aggregate.rs:117 — its own doc says
  "measured SLOWER… prefer `discharge`"), `fold_and_discharge`
  (aggregate.rs:741 — the only grep hit is a test-fn *name*, verified)
- `TranscriptShape::is_empty` (transcript_record.rs:184)
- `F256::is_zero` (gf2_256.rs:73), `F8::{new, is_zero}` (gf2_8.rs:27, :32)

**flock-prover**:
- blake3.rs: `IO_CV1`, `IO_OUT_LO1`, `IO_OUT_HI1` (:255-261),
  `Blake3Setup::generate_witness` (:1664)
- sha2.rs: the dead `generate_witness` chain (:1169, :1306, :1315).
  **Correction (PR #37 review):** the private `generate_witness_ab` is LIVE
  (`sha2.rs:1454`, `:1539`) and was not deleted.
- merkle_glue.rs: `SWAP_IO_*` ×4 (:230-233) — superseded by
  `SwapTable::io_schema()`
- merkle_r1cs.rs: `io_leaf`/`io_index`/`io_root` (:607-618; `io_schema()`
  recomputes inline)
- prover.rs: `CircuitProverInput` pub visibility (:493 — demote, used only via
  the private `UnionProveBinding`)
- proof_io.rs: `MAGIC`, `MAX_BUNDLE_BYTES`, `VERSION` pub visibility (:43,
  :49, :140 — internal-only; the 89-line v2–v21 changelog block at :51-139 is
  prose bloat, v11–v18 describe RS-era transcripts)
- lib.rs:17 `pub mod merkle_path` → demote to `pub(crate)` (only consumer is
  `merkle_path_common`)

**Pub→private demotions with no line savings** (~95 items across all agents;
zero risk, do in one sweep — **DEFERRED, not executed in Phase 1**; only the
`merkle_path` demotion landed, and it needed the PR #37 re-exports because
its types appear in `merkle_path_common`'s public signatures): circuit.rs accessors, union.rs internals,
schedule.rs `Slot`/`MAX_K_LOG`, zerocheck `SPARSE_TAIL_GATE`/`gamma_pow`/
`Round1Message`/…, sha2's 34 layout constants, blake3's 8, merkle_glue's 7,
`s_id_eval`, `jagged_bilinear`, `pow_has_leading_zero_bits`, `verify_core_ag`,
`r1cs::apply_{a,b,c}`, etc. Full lists in the census transcripts.

## B. Bench-only modules and superseded paths (Phase 1, uncontested)

These have zero production callers and no oracle/API role — verified:

| Item | Lines | Evidence |
| --- | --- | --- |
| `flock-core/src/permutation.rs` — whole module | 1,041 | only callers: `benches/permutation.rs`, `benches/perm_vs_gkr.rs`. product_gkr is production (`prover.rs:1255,1347`; `verifier.rs:1018-1092`). Delete with those two benches. |
| `product_gkr.rs` non-batched path (`prove` :1252, `verify` :1318, `LayerProof`, `ProductGkrProof`, `ProductGkrClaim` + private `prove_product`/`verify_product`) | ~229 | only caller `benches/perm_vs_gkr.rs:173,175`. Production is the batched path (`circuit.rs:780`). |
| `zerocheck/univariate_skip_deg4.rs` + `_deg4_optimized.rs` + `ntt/inv_table_deg4.rs` | 1,620 | whole trio reached only from `benches/round1_deg4.rs`; `univariate_skip_deg4*` has no production caller anywhere. RS-flavored but deletable now, with its bench. |
| `ntt/parallel_f128.rs` — whole module | 365 | only its own tests; the benches/ntt.rs mention is a doc comment about a different crate. |
| genus95 `round1.rs` superseded kernel prototypes (`bin_abc*` family, `round1_raw`, `round1_lut_packed`, `round1_slp_packed_fused`) | ~846 | superseded by `round1_slp_packed_banks_fused`. **Caveat: the `round1_raw*` pair is the differential oracle for the SLP kernels — keep `round1_raw_packed` + its test, delete the rest.** Keep `_generate_slp_derived` (codegen). **Second-order sweep (after the PR #37 review, Ron's call):** the prototype deletions orphaned a further ~2,600 lines that the file-level `#![allow(dead_code)]` hid — the legacy bench SLP (`M_MASK` + the 2,009-line `encode_slp`), the linear-C prototypes (`encode_c`, `deferred_c`, `encode_c_derived`), the four-Russians LUT encode (`derived_lut`, `encode_lut_v`, `transpose_128x160_hybrid`), the single-bank `transpose_fold_c_2src`, the test helpers `bitslice`/`scalar_ref`/`scalar_c`/`par`, and the `kernel_vs_m2_evaluator_span` diagnostic (whose only purpose was the legacy `M_MASK` labeling; its own body said "drop this diagnostic"). Measured by removing the `allow` and iterating rustc's dead-code lint to a fixed point (two rounds); the `allow` stays removed. Kept, with callers: `round1_slp_packed` (prover round 1), `*_banks_fused` / `*_fused_padded` (production), `round1_slp_packed_banks` (unfused oracle for the two fused-vs-banks tests), `round1_raw_packed` + `round1_raw` + `blocks_from_packed` + `encode_direct` (the raw oracle chain), `derived_m` (production `M`) and `m_derived_from_evaluator_is_identity_bridge` (its oracle). `round1.rs`: 4,014 → 1,419 lines. |
| genus95 product-code path (`ProductFunctional`, `product_code_message`, `ProductMessage`, `evaluate_product_functional`) | ~273 | **CENSUS ERROR — LIVE, do not delete.** `product_evaluation_functional` / `ProductFunctional` are production: the AG-skip verifier's round-1 AB evaluation `eval_ab_at` (`ag_skip.rs:147`, called from the verify path at `:1929`) builds the 222-coord product functional at `r₁`. `product_code_message` / `evaluate_product_functional` have only test + bench callers, but those tests are oracles for production code: `base_evaluator_matches_product_path_at_sampled_points` (the Sage audit's `check_base_evaluator_matches_product` mirror, validating the 64-coord base evaluator that `lincheck.rs:121` uses) and `m_derived_from_evaluator_is_identity_bridge` (validating `derived_m`, the production kernel's `M`). The census verdict looked only at `lincheck.rs` and missed `ag_skip.rs`. Re-verified 2026-08-27 during the second-order sweep. |
| blake3.rs direct/AG A/B entry cluster (`prove_fast_ag{,_timed}`, `verify_ag`, `prove_fast_union_ag`, `verify_union_ag`, `prove_fast_timed`, `direct_pcs_params`) | ~190 | sole caller `benches/blake3_rs_vs_ag.rs` (itself superseded, §F). Salvage the union+AG arm into `blake3_proof` or `ag_e2e_zerocheck` first. |
| prover.rs bench-only `*_timed` chain (`prove_fast_ligerito_timed` :1992, `prove_fast_ligerito_ag_timed` :1674, `ProvePhaseTimings` :1973) | ~280 | reached only via the blake3/keccak3 `*_timed` wrappers → benches. **Phase 1 removed the blake3 wrapper only; `prove_fast_ligerito_timed` + `ProvePhaseTimings` survive via keccak3 and go with the §2.1 keccak decision.** |
| A/B toggle statics + dead arms (`ROUND1_UNFUSED`, `LOOKAHEAD_DISABLE`, `NXT_ZEROFILL`, `DISABLE_FRIENDLY_HORNER` + the unfused segment driver `ag_skip.rs:1389-1439`) | ~150 | **Correction (PR #37 review): `LOOKAHEAD_DISABLE` is still flipped by the retained `benches/ag_breakdown.rs:106-116` — not dead, do not delete.** `DISABLE_FRIENDLY_HORNER`, `NXT_ZEROFILL` and `LOOKAHEAD_FRIENDLY` lost their only writers with the `blake3_rs_vs_ag` / `ag_lookahead_ab` deletions and are now unflippable in-tree — deleted 2026-08-27 (Ron's call) together with `lookahead_friendly_pass`, `shl_xor_generic` and the `prove_tail` A/B arms; `LOOKAHEAD_DISABLE` stays (`ag_breakdown` + `lookahead_matches_classic` need it); the answers are recorded in `ag_skip.rs:44-56` doc comments and `docs/ag-recursion-plan.md`. |
| `with_blake3_chunk_leaf` + chunk-leaf remnants (merkle_r1cs.rs:459 + `*_chunk` reachability) | ~42 direct | L0-table revert leftover (`4e96d23`); remaining callers are `tests/merkle_glue.rs:129`, `benches/merkle_l0_opening.rs`, `tower.rs:1047` (cfg(test)). Dies with the `merkle_l0_opening` bench; the `*_chunk` witness family (~500 lines, §G cluster 13) follows the Phase 2.2 decision. |

**Test-only-but-keep** (explicitly not Phase 1 targets): `RandomChallenger`
(test infra, ~80 call sites), `merkle_tree_sequential` (oracle),
`regen_embedded_tomls` + `gen_ligerito_configs` example (codegen for 98
embedded TOMLs, `ligerito.rs:533-541`), `virtual_b_microbench` /
`virtual_a_node_shape_bench` (byte oracles), `assist_shapes_probe`
(revision-proof instrument), the `TranscriptShape` inspection surface
(deliberate diagnostics), `init_perf_thread_pool` (documented startup API).

**Judgment tier** (test-only API that doubles as differential coverage — sweep
in Phase 1 only with per-item review): the jagged.rs standalone
sumcheck/assist API (~600 + ~300 stranded helpers), the ring_switch
single-claim + non-unbatched API (~700), the ligerito.rs legacy F128
prover/verifier family (~750 — **`recursive_prover_with_basis` and
`SumcheckProver` are consumed by four CUDA dump bins; CI-load-bearing, keep**),
the non-grinding wrapper pattern (~250 across lincheck/element/circuit — every
production caller uses the `_with_grinding` twin), `FsChain` (~259; production
tape path is `trace_duplex` → `FsChainSponge`), the hand-written
`Blake3LincheckCircuit`/`Sha2LincheckCircuit` walkers (~410; production uses
`csc_lincheck_circuit()`), `mixed.rs` `MixedSetup`/`MerkleMixedSetup` (~284),
the standalone element PIOP (§C note), `verify_ligerito_union_mixed_class_deferred`
(verifier.rs:763, ~83 — the matrix-claim-accumulation measurement artifact).

**Recursion-track carve-out (goal clarification 2026-08-27):** items whose
only callers are the tower / recursion test surface are part of the recursion
product's API-in-progress and are **excluded from bloat deletions** — the
recursion track owns them: `CircuitBuilder` + `BuiltCircuit` methods,
`FsChain` (the circuit-builder FS tests ride it), the `matrix_fold`
fold/verify entries (`prove_fold`, `verify_fold`, `prove_fold_jagged`,
`verify_fold_jagged`, `DenseMatrix` — the folding-verifier protocol API),
the `aggregate` non-grinding wrappers, and
`prove_fast_ligerito_union_circuit{,_ag}` (the tower's prove entries — the
docstring's "the production shape" is now literally true).

## C. Decision-gated inventory (Phase 2 — Ron's three calls, corrected numbers)

**2.1 keccak retirement.** The census corrects the scope: **`keccak.rs`
(1,693 lines) has no production caller either** — `KeccakSetup` (union path)
is constructed only by `hash_throughput` and the keccak3 benches. keccak3.rs
(960 lines, padded commit, self-documented "next candidate for
consolidation-or-retire" at :490) + keccak.rs single-wide setup (~145) + the
bit-`State` reference chain (~142, consumed only by keccak3 tests) fall
together. **Retiring keccak entirely frees ~2,653 lines + 4 benches (1,044
lines)**; keeping keccak-as-product but retiring the 3-wide keccak3 frees
~1,250. The GPU is NOT an obstacle: `keccak3_witness.cuh` is included only by
`bench_keccak3_gpu.cu` (bench-side, not the roundtrip harness).

**2.2 Merkle-path product.** The "~6k lines" estimate splits into:
- the shift protocol proper: `merkle_path.rs` 1,015 + `merkle_path_common.rs`
  574 + sha2.rs merkle section ~220 = **~1,809**, whose only entry points are
  four `Sha256HybridSetup::*_merkle_path*` methods called only from
  `benches/sha2_merkle_proof.rs`;
- the monolithic Merkle block product: `merkle_r1cs.rs` **2,330** (only
  `SLOT_WORDS` is production-reachable, via tower.rs:27);
- **NOT gated: `merkle_glue.rs` (1,234)** — `SwapTable`, `BitSpreadTable`,
  `PowMaskTable`, `FamilyTransposeTileTable` are production recursion-tower
  gates (unconditional import at tower.rs:4063, uses at 4114/4189/5660/5809).
Retirement ≈ **4,139 lines** + 2 benches (667) + 6 ignored guard tests +
`prover.rs` `ProveCore`/`prove_fast_core` chain (~163).

**2.3 Direct/padded path & GPU.** The GPU anchor is **blake3-only**:
`flock-cuda-ffi` declares one extern (`flock_cuda_device_count`); the
roundtrip harness references only `blake3::build_block_r1cs` and builds its
FFI decls locally; `blake3_witness.cuh` is the only witness kernel in the
prove path; just 2 of 20 dump bins touch the hash modules (both blake3).
So keccak3/sha2/merkle decisions **do not affect the GPU**, and the CPU-side
direct-path surface that can retire independently of the GPU port is:
`prove_ligerito` (~120, test-only), `prove_ligerito_ag` (~48),
`prove_fast_ligerito_from_witness` (~66, keccak3's entry),
`prove_fast_ligerito_ag_from_witness` (~114) + the §B `*_timed`/AG cluster —
**~1,100 lines**, most of it double-counted with the keccak3 and rs_vs_ag
deletions. The remaining direct-path consumer after Phase 1+2.1 would be the
GPU roundtrip alone.

**Directed (Ron, 2026-08-27): consolidate the profile matrix — delete the
grind-free `Fast` and `Slim` bases and keep the `*128` schedules as THE
strict profiles.** **EXECUTED 2026-08-27** (branch `bloat-phase2-profiles`):
`Fast128`/`Slim128` renamed to `Fast`/`Slim`, the grind-free variants and
their 28 TOMLs deleted (98 → 70 embedded configs); `gen_ligerito_configs`
regenerates the renamed set byte-identically, so the derivation and the
embedded files agree. Proof-IO v22. Re-pinned (two deterministic print runs
each): `union_m6_fixtures` (6), `union_element` (7), `transcript_shape` (1).
Tower pins hold — `Chain128` already ran on the `*128` twins. The CUDA
host-only ladder replay (`make ligerito_f256_host`, m22 fast) matches the
F256 driver on every proof field; the Blackwell GPU run needs the runner
back. `Fast100`/`Slim100` unchanged (frozen historical schedules).

*Premise correction (PR #37 review).* An earlier draft of this item said the
`*128` twins "differ from their bases only in the per-level rate ladder".
That is false for Fast. `Fast128` = the aggressive ladder (rate +2/level,
`derive_profile_ladder`) **plus 16-bit per-level query PoW** (the query term
targets 112 bits and the PoW supplies the rest, work-normalized:
`m32_fast128.toml` has `grinding_bits = 16` / `expected_eps_query_bits =
112.2` at every level where `m32_fast.toml` has `0` / `128.3`) **plus larger
claim-batch / consistency-batch grinding at the deeper levels** (claim_batch
7..11 vs 6..8). `Slim` already grinds 16 bits, so on the PoW axis
`Slim128` vs `Slim` really is ladder-only. The wrong premise came from the
stale `Fast128` docstring at `ligerito.rs:99` ("no query PoW as Fast"; it
predates the 2026-08-14 change) — corrected alongside this note. The
`query_grind` / `query_target_bits` matches at `ligerito.rs:~1795` are the
authoritative statement of each profile's schedule.

*Decision (Ron, 2026-08-27):* the `*128` schedules survive, PoW included;
`Fast` and `Slim` go. When executing, record that the enum `Default`
(`Fast`), the CUDA roundtrip vector dump (`dump_ligerito_f256_vectors.rs`)
and every byte-pinned fixture move from pure-query 128-bit to
work-normalized 128-bit (112 query bits + 2^16 hash trials per level), and
that prover cost gains the per-level grinding; the 128-bit audit rows and
`docs/recursion-100-128-variants.md` must say the same. *Naming (Ron,
2026-08-27, confirmed):* the survivor keeps the base name — `Fast128` is
renamed to `Fast` (serde `fast`, `m*_fast.toml`), likewise `Slim128` →
`Slim` — so selectors and the `Default` do not move. *The `*100` twins
(Ron, 2026-08-27, decided):* `Fast100`/`Slim100` stay at their historical
schedules — they are the fixed cost points the chain100 envelope was
iterated against. After the merge they differ from `Fast`/`Slim` in three
dimensions (ladder, PoW, target), so retire the "the base with only the
query target changed" wording in `ligerito.rs` and
`docs/recursion-100-128-variants.md` when executing; do not re-derive them.

Touches (count corrected by the review): the `LigeritoProfile` enum + its
grinding-policy matches (commit.rs ×6; ligerito.rs ~12 sites including the
`query_grind` / `query_target_bits` matches; `merkle_path.rs:127
MerklePathGrinding::for_profile`; `examples/gen_ligerito_configs.rs`), the
embedded TOML set (98 configs → ~70; `gen_ligerito_configs` regenerates),
`TowerConfig::{Chain100,Chain128}`'s profile selection, the 128-bit audit
doc rows, and `docs/recursion-100-128-variants.md`. **Transcript-affecting
for strict Fast/Slim users** (the query/rate/PoW schedule moves): needs a
proof-IO version note, fixture re-pins (`union_m6_fixtures`,
`union_element`, `transcript_shape` — all three run by default since PR
#37), and a Blackwell CI pass. Sequencing: fine any time; cheapest bundled
with another transcript-moving change so the re-pin cost is paid once.

**Also gated (new finding):** the standalone (non-union) element PIOP
(`element_r1cs::prove/verify` + `ElementProof`/`ElementClaim`/
`ElementStatement` + standalone lincheck/zerocheck wrappers, **~660 lines** +
~45 stranded config helpers) opens packed-direct claims — same family as the
direct path. It is also the differential oracle for
`element_only_agrees_with_the_standalone_proof`. Recommend: keep until the
direct path dies, then delete together with that guard test.

## D. RS/φ8 removal map (Phase 4 — now a list, not a guess)

Total: **~10,000 lines**, concentrated as follows.

| Where | Lines | Notes |
| --- | --- | --- |
| zerocheck.rs RS prover/verifier | ~1,700 of 2,262 | keep `PaddingRun`/`PaddingSpec`/`BlockCoverage`/`cleanse_block`/`ZerocheckGrinding` core (~400) — consumed by ag_skip/schedule/union. Still LIVE today via prover.rs:145/1095/1900/2042, verifier.rs:175/926/1425. |
| univariate_skip.rs | ~760 of 912 | **relocate, don't delete**: `build_eq` (ag_skip.rs:1202, tower.rs:6702) and `SplitEqGhash` (product_gkr.rs ×5) — ~150 lines move out first. |
| univariate_skip_optimized.rs + kernels | ~2,600 of 2,678 | **relocate `bit_transpose_64bytes` + its NEON/AVX-512 arms (~120) to bits.rs** — consumed via bits.rs:54 by r1cs_hashes/common.rs, keccak.rs, r1cs.rs. |
| ntt/inv_table.rs | 570 | fully removable. |
| ntt.rs `AdditiveNttGf8` | ~150 of 269 | consumers are inv_table*, zerocheck, benches only. |
| multilinear.rs RS surface | ~700 | `subspace_denominator_pair`, `UniSkipFoldTable`, the `uni_skip_*` family. **CUDA CI pins `uni_skip_fold_and_round_pair_optimized_packed` (Makefile:90) — GPU still runs RS round-1; this row waits for the Phase F CUDA-AG port.** |
| gf2_8.rs + field/phi8.rs | ~1,450 | every production consumer is the RS surface + the tower rs_lam blocks. |
| tower.rs RS arms | ~420 | `MixedProof::Rs` + `ZskipTapeRec::Rs` + `ZskipWires::Rs` + both `rs_lam` φ8 blocks + PHI_8 checker rebuilds + RS tape-head walks + `emit_lagrange_lows`; collapses three enums to structs. |
| deg4 trio | (1,620) | already deletable in Phase 1 (§B) — not counted here. |
| RS benches (`round1`, `zerocheck_phases`, `e2e_zerocheck`, RS arms of `ag_e2e_zerocheck`, `round2`) | ~960 | `round2` pins the CUDA-CI kernel — last to go. |
| proof_io v11–v18 changelog prose | ~40 | with the §A cleanup. |

## E. Ignored tests (127 attributes classified)

**GUARD 83 / CONCLUDED-PROBE 37 / UNCLEAR 5 / ROT 2.**

**ROT — re-pin, do not delete (Phase 1, first batch):**
1. `tests/union_element.rs:1307` `mixed_class_merged_proof_bytes_pinned` — 7
   hex pins from `f0996e6` (08-13); `700cace` (08-19, per-level grinding
   off-by-one) re-pinned `union_m6_fixtures` but never this file. Re-pin +
   history line. **Done in Phase 1. Provenance (PR #37 review, measured by
   building `700cace~1`): the two element-only pins that moved did so under
   `700cace` alone (`nu12-0` held throughout); `mix-100-90` / `mix-100-0`
   had already reached their new values on 08-14 (`176c869`, blake3
   Option F) and `700cace` left them; `mix-128-128` / `mix-0-90` moved
   under both. Runs by default and verifies each fixture since PR #37.**
2. `tests/transcript_shape.rs:265` `element_only_transcript_shape_is_pinned` —
   digest last re-pinned 08-05; `700cace` changed per-level `Pow{bits}`.
   Re-pin via `TRANSCRIPT_SHAPE_PRINT=1`. **Done in Phase 1. Provenance
   (PR #37 review): at `700cace~1` the digest was already a third value
   (`c99198b9…`), so the shape moved at least twice since 08-05 — the 08-11
   assist-transcript fork (`4787509`) and then `700cace`. Runs by default
   since PR #37.**

**Stale memory-ledger rows resolved:** the 15 `circuit_merkle` tests @
`59525a4` — the file was *renamed* into `src/tower.rs` (`c1cdb1b`), the mvp
half deleted (`514b68a`, −9,800 lines); 12 ignored descendants live in
tower.rs. `kappa7_probe` runs un-ignored by design and is out of scope.

**CONCLUDED-PROBE (delete, ~3,300–3,700 lines):** whole files
`tests/b3_width_audit.rs` (372), `tests/wiring_scaling.rs` (137),
`tests/jit_fold.rs` (138); the union_mixed capacity family (C4: smoke ×2,
sweeps ×2 + 135-line helper, attribution, two_blake3_tables_vs_direct);
`mixed_class_cost_probe` (300); the ligerito ladder-size sixpack + the two
slim-ladder experiments (the shipped ladder IS the baseline arm);
`lookahead_fold_ab_probe`; jagged `scaling_diag`/`bits30_breakdown`/three
assist-cost duplicates (C3 — `assist_shapes_probe` + `runtime_assist_m25`
survive); `gkr_skip_cost_bench`, `fold_scaling_probe`;
`zerocheck_grinding_overhead_probe`; `zzz_bench_fold_1b`;
`f256_mul_throughput_probe`; `_bench_encode_paths`; keccak3 `pack_win`;
`sha2_linid_drop_sim`; `factored_basis_cost` (test only, keep the file's
default-run pin); `measure_integer_lane_savings` (or keep only its ratio
assert); `wiring_cost_probe`. **Tower-internal probes
(`envelope_content_probe`, and the UNCLEAR `chain_child_region_emits_alone`)
are deleted only with recursion-track sign-off** — with the tower as the
product surface, its instruments may be re-used for the next envelope/bisect
question. Full table with per-item evidence in the census transcript.

**UNCLEAR (run to decide):** `pcs_ligerito_backend_roundtrip` (candidate to
**un-ignore** — 50–100 ms; note its documented invocation string doesn't
select the fn name), `merged_transport_m30_probe` (only asserting capacity
claim; 25 ms absolute bound may need recalibration),
`mixed_low_utilization_smoke`, `mixed_m30_throughput` (only standing m30
headline — Ron's call), tower `chain_child_region_emits_alone` (bisect
leftover).

## F. Bins, benches, examples (63 targets, 12,664 lines)

Facts: **no bench or example is executed by CI** (lint/test workflows only
compile them via `--all-targets`); the only executed targets are the `dump_*`
bins via `cuda.yml:81` → `cuda-ghash/Makefile:359-363`.

- **CI-LOAD-BEARING (22, keep):** 19 dump bins with verified Makefile rules +
  `dump_common/merkle_octopus.rs`; `dump_blake3_lincheck_matrices` (manual GPU
  harness, feeds `bench_ligerito.cu:515`); `profile_prover`
  (scripts/profile-aggregate.sh:8); `gen_ligerito_configs` (codegen source for
  the 98 embedded TOMLs — do not delete).
- **ORPHAN (fix or delete):** `dump_transpose_induce_vectors` (119) — zero
  Makefile rules; its consumer `test_transpose_induce` is *compiled*
  (Makefile:339) but in no run target, so the oracle it needs is never
  generated. Recommend wiring it into `run-tests` (restores written GPU
  coverage); deleting both is the fallback.
- **SUPERSEDED (7 benches, 1,718 lines — Phase 1):** `merkle_l0_opening`
  (revert `4e96d23` confirmed), `perm_vs_gkr` + `permutation` (loser has zero
  src callers), `blake3_rs_vs_ag` (Phase C/D answers recorded in
  ag-recursion-plan; salvage the union+AG arm first), `round1_deg4`,
  `ag_round1_ab` + `ag_lookahead_ab` (answers live in ag_skip.rs doc
  comments; deleting them unlocks the §B toggle statics).
- **DECISION-GATED (6 benches, 1,711 lines):** keccak3 ×4 (2.1);
  `merkle_vs_plain_blake3`, `sha2_merkle_proof` (2.2).
- **USEFUL-CURRENT (27):** the rest; the RS-flavored ones (`round1`,
  `round2`, `zerocheck_phases`, `e2e_zerocheck`) retire in Phase 4 (`round2`
  last — CUDA pin).

## G. Duplication clusters (Phase 3 map)

Repo-wide: 21,973 clone-covered normalized lines; **recommended consolidation
≈8,500 lines** (+780 in deliberate SIMD families, not recommended). tower.rs
alone: 23% clone-covered, ~2,970 of the recommended total.

| # | Cluster | Dup lines | Consol. | Risk |
| --- | --- | --- | --- | --- |
| 1 | tower.rs Real/Child tape+region+node family (walkers share 76%, emitters 78%) | 2,488 | ~2,000 | **high** — transcript + byte pins; unify field-by-field, re-run pins |
| 2 | LegacyPow payload-ordinal walkers ×4 (2 of 4 had the bug) | 40 | 40 | high value, mechanical: one `payload_ordinals()` iterator |
| 3 | `GateType::witness()` 12-line body ×18 + `*256` twin gates | 420 | ~330 | low (witness) / medium (`*256` column order) |
| 4 | tower native replicas (`sk_at_vks` ≡ `eval_sk_at_vks`, `frob_inv_native` ≡ jagged's) — export core fns instead | 700 | ~250 | low for replicas; do NOT merge the `check_*` validators (soundness backstop) |
| 5 | fold ↔ jagged-fold twin family ×5 | 251 | ~200 | high — transcript |
| 6 | tower gates duplicated into test files (Blake3Gate, AssistLayerGate, MultGate; circuit_wiring↔kappa7 harness 219 exact) | 371 | ~150 | low — export as `pub(crate)` |
| 7 | r1cs_hashes four-encoder family (Setup ctors byte-identical keccak↔keccak3; `accumulate_subkeccak` ≡ `fold_alpha_batched` 94 shared; keccak.rs:1234 re-implements the common driver) | 840 | ~640 | medium; leave the hot `bm_*`/`build_group_batch_major` loops |
| 8 | **SplitMix64 RNG: 117 sites, 52 variant bodies, byte-identical groups mapped** | 2,072 | ~1,950 | low but byte-pinned — one `test_rng::Rng` (~45 lines); preserve per-site draw order verbatim (`f128()` order is load-bearing) |
| 9 | Ligerito F256 transcript walk ×4 semantic copies (Rust prover/verifier + `.cuh` + host `.cpp`; `700cace` fixed 3, `ffabc07` the 4th) | ~1,500 sem. | ~180 | **maximum** — only unifiable artifact is a declarative op schedule + conformance vector; needs Blackwell CI |
| 10 | bench boilerplate (keccak3 3-way ~90% clone-covered; native-chain pair has a 134-line exact clone) | 1,270 | ~1,270 | low |
| 11 | dump-bin frame | 270 | ~270 | byte-pinned draw order |
| 12 | verifier/prover `_ag`/`_deferred`/`_timed`/`_jagged` twins | 425 | ~425 | `_timed` safe (timer sink); `_ag`/`_deferred` are protocol variants |
| 13 | merkle_r1cs `*_chunk` twins | 180 | ~180 | follows Phase 2.2 |
| 14 | merkle_glue gate-table boilerplate ×4 | 131 | ~130 | low — macro/blanket trait |
| 15 | ligerito.rs test roundtrip boilerplate | 883 | ~440 | low — `roundtrip_case` helper |

Also mapped for the Phase 3 tower.rs module split: the full region map
(41 regions with line ranges) is in the census transcript; inline test code is
~2,745 lines (13.5%) in 16 scattered `#[test]` fns + 25 cfg(test) helpers.

## H. Doc sediment

27 entries: 12 CURRENT, 7 STALE-SECTION, 6 SUPERSEDED, 2 PLAN (verified
truthful). Phase 1 actions:

- **Delete:** `.claude/RESUME.md` (tool artifact).
- **Historical header or delete** (completed work orders in gitignored
  `docs/local/`): `chain128-handoff.md` (cites the removed `TOWER_PROFILE`
  knob + deleted circuit_merkle.rs), `element-union-handoff.md`,
  `wiring-handoff.md`, `element-r1cs-handoff.md`,
  `single-dense-buffer-handoff.md`.
- **Fix stale sections:** `multi-table-design.tex` (claims the removed
  unmerged-jagged oracle is in-tree), `circuit-wiring-design.tex` (§status
  "Next" items superseded: unmerged-as-oracle, boundary-select → stratified
  queries, direct-path recursion → tower), `folding-verifier-design.tex`
  (branch refs — landed), `merkle-r1cs-notes.md` (branch header; presents the
  shift product as uncontested — link Phase 2.2), `cuda-ghash/README.md`
  (describes a 4-file field experiment; the directory is a 72-file GPU
  prover; still says "flare"), `docs/local/recursion-verifier-map.md` (§0/§5/
  §6 superseded by the tower; keep §1–§4 cost map),
  `docs/local/recursion-handoff.md` (add "HISTORICAL JOURNAL" header — still
  cited live by ag-recursion-plan Phase E).
- **Stale code doc-comments found on the way:** `TOWER_PROFILE` references in
  `tower.rs:11350` and `tower.rs:20104` (the knob is now `TOWER_CONFIG`);
  `proof_io.rs:135` links `ChainProofBundle` (never existed);
  `tower.rs:6001` `LeafOuter` doc links nonexistent `build_leaf_outer`;
  `pcs.rs:2362`'s ignore string names the wrong test filter.

## Phase 1 execution order (proposed batches)

Each batch: full suite + m6 byte pins + fold_lookahead oracles; fmt + clippy
1.98 + x86 cross-check before push.

1. **Re-pins first** (the two ROT fixtures) — so every later batch runs
   against green pins.
2. Dead pub items + pub demotions (§A) — behavior-free.
3. Concluded probes (§E) — test-code only.
4. Superseded benches + their exclusive src surface (§B: permutation.rs,
   product_gkr non-batched, deg4 trio, parallel_f128, toggle statics + unfused
   arms, blake3 A/B cluster after salvaging the union-AG arm, chunk-leaf
   remnants w/ merkle_l0_opening).
5. genus95 prototypes (keep the raw oracle) + doc sediment (§H) + the orphan
   dump bin resolution.
6. Judgment-tier sweeps (§B last paragraph), each with its own review.

## Guardrails (from the plan, restated)

New probes/benches land with an expiry note; `cargo-machete` in CI; refresh
this ledger after each phase (cheap now that the method is scripted in the
census transcripts).
