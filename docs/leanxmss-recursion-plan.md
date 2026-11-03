# leanXMSS recursion: plan

Status: phases 1–5 working, 2026-11-02, branch `leanxmss-recursion`.

## Goal

An aggregation tree:
- Each leaf is a leanXMSS aggregate proof over a batch of signatures.
- Each node verifies two children and proves the union of their signature sets.
- The statement travels up the tree as a digest of the `(pk, msg)` list.

The first milestone is a first-level node over two leanXMSS proofs, with a round-trip and tamper tests.

The leaf will be most of the prover time, so the leaf keeps its optimisations (user, 2026-11-02). The recursion-leaf posture below is scaffolding to get a first node working. The in-circuit support for each optimisation is part of the main line, done in the order of what it saves at the leaf.

## Starting point

The chain tower (`crates/flock-prover/src/tower/`) already verifies circuit-union proofs in-circuit. A leanXMSS proof is the same proof type, `R1csProofCircuitMerged[Ag]`. It has the AG zerocheck, the union lincheck, the wiring product-GKR and the merged Ligerito opening at the 128-bit profile.

**What carries over**
- **Fiat-Shamir:** BLAKE3 duplex rows (`fs_chain.rs`).
- **PCS:** Merkle and query rows (`query.rs`), the Ligerito F256 sumcheck spine, ring-switch family H and the multipoint assist.
- **PIOPs:** the AG zerocheck point binding, the lincheck rounds and reported checks, and the wiring GKR walk and recombination (`gkr.rs`, `walker_common.rs`).
- **Structure claims:** the fold and jagged-fold regions. They defer matrix MLEs and structure tables to accumulators that are discharged at the root.
- **Upper levels:** the internal and spine nodes, which verify the tower's own envelope outers (`RealTape`, `build_node_outer_app`).

**What the tower assumes that leanXMSS breaks**

| Tower assumption | leanXMSS | Where |
|---|---|---|
| Chained BLAKE3 transcript | Tree BLAKE3 (`FsChallenger::new`) | `leanxmss/circuit.rs` prove/verify |
| One lincheck circuit (`build_block_r1cs`) | 7 gate kinds over 15 slots | `child_walker.rs:229` |
| One product GKR (the wiring) | Wiring plus the tuple bus (`flock-tuple-bus-v1`, ν + 4 layers) | `child_walker.rs:303`, `real_walker.rs:266` |
| Gathers for every gate slot | `prune_unwired` | `gkr.rs:65-87` (`pin_recombination`) |
| No public columns, no packed slots | Both on (a8561c2) | `walker_common.rs` assert |
| No zeroed degree-one words | Every degree-one word dropped (#67) | `pcs.rs` deferred open, `lincheck/union.rs` `assert_no_zeroed_words` |
| Chain statement: `h_start`, `h_end`, span counter, `ChainLane` | `(pk, msg)` list, 4,046 public words at n = 1000 | `envelope.rs`, `fl_node.rs`, `node.rs` |
| Envelope sized to the m32 chain leaf (m* = 29, ν = 14) | Larger verifier surface per child | `config.rs`, `envelope.rs` |

## Phases

**1. Recursion leaf.** Add an `XmssCircuit` option that proves in the tower's posture:
- the chained BLAKE3 transcript;
- no dropped words, no pruning, no public columns, no packed slots.

Measure what each switch costs the leaf prover against the default leanXMSS proof. The default proof stays as it is. The measurement sets the order of phase 5.

Done: `XmssOptions` (`DEFAULT`, `RECURSION_LEAF`, one switch per optimisation), and the bench knob `LEANXMSS_OPTIONS`.

Measured on 2026-11-02: M4, 1000 signatures, quiet machine, 3 alternating processes per arm, median of each process's fastest prove. Each row turns one option off, or all of them for the recursion leaf.

| Prove (ms) | 1 thread | Δ | 10 threads | Δ |
|---|---:|---:|---:|---:|
| Shipped proof (`DEFAULT`) | 1804 | | 426 | |
| Recursion leaf (all off) | 1904 | +100 | 459 | +33 |
| Chained transcript | 1823 | +19 | 429 | +3 |
| Keep degree-one words | 1880 | +76 | 444 | +18 |
| No pruning | 1886 | +82 | 453 | +27 |
| No public columns | 1842 | +38 | 439 | +13 |
| No packing | 1869 | +65 | 438 | +12 |

- **Where the recursion leaf's +100 ms goes (one thread).** The zerocheck takes +53, from the kept degree-one words. The wiring takes the other +53: 44 → 98 ms, from the three wiring options and the bit grind.
- **The wiring options overlap.** Each costs 38–82 ms alone, but only 53 together. Packing only pays once pruning and public columns leave dead slots, so they are one package.
- **The chained transcript costs almost nothing.** The +19 is the grind falling back to whole bits. Recursion leaves can keep it permanently, so the tree transcript never needs an in-circuit emitter.
- **Ranges.** Every arm's processes were within 15–40 ms of each other at one thread and 2–20 ms at ten. A grind's luck (±20 ms) is the main noise.

Phase 5's order: the dropped words first (half the cost, all in the zerocheck), then the wiring package.

**2. Child tape over a leanXMSS proof.** Done.
- **Lincheck.** `ChildTape::new` takes the child's lincheck circuits; the reported check already handled several types.
- **Full-height commit.** `MixedInner::full_height` rebuilds the union the leaf proved over.
- **Tuple bus.** `walk_bus_gkr` locates the bus region (β, the γs, the leaves product) and `emit_bus` replays it. The product GKR layers are the wiring's (`emit_gkr_layers`, shared). Each side `1 + Σ_p eq(ρ_port, p)·[(β+1)·E_p + Σ_k γ_k·(c·E_p + ŵ + T)]` is rebuilt from wires. The static columns (each port type's live indicator `E_p` and every row tag `T`) defer to a new plane of the circuit-structure table: `bus::BusColumn`, `SigmaAssertion::bus`. They fold and discharge like σ.
- **Public words: a hashed statement instead.** Hashing the 4,046 public words in every parent cost 1,140 BLAKE3 rows per child. `XmssOptions::hashed_statement` (on in `RECURSION_LEAF`) makes the signers' words private: the leaf hashes them (`statement_digest`, the `publics_digest` shape, 1 compression a signer) and publishes the 2-word digest. The leaf's public segment is 51 words.
- **Probe.** `xmss_child_region_emits_alone`: the region emits, runs (every connect holds) and checks.

**3. First-level node over two leanXMSS leaves.** Done.
- `build_first_level_core` is the first-level node over any leaf family (`FlFamily`: domain, matrices, lincheck circuits, lane width, binding) and statement (`FlApp`). The chain node is one instance; its circuit digest is unchanged.
- **Statement.** `D = hash_pair(publics_digest(leaf 0), publics_digest(leaf 1))` in app words 0..2, and the signature counter `g^n` in words 8 and 9 (the chain tower's span counter).
- **Lane width.** `EnvShape::acc_chain_words` is per tower (`envelope_shape_with_lane`); the body and registry are the chain tower's.
- **Fit.** Two 1000-signature leaves first needed 35 lanes against the pin of 31. Three changes fit two 992-signature leaves in 30.56 lanes:
  - the hashed statement above;
  - **shared binding** (`aggregate::prove_aggregate_classes_shared_with_grinding`): the fold binds each child's boolean assertion once (skip point, `x_inner_rest`, `z_partial`, `rr`, per-type values) instead of 30 claims of ~145 words each — the fold tape's absorbed words drop ~60%;
  - **folding per matrix class** (`matrix_classes`): the 15 slot types use 6 distinct gates, so 12 fold groups instead of 30; the lane drops from 863 to 371 words.
- **Leaf size.** 992 signatures (31·32): the statement hash's 1,054 compressions keep the compression kind in two 2^15-row slots and the leaf at 2^32 bits. 1000 signatures would push the layout planner to ν = 14 (twice the slots and gathers).
- **Measured** (M4, 10 threads, an unquiet machine): FL over 2 × 992 signatures proves in 559 ms, verifies in 8.5 ms, 254 KiB. Slot headroom is thin: BLAKE3 14,551 and MAC 15,835 of 16,384.

**4. Internal and spine nodes, and the root verifier.** Done.
- `ChainLane::family` (`LaneFamily::Batches`): the lane folds per matrix class under shared binding, at its block width, and the node statement is `hash_pair(D_left, D_right)` with the counter multiplied.
- `XmssTower::prove` runs the chain tower's spine over leanXMSS first-level nodes: the base over the last two, then each level prepends the next-earlier one. `xmss_statement` recomputes the root digest from the batches: `H(D_fl0, H(D_fl1, … H(D_fl{k-2}, D_fl{k-1})))`.
- `XmssTower::discharge_root`: the statement, the lane against the leaf tables per class, and the main fold and passenger (`discharge_outer`, shared with the chain tower).
- **Tests.** `xmss_internal_node_over_two_first_level_nodes`, `xmss_tower_e2e` (8 batches, a steady root; refuses a wrong digest, a wrong count, a moved lane claim).
- **Root verifier.** `XmssTowerVk::generate` builds the VK from a reference tower over six batches. `verify_xmss_root(vk, statement, bundle)` checks the root proof against the VK alone. It then reassembles the accumulator from the root's publics, checks the statement, and runs the discharge. `xmss_tower_e2e` also refuses swapped batches, an edited message and a forged public.
- **Full size.** 8 × 992 = 7,936 signatures, with the final leaf posture (phase 5): the end-to-end test passes, including the standalone verify. The root proof is 254 KiB. Per-stage timings are in the measured section below.

**5. Put the leaf optimisations back.** Done: `RECURSION_LEAF` now has every optimisation of the default proof, plus the chained transcript and the hashed statement.
- **Dropped degree-one words.**
  - **Lincheck report.** A dropping type reports its `(A, B)` values over its kept rows. When η checks the dropped rows, the type also reports their identity value `⟨W, D(A_0 + I)⟩` as an extra `matrix_evals` pair after the `T` pairs. A union that drops nothing keeps its proof bytes, so the m6 and element byte pins hold.
  - **Folding.** `MatrixAssertion::linear_claims`; a linear-identity fold group per dropping class (`Accumulator::per_linear`); `aggregate::Dropped` and `SharedTables`. Matrix classes key on each type's dropped words.
  - **In-circuit.** The tape locates η, and the reported check adds `α·η·v_L`. The union's raised AG round-one grinding applies.
  - **The C claim** (`C = diag(mask)`) exports as `JaggedRowWeight::Masked` on the same layout table. It costs one prefix chain per masked column in the jagged fold: 76 columns in 13 runs at 992 signatures.
- **The wiring package (pruning, public columns, packing).**
  - **Recombination.** It reads each gathered slot and public slot at its grand-product position (`recombination_positions`). The circuit-structure table already indexes σ and the live mask in packed order.
  - **Public columns.** Under the hashed statement they read only fixed public words. So their values are static functions of the row point, and defer to plane 7 of the circuit-structure table (`SigmaAssertion::public_columns`).
- **σ binding.** The public columns added 19 structure claims per child, which pushed the first-level node to 31.11 lanes. Shared binding now absorbs each σ assertion's surface once (`SigmaAssertion::shared_surface`). The claims' constant coordinates bake into the in-circuit fold (`ClaimLoc::consts`), and the fold tape's absorbed words drop from 5,781 to 2,729.
- **Fit at 992 signatures per leaf** (first-level node):

  | Leaf posture | Lanes (pin 31) | MAC slot (of 16,384) |
  |---|---:|---:|
  | Dropped words, no wiring package | 30.97 | 16,073 |
  | Full posture | 30.47 | 16,099 |

  The MAC slot is the binding constraint now.

## Measured (2026-11-02, quiet M4, bench profile)

**The leaf.** 992 signatures; the median of each process's fastest prove, 4 alternating processes per arm, in ms. The recursion leaf costs nothing measurable. Phase 1's binary gives the same default, so the dropped-words lincheck report left the shipped proof's time alone.

| Arm | 1 thread | 10 threads |
|---|---:|---:|
| `DEFAULT` | 1,870 | 431 |
| `RECURSION_LEAF` | 1,853 | 430 |
| default + chained transcript | 1,900 | 434 |
| default + hashed statement | 1,888 | 437 |
| phase 1's binary, default | 1,871 | 434 |

**The tower** (`xmss_tower_online_bench`). Online ms per proof: the median of 3 steady iterations after a warmup. The leaf here is witness + prove inside the test binary.

| Stage | 1 thread | 4 threads | 10 threads |
|---|---:|---:|---:|
| Leaf (992 signatures) | 2,054 | 604 | 471 |
| First-level node | 780 | 295 | 270 |
| Base | 693 | 250 | 208 |
| Spine node | 712 | 250 | 216 |
| Per leaf: leaf + FL/2 + node/2 | 2,801 (+36%) | 877 (+45%) | 714 (+52%) |
| Per signature | 2.82 | 0.88 | 0.72 |

The nodes scale worse with threads than the leaf (3.3× from 1 to 10 threads, against the leaf's 4.4×), so the overhead grows with the thread count.

**Where a node's time goes** (spine node, 1 thread, ~700 ms; `PCS_TRACE`):

| Phase | Node ms | Notes |
|---|---:|---|
| Witness generation | 62 | Slot generation; the buffers come pre-zeroed from the zero pool. The swap gate's per-bit bool path took 28.5 ms, and its packed rewrite (a59e952) brought witgen to 32 ms. |
| Commit | 192 | A plain F128 NTT (126) and Merkle (51). The nodes commit at rate 1/4 (Slim, half the queries for the parent) and the leaf at 1/2 (Fast), so per input word they encode and hash twice as much; per codeword word the NTT costs the same. |
| Zerocheck + lincheck + wiring | ~150 | |
| Element PIOP | 13 | |
| Open | ~270 | Ligerito 112; W build + merged sumcheck 103; multipoint 97. |

Per committed word, a node costs 2.8× a leaf. The leaf's bit-structured witness takes the fast paths: the ranked NTT, block statistics, and closed-form multipoint weights over 703 full columns (3 ms). The node's ragged layout (730 columns, 11 full) takes the dense virtual weight: 74 ms of weight passes and two-product sumcheck. That is the next place to look. After the swap fix, witgen's remaining 32 ms is BLAKE3 rows (12), the family-transpose tiles (7, still on the bool path), the element slots (7) and the bit spread (3).

## Node optimization pass (2026-11-02)

Goal: every per-byte difference between a node and the leaf is either removed or explained. Measured at 1 thread on a quiet M4 with the bench profile (`xmss_tower_online_bench`). The fingerprint PoW enters at its expectation (live leaves × 22.4 ns/hash), since its sampled time is a random draw, fixed per transcript (8–91 ms observed).

**Removed:**
- **Glue witness** (aa13455, cfadf18). The swap, bit-spread, PoW-mask and family-tile tables write packed words through the shared batch-major driver instead of `Vec<bool>` rows. Node witgen went 60 → 21 ms.
- **Lane fill** (2ce1fd4). The rate-1/4 lane-major fill transposes once and copies the replicas: 14 → 7.7 ms, the leaf's rate per encoded byte. Fusing the fill into the first NTT pass lost: 32–38 ms against 24 ms for the memcpy replicate plus the in-place pass.
- **Uneven-column multipoint** (dea13dd). Envelope counts are aligned to 2^8 with real padding gates, and the closed form now runs over 2^g-row sub-columns, prover-only.
  - Multipoint weights went 37 → 5 ms and its sumcheck 37 → 0.7 ms.
  - The first-level node grows 30.47 → 30.96 lanes.

| Per-proof ms | 1 thread | 4 threads | 10 threads |
|---|---:|---:|---:|
| Spine node before → after | 712 → 580 (expected PoW) | 250 → 222 | 216 → 184 |
| First-level node before → after | 780 → 629 | 295 → 226 | 270 → 193 |
| Per-leaf overhead before → after | +36% → ~+31% | +45% → +38% | +52% → +41% |

**Explained (spine ÷ leaf, ns per live byte):**
- **Commit 2.0×.** The rate is 1/4 (Slim) against 1/2. Per encoded byte: NTT 0.8×, fill 1.0×, Merkle 1.1×. The Merkle excess is 31-lane rows of 496 B rounding up to 64-byte blocks, and a pinned lane count that is 15% zero.
- **Ligerito 2.2×.** The rate again: 0.94× per encoded byte.
- **Zerocheck 0.9×.**
- **Lincheck 1.8×.** A fixed comb build per distinct gate (10 ms node, 19 ms leaf) plus count-proportional folds at 1.4×.
- **Wiring 6.5× and fingerprint PoW 5.5×.** Cell density: the node has 2^23 cells for 57 MB, because nearly every node word is a gate input or output; the leaf has 2^22 cells for 369 MB. Per cell the node is cheaper. Pruning and packing do not shrink the node's μ.
- **Element PIOP (13 ms).** Node-only full-field gates.
- **Multipoint values (22 ms).** A fixed cost in both leaf and node: 2 claims × 128 conjugates × columns × layers.
- **Merged path 3.0×.**
  - The leaf takes the block-first transport. It needs per-column bit-banks, which the leaf gets from its boolean zerocheck (the 71 ms C-claim fold); counted together with that fold, the leaf is 0.53 ns/B and the node is 1.8×.
  - For the node's full-field words a bit-bank costs about what the W build does.
  - Open: a node block-first path at g = 8 with byte-table banks could save ~25 ms, but it changes the transcript order. That means a verifier rotation and the parent's in-circuit RealTape.

## Open questions

1. **Tree shape.** The driver is the chain tower's spine: a base over the last two first-level nodes, then each level prepends the next-earlier one. The statement is `H(D_fl0, H(D_fl1, … H(D_fl{k-2}, D_fl{k-1})))` over fixed 992-signature batches, and the verifier rebuilds it from the batches (`xmss_statement`). A balanced tree would need a second internal shape.
2. **Leaf batch size.** It is 992. One more signature tips the statement hash's compressions into a third slot, and the layout planner to ν = 14.
3. **Margin.** The first-level node runs at 30.47 of 31 lanes, and its MAC slot at 98%. The next diet: merge the lagrange-low evaluations of claims that share a low (the boolean claims of one child still evaluate the same 64-word low per claim in-circuit).
4. **Node cost.** A node is 35–46% of a leaf's time for an eighth of its words (above). The candidates:
   - the ragged multipoint weight: aligned envelope columns, or a closed-form ragged weight;
   - the family tiles' bool-path witness;
   - the Slim-vs-Fast trade for the node commit, which is in-circuit query cost against prover time.
