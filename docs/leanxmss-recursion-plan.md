# leanXMSS recursion: plan

Status: phases 1–4 working, 2026-11-02, branch `leanxmss-recursion`.

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
  - **shared binding** (`aggregate::Binding::Shared`): the fold binds each child's boolean assertion once (skip point, `x_inner_rest`, `z_partial`, `rr`, per-type values) instead of 30 claims of ~145 words each — the fold tape's absorbed words drop ~60%;
  - **folding per matrix class** (`matrix_classes`): the 15 slot types use 6 distinct gates, so 12 fold groups instead of 30; the lane drops from 863 to 371 words.
- **Leaf size.** 992 signatures (31·32): the statement hash's 1,054 compressions keep the compression kind in two 2^15-row slots and the leaf at 2^32 bits. 1000 signatures would push the layout planner to ν = 14 (twice the slots and gathers).
- **Measured** (M4, 10 threads, an unquiet machine): FL over 2 × 992 signatures proves in 559 ms, verifies in 8.5 ms, 254 KiB. Slot headroom is thin: BLAKE3 14,551 and MAC 15,835 of 16,384.

**4. Internal and spine nodes.** Done, without the standalone verifier yet.
- `ChainLane::family` (`LaneFamily::Batches`): the lane folds per matrix class under shared binding, at its block width, and the node statement is `hash_pair(D_left, D_right)` with the counter multiplied.
- `XmssTower::prove` runs the chain tower's spine over leanXMSS first-level nodes: the base over the last two, then each level prepends the next-earlier one. `xmss_statement` recomputes the root digest from the batches: `H(D_fl0, H(D_fl1, … H(D_fl{k-2}, D_fl{k-1})))`.
- `XmssTower::discharge_root`: the statement, the lane against the leaf tables per class, and the main fold and passenger (`discharge_outer`, shared with the chain tower).
- **Tests.** `xmss_internal_node_over_two_first_level_nodes`, `xmss_tower_e2e` (8 batches, a steady root; refuses a wrong digest, a wrong count, a moved lane claim).
- **Open.** A standalone `verify_root` from a VK, and the full-size run (8 × 992 signatures).

**5. Put the leaf optimisations back (required).** Add in-circuit support, in the order of the leaf time each saves (phase 1), for:
- the dropped words;
- pruning;
- public columns;
- packed slots.

## Open questions

1. **Tree shape.** Is it a fixed balanced tree over fixed-size batches, which the final verifier rebuilds from the list? Or may a node merge children of different sizes, which puts the shape in the statement?
2. **Leaf batch size, and whether the envelope must grow.** A leanXMSS child adds the bus GKR, 15 lincheck pins and the public words to each region.
3. **Statement digest format.** A hash tree over batch digests is the starting proposal; the alternative is a hash chain over the concatenated list.
4. **Phase 5's scope.** The whole package is worth about 5% at one thread and 7% at ten (phase 1). Are the dropped words cheap enough in-circuit to fold into phase 2 directly?
