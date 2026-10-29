# The tuple bus: content-addressed dataflow beside the wiring

Status: implemented on branch `leanxmss` (`flock-core/src/circuit/bus.rs`,
`flock-prover/src/leanxmss/circuit.rs`). Needs a soundness review before it
goes to `main`.

## Summary

Flock's wiring argument proves copy constraints along a permutation σ that is
fixed when the circuit is built (`docs/circuit-wiring-design.tex`). Some
circuits need dataflow whose *route* depends on the witness. leanXMSS is the
example: which row finishes Winternitz chain `c` depends on the message
digest's digits. With σ only, the circuit has to carry that routing in its
rows. The chain machine did this: a 122-word row around a 92-word BLAKE3
compression. About 28 of those words were a leaf-block buffer, its update
selects and a digit shifter.

The tuple bus adds one more grand product to the wiring proof. Rows **send**
and **receive** tuples. The argument proves that the multiset of sent tuples
equals the multiset of received ones. A tuple coordinate is a committed word
of the row plus a constant plus a static per-row tag. So a value can travel to
"whichever row receives `(layer, chain, position)`" instead of to a fixed cell.

For leanXMSS this removes the machine. Every one of the 201 compressions per
signature is a plain 92-word compression row, the same width as the
hand-written BLAKE3 gate. The chain steps exchange `(layer, chain, address,
value)` over the bus.

## 1. The argument

### 1.1 Objects

- A **port** `p` is one registry type `ty(p)` together with a direction
  `dir(p) ∈ {Send, Recv}` and `w` **terms**. Every live row `j < count(ty)` of
  that type contributes exactly one tuple. A port always covers whole slots,
  which keeps every claim a plain column evaluation (§1.3).
- A **term** is `word + constant + row_tag`, each part optional:
  - `word` is one of the type's gate cell-slots, the same wireable words the
    wiring uses;
  - `constant ∈ F`;
  - `row_tag(j) = ((offset + j) / div) mod modulo`, read as the field element
    with that bit pattern.
- The **bus** is the list of ports. All ports share one width `w`. The bus is
  part of the circuit: `Circuit::digest` appends its encoding after the wiring,
  so circuits without a bus keep their digests.

The statement the bus adds is the multiset equality

    { t(p, j) : dir(p) = Send, j < count(ty(p)) }  =  { t(p, j) : dir(p) = Recv, j < count(ty(p)) }

where `t(p, j)_k = word_k(j) + constant_k + row_tag_k(j)`.

### 1.2 Protocol

The bus runs inside `prove_wiring` / `verify_wiring`, right after the wiring's
grand product, on the same transcript. The wiring's gather has no transcript
operations, so the bus is the next thing absorbed.

1. Absorb the label `flock-tuple-bus-v1`. Sample `β` (with the same PoW
   grinding as the wiring's fingerprint challenge `α`, sized to the number of
   live bus entries), then `γ_1, …, γ_w`.
2. Leaves:
   - Each port gets one group of `2^ν` leaves.
   - Send ports fill the left product's groups and receive ports the right
     product's. The side with fewer ports is padded with empty groups up to
     `2^c`, where `c = ⌈log₂ max(#send, #recv)⌉`.
   - Row `j < count` of port `p` holds `φ(p, j) = β + Σ_k γ_k · t(p, j)_k`.
     Every other leaf is `1`.
3. `product_gkr::prove_leaves` proves `∏ L = ∏ R`. It is the wiring GKR's
   layer pipeline (grouped live prefixes over implicit ones, one λ-batched
   sumcheck per layer), now fed caller-built leaves. It ends at one point
   `ρ = (ρ_row, ρ_port) ∈ F^{ν+c}` with the claims `L̂(ρ)` and `R̂(ρ)`.
4. The prover sends one **gather** value per distinct cell-slot `ι` the ports
   read: `g_ι = Σ_{j<count} eq(ρ_row, j) · w_ι(j)`. It is the same row fold as
   the wiring's gather and becomes the same packed-direct claim, at the point
   `[ρ_row ‖ bits(word_col) ‖ bits(slot_prefix)]`. The claims are appended
   after the wiring's gather claims and are discharged by the one merged
   opening.
5. The verifier rebuilds each side's leaf MLE at `ρ`:

       L̂(ρ) = 1 + Σ_{p ∈ Send} eq(ρ_port, p) · [ (β + 1)·E_p
                 + Σ_k γ_k · ( g_{ι_k} + c_k·E_p + T_{p,k}(ρ_row) ) ]

   Here `E_p = Σ_{j<count} eq(ρ_row, j)` is the existing `eq_prefix_sum`, and
   `T` is the row tag's sum over the live rows. The verifier checks both sides
   against the GKR's claims. (Characteristic 2: `−1 = +1`.)

The proof adds `WiringProof::bus: Option<BusProof>`. A `BusProof` is the
GKR transcript, the gather values and at most one PoW nonce. A proof whose
`bus` presence does not match the circuit's is malformed.

### 1.3 Soundness

- **Multiset equality from the products.** The polynomials
  `Φ(t) = B + Σ_k G_k t_k` in the formal variables `(B, G_1, …, G_w)` are
  distinct, irreducible and of degree 1 for distinct tuples `t`. Unique
  factorization then gives `∏_send Φ = ∏_recv Φ` as polynomials if and only if
  the multisets agree. The verifier samples `(β, γ)` after the commitment and
  statement are bound, so Schwartz–Zippel bounds a false equality by
  `N / |F|`, with `N` the number of live entries. There is one challenge per
  coordinate, so no power-of-γ degree blow-up. The `1`s on dead leaves are
  multiplicative identities, not factors.
- **The GKR.** This is the wiring's reduction and the same layer-by-layer
  sumcheck, now as `verify_reduction`, shared by both entry points. The only
  change is who builds and checks the leaves.
- **Binding the leaves to the commitment.** `L̂(ρ)` is affine in the gather
  values once `β`, `γ` and the static parts are fixed. Each gather value is a
  packed-direct claim on a committed column at a point with Boolean high
  coordinates, which the merged opening checks. This is the same contract as
  the wiring's `f`-side gather (`circuit.rs`, "What the caller owes", item 4).
  The prover's fold reads only the live prefix, so it relies on dead rows being
  zero in the committed buffer, exactly as the wiring does.
- **Both sides bound.** Each side's claim is checked against its own
  reconstruction, so neither product's leaves are prover-chosen. Compare the
  wiring's `f_eval == g_eval` requirement.
- **Static parts.** Constants and row tags are digest-bound and evaluated by
  the verifier itself. A row tag's sum costs `O(count)` per port, the same
  order as the σ-aware wiring verifier.

The bus does **not** know or check which side "produced" a value. As with
wiring classes, semantics come from the circuit around it (§2.3).

### 1.4 Cost

- **Prover:** one leaf per live port row (a `w`-term linear combination), a
  grand product over `2^{ν+c}` grouped leaves (only live prefixes are
  touched), and one row fold per distinct word.
- **Proof:** `O((ν+c)²)` field elements, plus one value per word.
- **Verifier:**
  - `O((ν+c)²)` for the GKR;
  - `O(2^ν)` for one `eq` table;
  - `O(count)` per row-tag term.

### 1.5 Limits

- **Whole slots.** A port covers every live row of a slot. Rows with different
  bus roles need different slots. leanXMSS puts its chain steps and its other
  compressions in separate slots of the same gate.
- **Recursion.** The tower's in-circuit verifier (`child_walker`) replays the
  wiring GKR only. A circuit with a bus cannot be a recursion child until that
  replay learns the bus. Nothing in the tower uses one.
- **Multiplicity one.** Each tuple is sent once and received once. A value
  needed by several receivers must be sent several times, or fanned out by
  ordinary wiring after one receive, which is what leanXMSS does for chain
  ends.

### 1.6 Pruning unwired cells

A bus circuit has whole io words that no wire touches. On leanXMSS these are
the step rows' `m1, m2, m3, o0, o1, next`, the chain rows' `x` and `s`, and
`next` on the other compressions. Each of their cells is σ-fixed, so it
contributes the identical factor `w + α·x + β` to both wiring products.
`Circuit::with_unwired_pruned` (`ShapeBuilder::prune_unwired`) marks every
gate cell-slot with no wired cell dead in the wiring's live mask:
- Its leaves become 1 on both sides.
- Its gather value and claim disappear.
- The verifier skips it in the recombination.
- The digest records the pruned set.

The products are unchanged, so soundness is the wiring's, restricted to the
cells it still covers. The bus reads its words through its own gathers.
Pruning is opt-in, so recursion circuits keep their transcripts.

## 2. leanXMSS on the bus

### 2.1 Rows

Per signature (two layers):

| Kind | Rows | Width | Bus |
|---|---:|---|---|
| compression, chain steps | 136 | 92 words | recv + send |
| compression, digests, leaves, nodes | 65 | 92 words | none (wired) |
| chain row | 116 | 6 words | send + recv |
| swap, codeword, address, address-step | 72 | 3–13 words | none |

The compression row (`leanxmss/compress.rs`) wires in `cv0, cv1, m0..m3` and a
control word (flags and block length). It outputs `o0, o1` (the output
chaining value) and `next`: `m1` with its chain step field advanced by one,
mod 4. The rounds are padded to end on a word boundary, so the output words
follow them contiguously and the hand-written BLAKE3 generator still writes
the whole operation.

### 2.2 Tuples

A tuple is `(S, c, A, v)`:

- `S`: the layer. A row tag, `⌊k / 68⌋` on step rows and `⌊k / 58⌋` on chain
  rows, where `k` counts the kind's rows across its slots. Rows are emitted
  signature by signature, layer by layer, so `S = 2·signer + layer` on both.
- `c`: the chain index. The block word `m3 = LE128(c)` on a step row. The row
  tag `k mod 58` on a chain row.
- `A`: the chain address `A(µ) = A(0) ⊕ µ·2^61`. Its step field holds the
  position. `m1` (receive) and `next` (send) on a step row. The chain row's
  `s = A(d_c)` (send) and `a ⊕ 3·2^61` (receive).
- `v`: the value. `m2` (receive) and `o0 = Th(…)` (send) on a step row. The
  chain row's `x` (the revealed value, send) and `e` (the end, receive).

The ports:

| Rows | Sends | Receives |
|---|---|---|
| step | `(S, m3, next, o0)` | `(S, m3, m1, m2)` |
| chain `c` | `(S, c, A(d_c), x_c)` | `(S, c, A(3), e_c)` |

Around the bus, fixed wiring does the rest:
- A step row's `m0` is wired to its signer's `P`, its `cv` to the IV, and its
  control word to "one 64-byte block".
- Chain row `c` takes the digest shifted by `2c` (a shift chain through the
  chain rows) and the layer's `A(0)` from the address gadget.
- Its end `e_c` is also wired into the leaf's block words. The leaf's 15
  compressions, the Merkle path and the digests are wired as before.
- The codeword gadget checks the digest's digits (top digits zero, sum 106).

### 2.3 Why a balanced bus means the chains were walked

Fix a layer `S` with digits `d_0..d_57` and `Σ d_c = 106`, which the codeword
gadget checks. Its tuples come only from its own 68 step rows and 58 chain
rows, because `S` is static per row. A matching between equal sent and
received tuples exists. Follow it from chain `c`'s start `(S, c, A(d_c), x_c)`:

- **Paths stay on one chain.** Each received tuple belongs to a step row,
  which sends exactly one tuple, or to a chain row's end. A step sends its own
  `m3`, so the chain index is preserved, and the path ends at chain `c`'s end
  receive `(S, c, A(3), e_c)`. Step rows not on any path form cycles.
- **Path lengths.** Each step advances the step field by one, mod 4. A path
  from `d_c` to 3 therefore takes `3 − d_c + 4k_c` steps, and a cycle takes a
  positive multiple of 4.
- **Counting forces the honest walk.** The paths and cycles use exactly the
  68 rows, and `Σ_c (3 − d_c) = 3·58 − 106 = 68`. So every `k_c = 0` and
  there are no cycles. Chain `c` is walked exactly from `d_c` to 3.
- **Each step is the right hash.** The step leaving position `µ` hashes
  `P_S ‖ A(µ) ‖ v ‖ LE128(c)`:
  - `P_S` is fixed by wiring;
  - `A(µ)`'s high part is the address gadget's `A(0)`, which steps never
    change;
  - its step field is `µ`;
  - `LE128(c)` is `m3`.
  So `e_c` is the honest chain end, and the leaf, wired to the `e_c`, is the
  honest leaf.

Two points the static `S` handles:
- **Without `S`,** step rows could serve another signature's chains. A prover
  with its own key could then sign a codeword with digit sum 105 (69 steps)
  next to a forgery for someone else's key with sum 107 (67 steps). The global
  balance would hold, and the 107 codeword dominates an honest one.
  `P` does not help: it is public, so the attacker's key can reuse a victim's
  `P`.
- **Junk high bits in `m3`** would not merge two layers either: `S` is a
  separate coordinate, not packed into the chain index.

### 2.4 What changed

- The machine (`machine.rs`), its 7-word state, digit shifter and leaf buffer
  are gone.
- Every compression row is 92 words (the machine row was 122). The new chain
  rows cost 6 words each, 116 per signature.
- The circuit's statement is unchanged: per signer `P`, root and message
  public, everything else private.

## 3. Results

1000 signatures, M4 (4P + 6E), `cargo bench -p flock-prover --bench leanxmss`.
These are alternating A/B runs against the chain machine at `e09103b`, two
rounds each, with the bus circuit's unwired cells pruned (§1.6). Medians are
of warm proves; "best" is best online+prove.

| Pool | Machine median | Bus median | Machine best | Bus best |
|---|---:|---:|---:|---:|
| rayon, 10 threads | 686 ms | 562 ms | 674 ms | 564 ms (1773 sig/s) |
| 4 P-cores + hops + SME (`LEANXMSS_POOL=perf`) | 761 ms | 596 ms | 728 ms | 610 ms |

Phase medians, ms. Pure BLAKE3 is `blake3_proof` at 201,000 compressions,
full height, from the earlier ten-thread comparison.

| Phase | Machine, 10T | Bus, 10T | Pure BLAKE3, 10T | Machine, perf | Bus, perf |
|---|---:|---:|---:|---:|---:|
| witness | 70 | 48 | 26 | 68 | 47 |
| commit | 170 | 130 | 139 | 168 | 130 |
| zerocheck (∥ wiring GKR) | 220 | 161 | 134 | 256 | 156 |
| wiring GKR | 159 | 111 | — | 206 | 108 |
| bus | — | 9 | — | — | 10 |
| lincheck | 55 | 56 | 26 | 73 | 72 |
| open | 130 | 124 | 102 | 163 | 157 |
| total | 686 | 562 | 462 | 761 | 596 |

On 10 threads, leanXMSS now costs 1.22× pure BLAKE3 at equal compressions,
down from about 1.45× with the machine. On the P-core pool it costs 1.11×
(537 ms pure).
- **Pruning alone** took the wiring GKR from about 136 to 108 ms.
- **The bus itself** costs 9–10 ms.
- **Commit** is level with pure hashing.

The rest of the gap:
- the wiring GKR beside the zerocheck;
- per-slot costs in lincheck and the opening (15 slots against one);
- the packing of the non-hash words.

`ν = 16` (10 slots) is a wash: lincheck 54 → 33 ms but commit 128 → 147 ms.
Shape setup is 4.9 s against 4.1 s.

### After the packer and lincheck fixes

Two per-slot costs went next (`29b795a`, `9918098`):
- **Witness packing.** The gate packer copies wired input words straight
  from the rows, builds the BLAKE3 kernel's arguments from them, drops the
  inputs' identity rows from the walk, and runs the kernel and the stripe
  eight rows at a time. On one 2^15-row compression slot it is 1.1× the
  hand-written generator (was 1.57× on 10 threads).
- **Lincheck.** Each distinct gate's matrices fold once (7 of the 15 slots
  share the compression gate), the slots' stripe folds run together, and
  narrow full slots fold by rows.

Alternating A/B on a quiet machine, 10 threads, three rounds each, median
of each round's fastest prove (ms):

| Phase | Pure BLAKE3 | Bus |
|---|---:|---:|
| witness | 27 | 36 |
| compact q | 8 | 9 |
| commit | 139 | 133 |
| zerocheck (∥ wiring GKR 110, bus 9) | 135 | 158 |
| lincheck | 28 | 28 |
| bit-bank fold | 24 | 24 |
| open | 103 | 121 |
| total | 461 | 512 |

The rounds agreed to within 4 ms (pure 459–473, bus 509–512), so the bus
costs 1.11× pure BLAKE3. Best online+prove is 523 ms, about 1,910
signatures/s. What is left is the witness (+9), the zerocheck sharing its
cores with the wiring GKR (+23), and the opening (+18), whose block
statistics (20 against 2) and merged sumcheck (27 against 11) grow with the
committed columns (703 against 92).

## 4. Questions for review

1. Is the transcript position right? The bus samples `β, γ` after the wiring
   GKR on the wiring's (possibly forked) transcript, and its gather values are
   absorbed only by the opening, as the wiring's are.
2. Per-coordinate challenges vs. powers of one `γ`: per-coordinate is used
   (simpler degree bound, `w` extra samples).
3. Grinding: `β` is ground like the wiring's `α` (sized to the live entry
   count); the layer challenges reuse the wiring's per-layer policy.
4. The counting argument in §2.3 is what replaces a range check on the step
   field. It needs the codeword gadget's digit-sum check and the exact row
   count per layer. Both are fixed by the circuit, but they are now
   load-bearing for the bus.
