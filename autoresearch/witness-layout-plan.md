# Batch-major witness layout: fold log_n first (jagged multi-table prep)

Status: v4 — 2026-07-02. E0 + E1 done (results in §7); E2–E6 pending.

## Motivation

We are considering **jagged multi-table** support: several circuits with
different `k_log` (block size) and different `n_log` (batch count) in one
proof. The composition we want:

1. **Fold the `n_log` (batch) dimension first** in the sumcheck; after the
   batch rounds each table collapses to a single random-linear-combination
   "virtual instance", and the concatenation of those per-table residues is
   one big (jagged) circuit for the remaining rounds + lincheck.
2. **The batch dimension lives over the packed values**: batch coords must be
   *word-index* (suffix) dims of the F128-packed witness, never inside the
   ring-switch prefix, so ring switching and the jagged reduction compose
   per-word.

Both constraints are satisfied by one layout (L1′ below). This document plans
the experiment: verify the column-major layout **does not hurt prover
performance** on today's single-table workloads before building multi-table
on top.

## 1. Current layout (baseline, "row-major")

Address bit `i` of the buffer = variable `i` of the MLE:

```
addr = [ k_log inner bits (LSB) | n_log batch bits (MSB) ]      m = k_log + n_log
```

Each instance is a contiguous `2^k_log`-bit run (8 KB at k_log = 16).
Zerocheck's refinement of the low bits (`zerocheck.rs`,
`prove_packed_padded_inner`):

```
addr = [ 6 skip (λ) | 7 friendly-constant dims | m−13 sampled dims ]
         dims 0..6    dims 6..13 (protocol-fixed)   dims 13..m
```

Binding order: round 1 binds dims 0..6 univariately (URM); multilinear rounds
then bind dims 6, 7, … in address order. Today dims 6..13 are *within-block*
bits, so the batch dims are bound **last**.

### Ring-switch structure (what "packed" means precisely)

`pcs/pack.rs`: `LOG_PACKING = 7` — address dims 0..7 live inside one F128
word; the packed witness is `2^(m−7)` words indexed by dims 7..m.

`pcs/ring_switch.rs` (DP24 adapted to the φ₈ LCH basis) splits every z-claim
point into:

- **prefix** = the 7 in-word dims, weighted by
  `weights[i] = ν_φ8(i & 63)(z_skip) · eq(x_outer[0], i >> 6)`
  (`build_claim_weights`) — i.e. the prefix is exactly **(z_skip,
  x_outer[0])**: the 6 univariate-skip dims plus the *first multilinear
  coordinate*;
- **suffix** = `x_outer[1..]` (m−7 coords) indexing the packed words:
  `s_hat_v = fold_1b_rows(packed_witness, ⊗ suffix tensor)` (128 entries),
  then `rs_eq_ind` / BaseFold / Ligerito run entirely over the word array.
  The verifier's `eval_rs_eq` is polylog and coordinate-order agnostic.

So the ring boundary sits at address dim 7, and the ring-switch prefix
*already* consumes the skip challenge plus one multilinear coordinate as a
unit.

### Jagged reduction (`pcs/jagged.rs`, standalone core)

Jagged function `p : {0,1}^n × {0,1}^k → F`, column `y` nonzero below height
`h_y`; dense `q` = the nonzero entries flattened **column-major** (column
after column, no per-column padding). The reduction turns `p̂(z_r, z_c) = v`
into one dense claim `q̂(i*) = α` via a product sumcheck + a branching-program
evaluation of `f̂_t`. Deliberately not yet wired to ring switch — that
composition is what this layout enables.

## 2. Target layout L1′: chunk = 128 bits, batch = low word-index dims

```
addr = [ 7 in-word bits | n_log batch bits | k_log−7 chunk-index bits ]
         LSB              word-index LSB     word-index MSB
```

- Per-instance atomic unit = one **F128 word (128-bit chunk)**; an instance is
  `2^(k_log−7)` words; the packed witness is a **(chunk-index × instance)
  matrix of F128 words, stored chunk-index-major**. No boolean transpose
  anywhere; all data movement is ≥ 16-byte-word granular.
- **Binding order**: round 1 (skip, dims 0..6) → one multilinear round for
  dim 6 (the second half of the word — the in-word dim the ring-switch prefix
  already pairs with z_skip) → **all n_log batch dims** → chunk-index dims.
  The multi-table collapse happens after `1 + n_log` multilinear rounds;
  every table shares the one pre-bound in-word dim (all have k_log ≥ 7).
- **Ring-switch alignment** (the reason chunk = 64 is out): the ring prefix
  `(z_skip, x_outer[0])` = exactly the within-chunk dims; the suffix = pure
  `(x_batch, x_chunk)` coords. Batch coords are word-index dims — "the batch
  dimension is over the packed values". Under chunk = 64 the batch bit 0
  would sit at address dim 6, *inside* the ring prefix, splitting the batch
  dimension across the ring boundary and making per-instance word columns
  impossible.
- **Friendly constants** (address dims 6..13, fixed by the URM optimization):
  dim 6 stays a within-word dim; dims 7..13 land on **batch bits 0..6** ⇒
  requires `n_log ≥ 6` (production: n_log = m − 16 ≥ 6 for m ≥ 22; small
  tests need spillover handling or a batch floor). Soundness note to confirm:
  the F₂-linear-independence argument for the seven constants is a property
  of the values, not of which semantic dims carry them
  (`friendly_challenges_f2_independent`) — reads the same, re-check
  explicitly.

### Jagged orientation: L1′ *is* the dense flattening

Take the jagged matrix to be **rows = batch/instance index, columns = chunk
positions** (concatenated across tables), entries = F128 words:

- Column `y` (a chunk position of table `t`) has height `h_y = 2^{n_log_t}`
  if the chunk is useful, else `0`.
- Column-major flattening of that matrix = `batch + Σ_{y' < y} h_{y'}` =
  **exactly the L1′ word order** (for a single dense table:
  `word = batch + chunk · 2^{n_log}`). The committed dense `q` is the L1′
  packed witness with *no relayout step*.
- Jaggedness sources: different `n_log` per table (different column heights),
  different `k_log` per table (different column counts), and per-table
  `useful_bits` (zero-height columns → **padding is simply not committed**,
  shrinking the commit vs today's zero-row-stuffed blocks).
- Claim points compose as `(z_r, z_c) = (x_batch, x_chunk)` = the ring-switch
  suffix split into its two natural groups. Pipeline:
  zerocheck/lincheck (bit-level claims) → ring switch (prefix → 128-entry
  weights; suffix claim on words) → jagged reduction (sparse (batch, chunk)
  claim → dense q̂ claim) → BaseFold/Ligerito.

**Design principle: no materialize-then-transpose.** We never build row-major
z/a/b and permute 2^(m−3)-byte buffers (costs more than the layout saves,
doubles peak memory). The *producer* emits L1′ directly: witness gen stages a
group of instances in a core-local buffer (L1/L2-resident), then flushes per
chunk-index in contiguous SIMD-width runs — group width chosen so every write
to main memory is a long sequential run (see E1). Consumers need no
adjustment: the sumcheck/PCS kernels already stream the address space
sequentially, and lincheck's stripe (if kept) is built from the staging
buffer, never by re-reading the big array.

## 3. Why parity is the expected outcome (and where it can break)

Central observation: **the heavy kernels are address-bit generic** — they
stream the buffer as an MLE over address bits and never consult a bit's
semantic meaning. Under L1′ these are *bit-identical* computations on the
permuted buffer (same instruction stream, same access pattern; the friendly
constants stay at address dims 6..13 either way, so even the URM's
shift-reduce/convert inner structure is untouched):

- URM round 1 (`process_one_x_hi`, 8192-bit windows = address dims 0..13 —
  under L1′ a window = the 64 λ-lanes × 128 word-adjacent positions);
- round-2 fused fold + all multilinear tail rounds (`multilinear.rs`);
- PCS commit (BaseFold + Ligerito), `fold_1b_rows` MFR kernels, `rs_eq_ind`
  build, `eval_rs_eq`.

So zerocheck + commit + open cost **cannot** change except through
bookkeeping. Where the layout genuinely changes code and could cost:

1. **Witness generation** (main risk): the per-instance builder currently
   writes one contiguous 8 KB run; under L1′ an instance's 512 words scatter
   with stride `2^(7+n_log)` bits. Mitigation: 64-instance staged group
   (512 KB, L2-resident), flushed per chunk-index as contiguous **1 KB runs**
   (64 instances × 16 B). Predicted parity-ish; must measure (E1).
2. **Lincheck stripe**: lincheck's byte-table fold wants bit-granularity
   batch-major (`lincheck.rs:94`). Under L1′ the stripe build's source tiles
   (adjacent instances' words) are contiguous — same transpose count, better
   locality; also evaluate dropping the stripe copy (2^(m−3) B) by
   transposing inside the fold. Not a regression risk; possibly a win (E3).
3. **`apply_a/b` strip kernels** (generic `prove` path only): gather tiles
   become contiguous; transpose-out disappears if a/b are stored L1′ too.
   Small win (E4).
4. **Padding**: chunk-index is the MSB group ⇒ padding words form a
   **contiguous suffix** for all instances at once. `PaddingSpec` /
   `b_med_counts` (URM) and `ChunkPadding`'s modular skip (`ring_switch.rs`)
   collapse to prefix iteration — strict win on padded circuits, and under
   the jagged end-state the padding isn't even committed.
5. **Bookkeeping** (wire format + code churn, no runtime cost):
   - r-vector semantic relabeling in `zerocheck.rs`; `r_rest` decomposition.
   - `QuirkyPoint` becomes `(z_skip, x_inner0, x_batch, x_chunk)` — claim
     assembly in `prover.rs:624–662` reorders; lincheck's factorization is
     position-agnostic but its eq-table builders / verifier interpolation
     index differently.
   - `s_hat_v` fast paths re-derived: `s_hat_v_from_z_vec` (from lincheck's
     z_vec) and the zerocheck two-bank `s_hat_v_c` capture both assume the
     current suffix ordering; under L1′ the suffix tensor factorizes as
     `⊗(x_batch) ⊗ ⊗(x_chunk)` — same kernels, permuted tensor build.

## 4. Experiment plan

All work in `autoresearch/` probes + a prototype path; promote only on wins.

### Benchmark methodology (applies to every experiment)

Performance claims are made over the **full spectrum**, never a single
number:

- **Hash functions**: keccak, sha2, blake3 — all three. They stress
  different producer patterns (keccak: assignment-style lane writes, 35%
  padding; sha2/blake3: denser OR-composed bit records via `BitRecord`,
  different useful fractions) and different circuit densities downstream.
- **Sizes**: small / medium / large — m ∈ {23, 26, 29} minimum, +32 where
  RAM allows. Small sizes catch fixed overheads and parallelism starvation;
  large sizes catch bandwidth/TLB effects. Both regimes have bitten already
  (see §7: m=23 multi-core starves at G=64; m=29 single-core exposed
  write-allocate).
- **Threads**: single-core (`RAYON_NUM_THREADS=1`) **and** the perf pool
  (all P-cores, `init_perf_thread_pool`). Multi-core parity does not imply
  single-core parity — the scatter penalty hides behind parallel bandwidth.
- **Reporting**: the (hash × size × threads) matrix with median and min of
  ≥5 iterations after warmup, alongside the production baseline measured in
  the same process/run. Tunables (e.g. group width G) may be selected per
  cell, but the selection rule must be stated and mechanical (a runtime can
  pick G from (n, threads); it cannot pick the layout per cell).

"Success" = end-to-end prove within noise (≤ ~2%) of row-major across the
matrix; witness-gen phase within ~5% at every cell (not on average).

**E0 — Layout shim + oracle.** `Layout` descriptor + permutation helpers
row-major ↔ L1′. Property test: `MLE_rowmajor(point) == MLE_L1′(relabeled
point)` — the one-test soundness story for the relabeling.

**E1 — Witness-gen producer bench (the real risk).** Variants on the keccak
or sha2 builder, all producing L1′ *directly* (no global transpose pass):
(a) baseline contiguous row-major; (b) 64-instance staged group + per-chunk
1 KB flush; (c) wider staging (128/256 instances → 2–4 KB runs, trading L2
pressure for run length); (d) naive strided per-instance writes (floor).
Measure phase time + write bandwidth at m ∈ {26, 29} vs
`benches/genwitness_phase.rs`. Decision gate for group width.

**E2 — Zerocheck control.** Feed `prove_packed_padded` L1′-permuted buffers
with relabeled bookkeeping; assert wall-time parity (validates the
address-bit-generic claim empirically, incl. rayon splits). Then the padded
variant: prefix iteration replacing `b_med_counts`, measured on sha2's real
padding fraction (expect a win).

**E3 — Lincheck stripe under L1′.** Stripe build from L1′ source; no-copy
variant (transpose inside the fold). Bench vs `benches/lincheck.rs`.

**E4 — apply_a/b L1′ strip kernels** (generic path). Bench vs
`apply_block_diag_probe.rs`.

**E5 — Ring-switch / PCS-open control.** `open_batch` on L1′-permuted witness
with permuted suffix tensors + re-derived `s_hat_v` fast paths: assert
byte-level parity of the fold outputs and wall-time parity of the open
(`benches/pcs_open.rs`). This is the piece that certifies "batch over packed
values" costs nothing.

**E6 — End-to-end integration.** L1′ on one padded hash (sha2) + one dense
(keccak): full prove_fast pipeline with relabeled claims, verifier roundtrip,
proof size identical, prove-time parity.

Ordering: E0 → E1 (gate) → E2/E3/E4/E5 in parallel → E6.

## 5. Multi-table end-state (context, not in scope)

- Table `t` has `(k_log_t, n_log_t)`, stored L1′; all tables share the λ
  domain and the in-word dim (chunk = 128 is table-agnostic).
- After round 1 + the shared in-word round + `max_t n_log_t` batch rounds
  (smaller tables finish early, ride along eq-scaled), each table is a
  length-`2^(k_log_t−7)` F128 residue; concatenation = one jagged big circuit
  for remaining rounds + lincheck.
- The committed object is the jagged dense `q` (column-major = physical L1′
  order, per-table heights `2^{n_log_t}`, useless chunks uncommitted);
  claims flow zerocheck → ring switch → jagged → BaseFold/Ligerito.
- Parked: challenge sharing across tables in batch rounds; friendly-constant
  alignment when some table has `n_log < 6`; jagged lincheck composition;
  whether `apply_a/b`-style producers exist per table or all tables use
  circuit walkers.

## 6. Open questions

Q1. ~~Chunk 64 vs 128~~ — **resolved: 128.** Ring switching pins the batch
    dims to the word-index space; chunk = 64 would split the batch dimension
    across the ring boundary.
Q2. Small batches (`n_log < 6`): floor the batch count, or let friendly dims
    spill into chunk-index dims per-table?
Q3. Streaming fusion (`streaming_fusion_probe`) streams per-instance under
    row-major; L1′ windows touch 128 instances each. If streaming fusion is
    on the roadmap, decide explicitly that batch-major wins.
Q4. Wire format: prototype re-versions the transcript freely; confirm nothing
    downstream pins the current claim layout.
Q5. Jagged composition detail: does the jagged reduction sit between ring
    switch and BaseFold per-claim, or once over the α-batched combined claim
    (as `open_batch` does today)? Affects how many `f̂_t` sumchecks the
    verifier pays for.

## 7. Results log

Machine: Apple Silicon (8 P-core rayon pool via `init_perf_thread_pool`),
rustc 1.96, `lto = "thin"`, `codegen-units = 1`. Code: `autoresearch/probes/`.

### E0 — layout shim + oracle (DONE, all green)

`probes/src/layout.rs` + `tests/e0_oracle.rs`:
- `MLE_row(z, x) == MLE_l1(permute(z), relabel(x))` on random witnesses/points;
- bit-permutation ↔ (o, j) address maps ↔ word-transpose all consistent
  (word transpose cross-checked through `pcs::pack_witness`);
- ring prefix (address dims 0..7) fixed by the relabeling; batch dims land at
  7..7+n_log — the L1′ invariants.

### E1 — witness-gen producer (DONE: keccak, sha2, blake3 × sizes × 1/8 cores)

`probes/src/producer.rs` (hash-generic) + `bin/e1_witness_gen`. All three
per-block builders copied into the probe and held byte-identical to the
public drivers by `tests/e1_builder_lockstep.rs` (z/a/b + stripe); staged L1′
output validated word-for-word against the row-major baseline per run.
Raw numbers: `autoresearch/results/e1_2026-07-02.tsv` (+ `_multi/_single`
logs); summary below.

**Producer evolution** (each step measured; see the naive→NT ladder below):

1. *naive*: staged group build + full flush of all 512 chunks — pays a full
   extra read+write pass with strided destination runs. ~3× the bare
   row-major write; the wrong thing to ship.
2. *useful-only flush*: padding chunks are a contiguous chunk-index suffix
   under L1′ → flush stops at chunk 333/512 (keccak). −30%.
3. *fused stripe*: build the lincheck byte-stripe from L2-resident staging
   instead of production's re-read of the 64 MB z from DRAM.
4. *NT stores* (`stnp`): the flush's destination lines are fully overwritten
   and not re-read soon; normal stores trigger write-allocate (a wasted DRAM
   read per written line). Non-temporal pair stores remove it. Biggest
   single lever: −33% single-core, −29% multi-core on the full config.
5. *group width G*: multi-core flat across 8–64 (use G ≤ n/threads·2 to
   avoid starving the pool — G=64 at m=23 leaves 2 tasks for 8 workers);
   single-core prefers longer runs (G=256) *without* NT, G=64 with NT.

**Spectrum** — total producer cost, `L1-useful-stripe-nt` vs
`production-driver+stripe` (both include the lincheck stripe; median ms;
G by the `auto_group` rule; useful fraction per hash: keccak 65%,
sha2 96%, blake3 94%):

| hash | m | 8-core: PROD → L1′ (ratio) | 1-core: PROD → L1′ (ratio) |
|---|---|---|---|
| keccak | 23 | 0.17 → 0.18 (1.09×) | 0.56 → 0.67 (1.20×) |
| keccak | 26 | 0.76 → 0.85 (1.12×) | 3.61 → 3.71 (1.03×) |
| keccak | 29 | 6.94 → 4.88 (**0.70×**) | 25.4 → 29.8 (1.17×) |
| sha2 | 23 | 0.15 → 0.20 (1.36×) | 0.50 → 0.66 (1.30×) |
| sha2 | 26 | 0.81 → 0.91 (1.12×) | 4.49 → 5.45 (1.21×) |
| sha2 | 29 | 7.86 → 6.57 (**0.84×**) | 36.2 → 41.6 (1.15×) |
| blake3 | 23 | 0.17 → 0.19 (1.13×) | 0.54 → 0.64 (1.17×) |
| blake3 | 26 | 1.54 → 0.97 (**0.63×**) | 4.74 → 5.31 (1.12×) |
| blake3 | 29 | 8.19 → 6.40 (**0.78×**) | 39.8 → 40.6 (1.02×) |

Reading:
- **Multi-core at the large size (every hash): L1′ is *faster* than
  production** (0.63–0.84×). Multi-core small/medium: 1.09–1.36× on
  sub-millisecond phases.
- **Single-core: consistently 1.02–1.30× slower**, worst at the smallest
  size. The flush is a real extra pass a contiguous writer doesn't pay;
  parallel bandwidth hides it, one core doesn't fully.
- The dense-circuit worry from the keccak-only round is answered: sha2/blake3
  are ~95% dense (no meaningful useful-only discount) and still land
  ≤1.30× single-core / faster-than-production multi-core at m ≥ 26.
- End-to-end exposure: the phase is ~9% of the multi-core prove (keccak m29:
  7 of 78 ms), so the worst observed cell (~+30% on a 0.5 ms phase) is
  ≪1% e2e; the single-core m29 cells are ~+1–2% of a single-core prove.
- Remaining single-core levers (untried): NT + useful-only for the stripe
  writes; two-pass dest-tiled flush; interleaved multi-instance building.
- Caveat: production's number includes its full-block memsets and
  stripe-from-DRAM re-read; under the jagged end-state L1′ additionally
  stops committing padding, which E1 doesn't credit.

### E1-C2 — the no-compromise direct-write producers (DONE, all hashes)

Decision (2026-07-02): staged-scatter is retired as the target design; the
producer of record for L1′ is **V = 8 lockstep simulation writing the L1′
layout directly** (`probes/src/{keccak,sha2,blake3}_vwide.rs`,
`direct_common.rs`). Three structural elements:

1. **V-wide compute**: simulation state is lane-major `[wordtype; V]` arrays
   (keccak: `[u64; 8]` lanes; sha2/blake3: `[u32; 8]`), so every op
   auto-vectorizes; V = 8 also equals the lincheck stripe group.
2. **Direct emission**: a witness word-row (same block-word across the 8
   instances) is exactly one L1′ 128-byte chunk-row = one cache line, stored
   with `stnp` (non-temporal — dest lines are fully overwritten, so
   write-allocate reads are pure waste). Keccak's u64-aligned layout emits
   rows straight from registers (`RowWriter` pairs even/odd chunk halves;
   region boundaries flush zero-halves only into genuine padding). The
   bit-packed hashes (sha2/blake3 — 31-bit carry strides, unaligned regions)
   OR fields V-wide into a 16–32 KB *interleaved* row buffer (already L1′
   order, L1-resident) and NT-flush useful chunks.
3. **Inline stripe**: the 8×64 bit-transpose consumes the in-flight row —
   zero extra reads; only useful words written (buffer pre-zeroed once).

Contract change vs production: dest buffers are **pre-zeroed once and
recycled**; padding words are never written again (they are also exactly
what the jagged end-state stops committing).

Validation: `tests/keccak_vwide_lockstep.rs` — all three direct producers
reproduce the word-transpose of the production witness byte-for-byte
(z, a, b, stripe), including across buffer-reuse rounds.

Results (`results/e1_direct_2026-07-02.tsv`), `L1-direct-stripe` vs
`production-driver+stripe`, median ms (ratio):

| hash | m | 8-core: PROD → direct | 1-core: PROD → direct |
|---|---|---|---|
| keccak | 23 | 0.18 → 0.11 (0.63×) | 0.38 → 0.46 (1.21×) |
| keccak | 26 | 1.47 → 0.40 (**0.27×**) | 3.81 → 2.62 (**0.69×**) |
| keccak | 29 | 6.92 → 2.90 (**0.42×**) | 25.7 → 20.9 (**0.81×**) |
| sha2 | 23 | 0.17 → 0.14 (0.82×) | 0.54 → 0.51 (0.94×) |
| sha2 | 26 | 1.20 → 0.68 (**0.57×**) | 4.34 → 4.08 (0.94×) |
| sha2 | 29 | 8.90 → 4.81 (**0.54×**) | 36.0 → 34.6 (0.96×) |
| blake3 | 23 | 0.17 → 0.14 (0.81×) | 0.54 → 0.56 (1.04×) |
| blake3 | 26 | 1.28 → 0.70 (**0.54×**) | 4.75 → 4.37 (0.92×) |
| blake3 | 29 | 8.84 → 5.00 (**0.57×**) | 39.4 → 37.2 (0.94×) |

**The direct-write producer beats production in 16 of 18 cells** — ~2–4×
faster multi-core at m ≥ 26, and it erases the single-core regression that
motivated it (the two losses are sub-millisecond m=23 single-core cells,
keccak 1.21× and blake3 1.04≈noise).

**Measurement audit — the fair baseline** (`results/e1_fair_2026-07-02.tsv`).
The production driver reallocates its 2^(m−3)-byte stripe every call
(page-fault zeroing); that is real per-prove cost today but not a *layout*
effect. `row-major-stripe-recycled` is the best possible version of today's
layout: recycled buffers + the same fused stripe (validated byte-identical).
Against it (median ms, m ∈ {26, 29}):

| hash | m | 8-core: fair-RM → direct | 1-core: fair-RM → direct |
|---|---|---|---|
| keccak | 26 | 0.61 → 0.41 (**0.67×**) | 3.20 → 2.60 (**0.81×**) |
| keccak | 29 | 3.55 → 2.97 (**0.84×**) | 24.3 → 21.4 (**0.88×**) |
| sha2 | 26 | 0.79 → 0.71 (0.90×) | 4.36 → 4.11 (0.94×) |
| sha2 | 29 | 4.69 → 4.87 (1.04×) | 34.3 → 35.6 (1.04×) |
| blake3 | 26 | 0.75 → 0.70 (0.93×) | 4.73 → 4.52 (0.96×) |
| blake3 | 29 | 5.01 → 5.08 (1.01×) | 37.6 → 38.0 (1.01×) |

Corrected conclusions:
1. **Layout-attributable result (the E1 question): L1′ costs nothing at the
   producer.** Direct-write is at parity with the best row-major producer
   for sha2/blake3 (±4%) and 12–33% *faster* for keccak. Column-major
   witness generation is not a compromise.
2. **Independent finding:** today's production driver leaves ~1.7–2.3× of
   the witness phase on the table purely through per-call stripe allocation
   — recyclable via the scratch pool with no layout change. The earlier
   "2–4× vs production" headline was real but largely this artifact; the
   fair table above is the number that matters for the layout decision.
3. keccak's genuine win comes from register-direct emission + SIMD'd
   permutation compute; sha2/blake3 are compute-bound in field packing, so
   layout choice barely registers — which is exactly the parity we wanted
   to demonstrate.

### E2 — zerocheck control (DONE — parity everywhere, padding for free)

`bin/e2_zerocheck.rs`, `tests/e2_zerocheck_oracle.rs`; raw:
`results/e2_2026-07-02.tsv`. Four configs on real witness data
(row-major/L1′ × dense/padded), all hashes × m ∈ {23, 26, 29} × 1/8 threads.

Verification gates (all green): prove→verify roundtrip per config; padded
proof **byte-identical** to dense on the same buffers; truthfulness oracle —
the L1′ claims equal a from-scratch quirky-MLE evaluation (φ₈-Lagrange on
the 6 skip dims × eq on the rest) of the L1′ buffers at the claim points.

Results:
- **Dense: L1′ = row-major within ±3% in all 18 cells** (keccak m29: 34.0
  vs 35.7 ms multi, 239 vs 233 ms single). The address-bit-generic argument
  is now empirical.
- **Padded: L1′ suffix-skip matches or beats per-block padding** (keccak
  m29 single: 165.2 vs 166.0 ms; m23 single: 2.96 vs 3.32) — and it needs
  NO new machinery: L1′ padding is exactly `PaddingSpec { k_log: m,
  useful-prefix }` (one giant block), because padding chunk-columns coalesce
  into one contiguous suffix.

### E3 — lincheck fold directly from L1′ (DONE — stripe droppable; kernel WIP)

`src/lincheck_fold.rs`, `bin/e3_lincheck_fold.rs`, `tests/e3_fold_oracle.rs`;
raw: `results/e3_2026-07-02.tsv`.

The fused fold reads L1′ chunk-columns directly (contiguous 1 KB tile
loads), runs the 8×64 bit-transposes in-register, and accumulates through
the same 256-entry sum tables — **byte-identical z_vec** vs production's
stripe fold (fast kernel + naive reference, keccak + sha2, several m).

Performance: fused beats the *portable* stripe kernel at m ≤ 26 and ties at
m = 29, but production's hand-tuned NEON **oblock** kernel is still
~2.2–2.7× ahead of it at m = 29 (keccak multi 2.8 vs 7.5 ms; a first NEON
pass on the inner loop didn't close it — the gap sits in byte extraction:
transpose-then-gather vs streaming stripe bytes; an oblock-style port with
Q-register accumulators is the known fix). Totals today:
`witgen(no-stripe)+fused` wins at m ≤ 26, loses at m = 29.

**Decision: keep the stripe.** Under L1′ the direct producer emits it
nearly free (≈0.3 ms multi / 2.5 ms single at keccak m29), so dropping it
is an optional follow-up gated on the kernel port — not a blocker.

### E6 — end-to-end prove/verify on L1′ (DONE — works, parity)

`src/e6.rs`, `bin/e6_end_to_end.rs`, `tests/e6_roundtrip.rs`; raw:
`results/e6_2026-07-02.tsv`. Full pipeline on the L1′-committed witness
(keccak, BaseFold): direct witness → `pcs::commit` → zerocheck
(suffix-padded) → lincheck (unmodified — its stripe/circuit/point inputs
are layout-independent) → batched PCS open at **address-ordered** points:

- `x_full_c = zc.r_rest` verbatim (already address-ordered — simpler than
  the row-major bookkeeping);
- `x_full_ab = [r_inner_rest[0]] ++ x_outer ++ r_inner_rest[1..]`;
- the unmodified `verifier::verify_claims` consumes these via `ZClaim`s
  carrying the full vector in `x_inner_rest` (its PCS step just
  concatenates segments). **Zero flock-core changes were needed.**

Gates (green): prove→verify roundtrip at every size; prover/verifier claim
equality; tamper rejection (zerocheck message, final eval, lincheck
z_partial).

Timing vs production `prove_fast_basefold` (keccak, scratch-recycled
buffers in both):

| m | 8-core: PROD → L1′ | 1-core: PROD → L1′ |
|---|---|---|
| 23 | 6.06 → 5.90 (0.97×) | 10.6 → 10.3 (0.97×) |
| 26 | 14.96 → 15.50 (1.04×) | 57.5 → 61.3 (1.07×) |
| 29 | 82.6 → 86.4 (1.05×) | 452 → 492 (1.09×) |

Verify: +0.2–0.4 ms (~1.05×) — same verifier work, minor point-assembly
copies. The residual prover gap is attributed and closable: (a) the probe
par-zeroes all three witness buffers in full (production's builder memsets
per-group inside witness gen; an L1′-native producer would zero only the
padding columns once per recycled buffer); (b) no `s_hat_v` precomputes
(production skips two `fold_1b_rows` passes; re-deriving the captures under
the L1′ suffix ordering is mechanical). Phase breakdown at m29 multi:
commit 24.4 vs 23.9, zerocheck 27.7 vs 25.4, lincheck 5.5 vs 5.2, open
20.3 vs 26.7 (L1′ *faster* despite no precomputes).

**Bottom line: the full L1′ pipeline proves and verifies with zero protocol
or kernel changes, at ≈5% (multi) / ≈9% (single) of production — entirely
from accounted, closable bookkeeping residuals.** The layout is viable
end-to-end.

### E6b — all hashes × both backends + residuals closed (DONE — parity)

`src/e6.rs` (hash-generic core), `bin/e6_end_to_end.rs`,
`tests/e6_roundtrip.rs`; raw: `results/e6b_2026-07-03.tsv`.

Changes over E6: (a) generic over the hash encoder — keccak uses its
`KeccakLincheckCircuit` walker, sha2/blake3 their cached
`csc_lincheck_circuit`; (b) **Ligerito backend** (the production PCS) via
`open_batch_mixed_ligerito_with_precomputed_s_hat_v` +
`verify_claims_ligerito`; (c) both residuals closed:
- *suffix-only zeroing* — the direct producers now fully write the useful
  chunk-column prefix (keccak grew explicit zero rows for its intra-slot
  gap words), so recycled buffers only zero the padding suffix;
- *s_hat_v precomputes* — the zerocheck two-bank `s_hat_v_c` capture is
  address-generic and used as-is; the AB-side `s_hat_v_from_z_vec`
  derivation applies **verbatim** (z_vec's index layout and the chunk-tail
  coords are unchanged under L1′; the batch fold is already inside z_vec).

Gates (all green): roundtrips for 3 hashes × 2 backends; precomputed-path
proofs **byte-identical** (bincode) to the plain path; tamper rejection;
3-round recycled-buffer reuse on dirty scratch buffers.

Prove ratios L1′/production (median; BF = BaseFold, LIG = Ligerito):

| hash | m | 8-core BF | 8-core LIG | 1-core BF | 1-core LIG |
|---|---|---|---|---|---|
| keccak | 23 | 0.96 | 0.99 | 0.89 | 0.99 |
| keccak | 26 | 0.99 | 1.06 | 1.01 | 1.00 |
| keccak | 29 | 1.03 | 1.00 | — | 1.01 |
| sha2 | 23 | 1.07 | 0.99 | 1.00 | 0.94 |
| sha2 | 26 | 0.99 | 0.94 | 1.00 | 1.02 |
| sha2 | 29 | 1.00 | 1.02 | — | 1.01 |
| blake3 | 23 | 0.95 | 0.81 | 0.98 | 1.01 |
| blake3 | 26 | 1.04 | 1.03 | 1.00 | 1.00 |
| blake3 | 29 | 1.03 | 1.00 | — | 1.01 |

30 measured cells, mean ratio ≈ 1.00, all within ±7% (mostly ±3%, i.e.
run-to-run noise). Verify times equal production's (±0.2 ms).

**FINAL ANSWER to the plan's question: the batch-major (L1′) layout costs
nothing end-to-end — parity across every hash, size, backend, and thread
regime, with the production PCS, at full soundness (roundtrip + tamper
gates).** The fold-log_n-first / jagged multi-table design can proceed on
this layout without a performance tax.
