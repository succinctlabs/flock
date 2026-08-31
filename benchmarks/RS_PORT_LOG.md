# RS-path port log

Record of porting engineering optimizations from the Yukon challenge repo
(`Layr-Labs/flock-challenge` @ `c576e68`) into this tree's RS zerocheck path.
Goal: close the gap while keeping added code minimal, and keep every change
attributable to a measured per-phase delta.

Every number here is `benchmarks/breakdown_phases.sh` at 2^18 BLAKE3 on an
**Apple M1 Max (8 P-cores, 32 GB)**, back-to-back A/B in the same session.
Reproduce a row with:

```sh
LABEL=<tag> OUT=phases.tsv ./benchmarks/breakdown_phases.sh
```

## Baseline (`9793190`) and the target

| phase | ours ST | ours 8T | challenge ST | challenge 8T |
|---|---:|---:|---:|---:|
| witness | 351.4 | 81.1 | *not reported* | *not reported* |
| commit | 1277.4 | 196.2 | 1796.7 | 255.9 |
| zc round 1 | 1441.4 | 196.7 | 314.1 | 43.7 |
| zc round 2 | 433.1 | 58.8 | *(combined)* | *(combined)* |
| zc rounds 3+ | 452.4 | 66.4 | 538.2 (r2+r3+) | 78.0 |
| lincheck | 183.4 | 30.6 | 31.9 | 13.4 |
| open | 636.9 | 100.2 | 565.2 | 89.5 |
| **total** | **4781.0** | **723.9** | — | — |

Two caveats on the challenge-repo column, both established with the session
that produced it:

1. Those numbers come from `prove_fast_timed`, which in that repo is a
   structurally separate path from `prove_fast` and measured **~15% slower**
   (405.7-420.7 ms vs 473-484 ms for the same work). Ours agrees with its own
   headline `prove_fast` to within 1.4%, so the two columns are not on equal
   footing. Where the 15% sits is unknown; if it is concentrated in `commit`
   (the dominant phase there, and the one touching its pinned allocation) then
   its commit figure is overstated by a lot.
2. That repo moves work **across phase boundaries** —
   `round1_c_fold4_from_lincheck_stripe` and `stage_c_prelude_for_tail_fill`
   shift C-side work between lincheck and the zerocheck. Its very low lincheck
   number is therefore partly re-attribution, not necessarily a real win.

Treat the column as directional, not as a scoreboard.

## Attempts

| # | change | phase | ST | 8T | verdict |
|---|---|---|---|---|---|
| 1 | batch reduced msg muls via `ghash_mul_vec2_neon` | rounds 3+ | 455.2 → 500.8 (**+10.0%**) | 69.2 → 78.3 (+13.2%) | **reverted** |
| 2 | `WideNeon` register-resident accumulator | round 2 | 433.1 → 375.2 (**−13.4%**) | 58.8 → 55.0 (−6.4%) | **kept** |
| 3 | same, rounds 3+ | rounds 3+ | 452.5 → 455.7 (+0.7%) | 68.6 → 69.2 (+0.9%) | dropped |
| 4 | defer round-1 partial-sum reduction to once per x_hi | round 1 | 1441.4 → 1447.1 (+0.4%) | 196.7 → 203.2 (+3.3%) | **reverted** |
| 5 | port the multilinear lookahead to the RS tail | rounds 3+ | −7.5% *(pre-measured)* | — | not attempted |
| 6 | fetch inv-NTT rows with one `LD1 x4` | round 1 | 1455.8 → 1634.6 (**+12.3%**) | 192.1 → 218.4 (+13.7%) | **reverted** |
| 7 | fold XOR accumulation into `EOR3` pairs | round 1 | 1444.3 → 1469.9 (+1.8%) | *(8T arm discarded, drift)* | **reverted** |
| 8 | hoist the challenge-independent AB transform out of the zerocheck, `rayon::join`ed with the commit (+ `stnp` non-temporal stores) | round 1 | 1474.6 → 601.1 (**−59%**) but **total unchanged** | 219.8 → 81.4 (−63%), total unchanged | **reverted** |
| 9 | nibble-split convert tables (64 KiB → 8 KiB hot table, gathers 48 → 96/lane) | round-1 drain | headline 53508 → 51582 comp/s (**−3.6%**, base 8/8) | — | **reverted** |
| 10 | lincheck-stripe dedup for round 1's C input | round-1 drain | *impossible* — byte groupings run on disjoint axes (stride 64 vs 2^14) | — | **closed on algebra** |
| 11 | geometric eq-build in lincheck | lincheck | whole eq build is 0.13 ms; geometric variant *slower* at 5/6 sizes | — | **closed for free** |
| 12 | **skip structurally-zero b K-rows** | round 1 | **1475.4 → 1433.0 ms (−42.4, −2.9%), head 8/8** | — | **KEPT** |
| 13 | constant-fold all-ones b K-rows | round 1 | 1420.0 → 1413.9 (−6.1, −0.43%), 7/8 | — | reverted (~90 lines for 6 ms) |
| 14 | **two lanes per iteration in the drain** | round-1 drain | **1483.1 → 1420.0 ms (−63.1, −4.3%), head 8/8** | — | **KEPT** |
| 15 | **unreduced pmull accumulate + x^K weight split (x⁴ table image · x² byte-mul · u16 shift)** | round-1 prep | **1401.4 → 1273.4 ms (−128.0, −9.1%), head 8/8, every pair −8.7..−10.1%** | — | **KEPT** |
| 16 | **stripe-fold C side: round-1 C banks from one multilinear fold of the lincheck stripe; drain runs AB-only, C transpose deleted** | round 1 | **1277.5 → 1204.7 ms (−72.7, −5.7%), head 8/8** | — | **KEPT** |
| 17 | **q-resident round 2: fold outputs stay in q registers, in-register karatsuba `mul_q` (5 PMULLs), `WideNeon` fed directly** | round 2 | **358.1 → 311.6 ms (−13.0%), 8/8, every pair −12.6..−13.3%** | — | **KEPT** |
| 18 | **fused q-resident rounds-3+ tail: fold+message in one pass, second read pass over multi-MB chunks deleted** | rounds 3+ | **449.5 → 306.6 ms (−31.8%), 8/8, every pair −31.1..−32.6% — largest single win of the effort** | — | **KEPT** |
| 19 | **stripe fold through lincheck's tiled dispatcher** (was calling the portable fallback; its 256 KiB accumulator thrashes L2 at k_log=14) | round-1 C fold | **1191.0 → 1023.7 ms (−167.3, −14.0%), 8/8, every pair −13.6..−14.4%** | — | **KEPT** |
| 21 | b === all-ones round-2 pair degeneration | round 2 | 312.0 → 312.4 ms, sign 4-4 — **null** (third partial-skip-vs-ILP confirmation) | — | reverted |
| 22 | non-temporal (STNP) round-2 output stores | round 2 | 313.1 → 350.3 ms (**+12%**), base 8/8 — STNP *inverts* on M1 (M4-specific idiom, their ablation +1.2%) | — | **reverted** |
| 23 | **four lanes per iteration in the AB-only drain** | round-1 drain | 960.5 → 954.4 ms (−6.0, −0.6%), 7/8 | — | **KEPT** (10 lines) |
| 25 | word-extract in the round-2 fold | round 2 | 313.1 → 305.3 ms (−7.9, −2.5%), 6/8 | — | **KEPT** |
| 26 | two round-2 pairs per iteration | round 2 | 306.5 → 317.9 (+3.7%), base 8/8; **re-verified under challenge**: 307.1 vs 325.4, unroll worse 8/8 disjoint, regression larger under load (register spills amplify with memory contention) | — | reverted, double-sourced |
| 24 | static-b partial loads — bounded by probe, not implemented | round-1 prep | deleting ALL b gathers: 954 → 750.6 ms, so ceiling = 33.7% × 204 ≈ **69 ms**; realistic ≈ 15–20 at measured partial-skip capture rates, vs 200–800 lines | — | **closed on the bound** |
| 20 | **word-extract addressing in the prep** (16 byte-loads per K-row → 2 word loads + shifts) | round-1 prep | **1027.6 → 974.8 ms (−52.7, −5.1%), 6/6** | — | **KEPT** |

Net kept: **round 2 −13.4% ST**, total 4781.0 → 4716.2 ST (−1.4%), for 179
added lines.

## What the negative results tell us

- **Round 1 is not multiply-bound** (attempt 4: reductions cut by a factor of
  `big_lo_size`, products from 6 PMULLs to 3, no movement). An earlier version
  of this log inferred from that "round 1 is gather-bound on the convert
  table". **That inference was wrong.** Measuring the split directly
  (temporary `FLOCK_R1_SPLIT` scaffold, since removed) gives, at 2^18 ST:

  | round-1 component | ST ms | of round 1 | of whole prove |
  |---|---:|---:|---:|
  | `shift_reduce_inner_ab` | ~810 | 56% | 17% |
  | `bit_transpose_64bytes` | ~136 | 9% | 3% |
  | `accumulate_convert` | ~521 | 36% | 11% |

  So the convert-table accumulate is only a third of round 1; the prep pass
  dominates. Reproduce by re-adding two `Instant` counters around the b_med
  prep loop and the `accumulate_convert_with_s_hat_v` call in
  `process_one_x_hi_with_s_hat_v` (per-x_outer_lo granularity; per-b_med timers
  add ~470 ms of their own overhead and only the ratio survives).
- **`shift_reduce_inner_ab` is limited by neither load-issue nor XOR-issue
  count.** Attempts 6 and 7 cut load instructions 4x (8 per byte-pair to 2) and
  XOR ops ~1.75x (56 to 32) respectively; the first cost 12.3% and the second
  1.8%. On Apple cores the four-register structured `LD1` is microcoded and
  loses to four independent 128-bit loads, and `EOR3` bought nothing. What is
  left as the plausible limiter is load *latency* / L1 port pressure against
  the 16 KB inv-NTT table — whose gathers are data-dependent and cannot be
  batched — plus the `gf8_mul_vec16` work. Neither yields to a local rewrite,
  which is consistent with the challenge repo needing specialized and generated
  kernels here rather than a tidier loop.
- **Interface shape can outweigh instruction counts.** Attempt 1 reduced PMULLs
  (4/mul vs binius's 6) and still lost 10%, because `ghash_mul_vec2_neon` takes
  and returns `[F128; 2]` and forces operands through memory. It earns its
  place in the NTT and `f128_slice` call sites only because one operand there
  is a loop-invariant broadcast.
- **Lookahead loses on this hardware.** `cargo bench --bench ag_lookahead_ab`
  (paired, same-process, proofs asserted bit-identical) gives classic 285.9 ms
  vs lookahead 307.2 ms ST at m=30 — lookahead faster 0/4 runs. The code's own
  comments target an M4; do not port it to the RS tail on an M1 without
  re-measuring.
- **Our `commit` is already ~1.4x faster than theirs** (1277 vs 1797 ms ST),
  plausibly from #29/#30 which `c576e68` predates. Cross-pollination runs both
  ways; commit is not a target for us.

## Where the branch ended up

Three changes kept, everything else reverted with its measurement in the commit
message. Against `9793190` (the harness-only commit, before any optimization),
2^18 BLAKE3 ST, paired n=8 with alternating arm order:

| level | base | head | delta |
|---|---:|---:|---|
| round 1 (`round1 URM`) | 1477.30 ms | 1023.72 ms | **-30.7%**, each step 8/8 |
| round 2 (`round2 fused fold`) | ~433 ms | 311.56 ms | **-28%** (WideNeon + q-resident) |
| rounds 3+ (`rounds 3+ tail`) | ~455 ms | 306.56 ms | **-33%** (fused one-pass q-resident) |
| end-to-end headline | ~52,400 comp/s | ~58,100 comp/s | **~+11%**, 8/8 (clean-band read; per-pair delta stable +9.6..+12.4% even through throttled pairs) |

(Supersedes the interim +5.1% figure from the four-win state. The final run was
taken with an active browser session; pairs 4-8 form tight bands on both arms
and cross-check against the predicted sum of the individual wins, ~+7.9%.)

Caveat on the end-to-end figure: the base arm spread that run was wide
(46894-51655, ~10%) while head was tight (51231-53209), so the point estimate is
soft even though the sign test is not. The round-1 number is the better
measured of the two.

The seven kept changes:

1. **Round-2 NEON register accumulator** (~179 lines) -- `WideNeon`, a 256-bit
   product held as two uint64x2_t instead of the GPR-resident F256Unreduced.
   -13% on `zc_round2`.
2. **Structurally-zero b K-row skip** (~10 lines) -- see above. Part of the
   -5.0%.
3. **Two lanes per drain iteration** (~50 lines) -- see above.
4. **Unreduced pmull accumulate + x^K weight split** (~80 lines) -- the
   challenge repo's top-attributed AB-prep mechanism, ported as an idea.
   -128 ms on round 1 by itself (1401.4 -> 1273.4, 8/8, predicted 100-130 from
   their attribution).
5. **Stripe-fold C side** (~150 lines) -- round-1 C banks from one multilinear
   fold of the lincheck stripe; drain AB-only, C transpose deleted. -72.7 ms.
6. **q-resident round 2** (~120 lines) -- fold outputs stay in q registers,
   in-register karatsuba mul_q, WideNeon fed directly. -46.6 ms (-13.0%).
7. **Fused q-resident rounds-3+ tail** (~90 lines) -- fold and message in one
   pass; the second read over multi-MB chunks deleted. -142.9 ms (-31.8%),
   the largest single win. Two earlier failures on this exact loop (pair-mul
   kernel +10%, WideNeon-alone 0%) had located the cost in the pass structure
   and struct crossings, not the arithmetic.

## What bounds each round-1 kernel (measured, not inferred)

Five local rewrites of round 1 have now failed. Taken together they say
something fairly precise about the two kernels, which is more useful than any
of the individual results:

- **The drain is bound by gather COUNT.** Doubling gathers (48 → 96 per lane)
  while shrinking the hot table 8x (64 KiB → 8 KiB) cost 3.6%. M1 has 128 KiB
  of L1D per performance core, so the 256-row convert table already fit and
  there was no footprint problem to fix. Calibrating from that regression, the
  drain's 48 gathers/lane are worth roughly 170 ms of its ~521 ms.
- **The prep kernel is bound by neither load-issue nor XOR-issue count.**
  Cutting load instructions 4x (LD1 x4) cost 12.3%; cutting XOR ops 1.75x
  (EOR3 pairing) gave +1.8%. Deferring its reduction entirely moved nothing.
- **And it is not bound by anything the witness's structure could unlock.**
  Byte statistics of the packed BLAKE3 witness at 2^12 blocks:

  | buffer | zeros | dominant byte | all-0xff rows | uniform 8-byte rows |
  |---|---:|---|---:|---:|
  | a | 6.6% | — | 0.0% | 5.9% |
  | b | 6.1% | **0xff at 28.9%** | 9.4% | 15.3% |
  | z | 12.6% | — | 0.0% | 5.9% |

  `b` is strikingly non-random, which is presumably why the challenge repo has
  a `static_b` / `mixed_const_b` / `single_k0_static_b` kernel family. But the
  0xff bytes are scattered inside mixed rows rather than clustered: only 9.4%
  of b's aligned 8-byte rows are uniformly 0xff, and the 5.9% all-zero rows
  (identical in all three buffers) are padding that `b_med_counts` already
  skips. Row-level constant specialization is therefore worth ~4% of prep
  loads here -- order 10 ms -- not the 262 ms of the AB-prep gap.

The uncomfortable implication: their AB prep is 548 ms against our ~810 ms
while doing strictly MORE memory work (it streams 512 MiB out through
non-temporal stores; ours writes a 1 KiB L1 scratch). So their kernel is
genuinely ~1.5x better code at the same computation, and none of the
structural explanations we can test account for it. That points at
`fused_apply_one_k_fast` / `fast_shift_reduce_with_policy` / the 839-line
generated `aarch64_bstatic_gen.rs` -- i.e. the specialized-and-generated
kernel zoo, which is exactly the bloat this effort set out to avoid.

## Anatomy of the remaining zerocheck gap (from the challenge-repo session)

The session holding the c576e68 checkout read its own kernels and explained the
mechanisms behind each sub-phase advantage. Recorded here because the *shape*
of each answer matters for what upstream should do next.

**AB prep (548 vs ~770 ms): one real arithmetic-kernel win.** Their
`fused_apply_one_k_fast` replaces the incumbent's per-lane REDUCED GF(2^8)
multiply (`gf8_mul_vec16`) with an UNREDUCED carry-less multiply (raw
PMULL/PMULL2) and defers all reduction into an incremental Horner fold
(`acc = acc*x XOR lo XOR hi`, one fused BCAX per step, precomputed carry
constant x^16 mod p = 0x5e). Same gathers, same passes; cheaper arithmetic.
~50-60% of the prep gap, per their attribution. The rest is traffic: zero-copy
views into the precompute buffer instead of a per-row scratch memcpy
(DIRECT_AB_ROWS) and skipping the zero-fill of dead tail rows
(AB_COMPACT_STORE). This is the one honest kernel-quality gap, and it is
portable as a technique.

**Round-1 drain (314 vs ~590 ms): structural — they do not run this
computation.** Their active path never reads c_packed, never bit-transposes,
never touches the convert table. Because C = I (C aliases z), the round-1
C-claim derives from the LINCHECK STRIPE via an eq-fold
(`partial_fold_packed_z_best`) plus ring-switch's own fold8
(`s_hat_v_fold8_from_z_vec`), a small quad fold, and one collapse to the
two-half s_hat_v_c layout. This refines the "stripe dedup does not exist"
section below: the index algebra there is correct — the stripe cannot feed
*this tree's drain shape* — but they compute the same OUTPUT by a different
algorithm, so the dedup exists at the algorithm level, not the buffer level.
Their per-lane gather cost model simply does not apply. (Unresolved caveat:
part of their 314 ms may be a Metal GPU prefix they could not isolate.)

Decomposition of OUR 590 ms drain, all measured: ~136 ms C bit-transpose,
~75 ms per-lane eq multiplies (timing probe: removing the muls took round 1
from ~1403 to ~1330 ms), ~380 ms gathers+XORs.

**Rounds 2+ (538 vs ~800 ms): NOT lookahead — lookahead is a shared regression.**
At our request they kill-switched their sumcheck lookahead on their own machine:
591 -> 514 ms with it OFF (13-15% loss when on, all 6 paired samples),
matching this tree's M1 measurement (285.9 vs 307.2 in the AG tail). It is
default-ON in their tree and loses on both chips. Cascade2/cascade3 (composed
double-folds via a 32 KiB rho byte table) was the initial candidate for their
remaining rounds-2+ edge, but a follow-up kill-switch run on their machine says
otherwise: with cascade OFF and lookahead ON, rounds 2+ measured ~736 ms avg
(noisy, 612-964, low confidence), while their fastest configuration remains
lookahead fully OFF (~514 ms) -- where cascade structurally never fires, since
it requires lookahead. So their rounds-2+ advantage over this tree's ~800 ms
lives in the BASE per-round kernels: the register-resident wide-arithmetic
family in their multilinear kernels (2779 lines against this tree's 56; the
same family our round-2 WideNeon win was one piece of), not the fusion
machinery. Cascade's own contribution could not be cleanly isolated.

Direct consequence for THIS tree: the ligerito OPEN runs the same lookahead
family by default (landed in #30, presumably tuned on the M4 Max reference
machine) with the kill-switch `LIG_LOOKAHEAD_DISABLE=1`. Measured separately —
see below if a result was recorded.

**Connective tissue:** a size-classed scratch allocator (take_f128/give_f128)
recycling large buffers across prove calls; the same recycle-don't-allocate
pattern as their largest historical non-GPU win.

Net revision to the earlier "no big idea" conclusion: the gap is NOT dozens of
micro-tunings. It is one structural algorithm change (drain), one arithmetic
kernel (prep), one fusion family (rounds 2+), plus a shared lookahead
regression that is an upstream opportunity rather than a deficit.

## The lincheck-stripe dedup does not exist (checked, closed)

The most promising remaining idea was that `z` gets transposed twice -- once
into the lincheck stripe, once again by round 1's `bit_transpose_64bytes` --
and that round 1 could read its C input out of the stripe instead, deleting
~136 ms of transpose plus the ~113 ms of C gathers it feeds. The challenge
repo's active path is even named `round1_c_fold8_from_lincheck_stripe`.

It does not work: the two byte-groupings run along different axes. Traced by
setting one logical witness bit at a time (m=18, k_log=14):

| byte | logical bits feeding it | stride |
|---|---|---:|
| round-1 C `[b_med=0][lane=0]` | 0, 64, 128, ..., 448 | 64 |
| stripe `[byte_idx=0][i_inner=0]` | 0, 16384, ..., 114688 | 16384 |

With `logical = i_inner + i_outer·K` and `K = 2^14`, round 1's byte runs along
logical bits 6-8 (three of the within-block inner dims) while the stripe's runs
along `i_outer`'s low three bits, logical bits 14-16. Disjoint. Converting one
grouping into the other is precisely the transpose we wanted to skip.

Two corollaries:

- The challenge repo does not avoid this transpose either. Its C drain still
  calls `bit_transpose_64bytes` into a local scratch (confirmed by the session
  that has that checkout), so `..._from_lincheck_stripe` names the buffer it
  reads, not an avoided pass.
- The z transpose is already better handled here than "in parallel" would be:
  `generate_witness_with_ab_packed_and_lincheck` fuses it into witness
  generation, bit-transposing z u64s into the stripe while they are still hot
  in L1, and replaces the standalone `pack_z_lincheck_from_packed` on the fast
  path. Splitting it out to overlap it would add a 512 MiB DRAM round trip to
  buy concurrency on an already-saturated pool -- the same trap as the AB
  hoist. The remaining `pack_z_lincheck_from_packed` call sites are the generic
  `prove_ligerito` path only.

## The lincheck gap is re-attribution, and its fold has no reachable headroom

Their lincheck is 32 ms against our 183 ms -- the largest ratio in the table
(5.7x) and the row we explored last. It is not a real gap.

**Physics.** `partial_fold_packed_z_neon_*` is byte-table driven: per input byte
it loads the byte, loads a 16-byte `build_sum_table` entry, and XORs into an
accumulator pinned in a Q register. At the 2^18 BLAKE3 shape z_packed is 512
MiB, so 2^30 loads. M1 sustains ~3 loads/cycle, giving a floor of ~112 ms
(~84 ms allowing for the `useful_bits` padding skip). Their 32 ms works out to
**0.19 cycles per input byte**, roughly 3.5x below that floor, and even a
16-bit-table variant (two bytes per lookup) only floors at ~0.34. So their
lincheck cannot be performing this fold -- the C-side fold is presumably
computed in round 1 from the stripe and reused, consistent with the active path
being named `round1_c_fold8_from_lincheck_stripe`.

Correcting for this, the real comparable gap is ~970 ms, not ~1120 ms.

**And the genuine headroom is not reachable.** Our fold does sit 1.6-2.2x above
its own floor, but:

- It is insensitive to blocking. `FOLD_AB=1` A/Bs the size-aware dispatch
  against forced `iblock` interleaved per m: 0.988x at m=26, 0.972x at m=28,
  0.994x at m=29. Both strategies land within 3%.
- The only load-reducing transform is arithmetically dominated. `build_sum_table`
  builds 256 entries with 255 XORs by doubling; a 16-bit table needs 65535 XORs
  to enable only `k = 2^14 = 16384` lookups per stripe, so the build costs 4x
  more than it saves. (And the two stripe bytes for one `i_inner` are `k` bytes
  apart, so forming a u16 index needs two loads regardless -- loads would go
  4 -> 3, not 4 -> 2.) It could only pay at much larger `k_log`.

**Stale comment worth fixing.** `benches/lincheck.rs` documents oblock beating
iblock by "≈1.4-1.7x by m=28-29 at this k_log". That does not reproduce here --
see the ratios above. Whoever tuned `OBLOCK_MIN_N_LOG = 16` did it on different
hardware or the win has since regressed; do not trust that comment on M1.

**Also closed for free:** the geometric eq-build (`build_eq_table_optimized` in
their tree, prototyped here in `benches/eq_build_probe.rs`). Running that probe
shows the entire `SplitEqGhash::new` at the round-2 shape costs 0.13 ms, and the
geometric variant is *slower* than the standard build at 5 of 6 sizes. Worth
approximately zero.

## The measured round-1 decomposition (post-stripe-C), and what it closed

A second FLOCK_R1_SPLIT probe (since removed) replaced the estimated
decomposition with a measured one and immediately found a defect:

| component | estimated | measured | note |
|---|---:|---:|---|
| AB prep | ~640 | ~715 | 90% gathers: a gathers-only probe put the whole multiply tail at ~67 ms, killing the h4-Horner port idea (ceiling ~15 ms for ~100 lines) |
| stripe fold | ~180 | **340 -> 166** | was calling the PORTABLE fold; `partial_fold_packed_z_best` (lincheck's tiled NEON dispatcher) halves it -- the portable kernel's length-k accumulator is 256 KiB at k_log=14, twice M1's L1D |
| AB drain | ~340 | ~167 | near its gather floor all along; the estimate that made it look like a target was wrong |

With the dispatcher fix, drain+fold is ~333 ms against their ~314 --
effectively at parity (and theirs may include an unquantified Metal prefix).
The entire remaining round-1 gap (~150 ms after word-extract, if it lands)
is AB-prep gather machinery, where the remaining lever is the static-b
partial-load import previously ruled out as bloat.

Zerocheck like-for-like standing: ~1643 vs ~1376 = **1.19x** (was 1.74x
fairly accounted, "2.7x" as first misread). Rounds 3+ (1.03x) and drain+fold
(1.06x) are closed; round 2 (1.45x, ~97 ms, their compact-fold mechanism) is
the largest remaining relative gap.

## Machine-specific tunings: three inversions/nulls on M1

Mechanisms measured good on the challenge tree's M4 that fail on M1:
oblock fold gating (comment claims 1.4-1.7x, measures 0.97-0.99x here),
zerocheck-tail lookahead (default-ON there, loses 7-15% on BOTH machines),
and STNP output stores (+1.2% there, **+12% regression** here -- the
non-temporal hint costs store throughput on M1 instead of saving RFO
traffic). Port memory-system hints only with a local paired measurement.

## Final ST+MT comparison vs the session baseline (lightweight, 2026-08-25)

Single instrumented invocation per cell; untouched phases (witness, commit,
lincheck, open) matched across arms to <1% at both thread counts, validating
the run. All 13 kept optimizations, vs `9793190`:

| phase | ST base | ST head | delta | 8T base | 8T head | delta |
|---|---:|---:|---:|---:|---:|---:|
| zerocheck | 2357 | 1638 | **-30.5%** | 322 | 221 | **-31.4%** |
| -- round 1 | 1465 | 961 | -34% | 191 | 129 | -33% |
| -- round 2 | 437 | 304 | -30% | 59 | 41 | -31% |
| -- rounds 3+ | 454 | 310 | -32% | 65 | 48 | -26% |
| **headline** | 52.5k | **62.5k c/s** | **+18.9%** | 349k | **406k c/s** | **+16.1%** |

The MT columns are the first multithreaded measurement since any optimization
landed: every win transferred to 8T at essentially its ST magnitude, and the
end-to-end gain in the threaded (production/scored) configuration is +16%.

Known anomaly, comparison-safe: the open phase measured ~850 ms ST on BOTH
arms today vs ~636 in earlier sessions -- a day-scale bimodality also seen
once before (the 857 ms lookahead-test reading). Same on both arms, so no
delta is affected; flag for the commit/open campaign.

## Cross-tree comparisons: GPU-status-uncertain (major caveat, 2026-08-25)

The challenge-tree session re-ran its own round-2 measurement (identical
config, same machine, days apart) and got **342 ms where it had reported
215** -- a 127 ms swing it flagged rather than explained away. Its raw dump
shows round-1 samples of 92-95 ms jumping to 309 ms MID-RUN with no code
change. 92 ms is implausibly fast for its CPU drain+fold; it is exactly what
its Metal GPU round-1 prefix produces, and that prefix "fires if the shape
matches and Metal's available" with no warmup latch. The Metal-assist hypothesis was TESTED AND REFUTED as a complete explanation:
with both GPU arms force-disabled (FLOCK_NO_GPU_ZEROCHECK=1 and the separately
gated FLOCK_NO_GPU_ZC_R2=1), their first samples remained chaotic (a 1182 ms
round-2 with no GPU involved). The broader driver: this shared machine ran
builds and benches from three concurrent Claude sessions that day, plus
ambient load. Widen the caveat from GPU-status to: fine-grained (sub-100 ms)
cross-tree bucket deltas from this date are unresolvable, period.

Consequences for this log:
- Every "theirs" column in the cross-tree tables is soft until their GPU
  test reports. Their bracket accounting itself was verified clean (buffer
  takes, tables, and padding all inside their round-2 timer; their tail
  honestly carries the compact-format reconstruction).
- What the stable window of their full-GPU-off run DOES support (samples 3-7,
  tight): their pure-CPU round 2 is 274-285 ms against our same-conditions
  305, i.e. a residual of ~27 ms -- the size of their compact format's
  modeled store saving (~25 ms), the one mechanism deliberately not ported.
  The original "90 ms gap" therefore decomposes as ~25 ms format + ~65 ms
  measurement conditions. Their tail (300-307) matches ours (307) exactly.
  Coarse conclusion that survives all of this: the trees are within ~10% on
  the zerocheck CPU-vs-CPU, and bucket deltas below ~50 ms cannot be
  adjudicated on this machine this week.
- Every KEPT win in this log is unaffected: all were internally paired A/B
  on this tree alone and never depended on their numbers.

## The SHA-256 cross-circuit control

Question: is the challenge repo's remaining advantage BLAKE3 specialization
(its static-b census, degen flags) or generic kernel quality? Control: SHA-256
at 2^16 (m=31), ST, identical command on both trees, where neither side's
structure guards fire at their tuned density.

| tree | best prove_fast | throughput |
|---|---:|---:|
| this branch | 2.05 s | 32,027 h/s |
| challenge (c576e68 frontier) | 1.72 s | 38,170 h/s |

Their advantage on SHA-256: **1.19x** -- statistically the same as the ~1.16x
comparable whole-proof gap on BLAKE3. Conclusion: their edge is uniform,
circuit-agnostic kernel quality (commit path, round-2 compact/NT stores, prep
tail, allocator recycling), and the BLAKE3-specific structure machinery is
performance noise at end-to-end scale on both sides -- consistent with their
own per-switch ablations (1-3% each) and with our ports of that family
(zero-skip -42 ms; b===1 degen null).

Also measured by the control: this branch's campaign improved SHA-256 by
**+18% for free** (27.1k -> 32.0k h/s vs the session-baseline matrix) --
no SHA-specific work was ever done, confirming the kept wins are
circuit-agnostic. The b===1 degeneration port (row 21, reverted) was the
last BLAKE3-structural candidate; with this control there is no reason to
pursue that family further.

## What finally worked, and why

Two round-1 wins landed after eleven failures, and they share a property none
of the failures had: they change **how much work exists** or **how much of it
can be in flight**, rather than how the same work is encoded.

**1. Skip structurally-zero b K-rows (−42.4 ms, 8/8).** A census of the packed
BLAKE3 witness -- 256 word positions per block, 256 blocks, 3 independent
witnesses -- found the circuit pins 38 of 256 8-byte b K-rows regardless of the
inputs, taking only three distinct values:

| value | positions | |
|---|---:|---|
| `0xffffffffffffffff` | 22 (8.6%) | const-one wires |
| `0x0000000000000000` | 15 (5.9%) | structural zeros |
| `0x0001ffffffffffff` | 1 | |

33.7% of all b bytes are fixed, cross-checking the byte histogram (28.9% 0xff,
6.1% zero) from the other direction. The zero case is the strongest: the
inv-NTT transform is F_2-linear, so row(0) = 0, so db = 0, so
y = gf8_mul(da, 0) = 0 and the K-row contributes nothing at all --
`fused_apply_one_k` returns immediately, skipping all 64 table loads and the
four F_8 multiplies. One u64 compare, no census data shipped, no position
tracking, and a disagreeing witness falls through to the generic path.

This is strictly better than the challenge repo's `static_b` fast path, which
still loads a precomputed partial for these rows.

**3. Unreduced pmull accumulate in the prep kernel (−128.0 ms, 8/8) — the
largest single win of the effort, and the challenge repo's own top-attributed
mechanism, ported as an idea (~80 lines).** `gf8_mul_vec16` spent 6 PMULLs per
K-row per block — 2 for the raw product, 4 for a reduction that was redundant,
since the accumulator gets one final reduce anyway. Now the raw product
accumulates unreduced, with the x^K row weight decomposed as x^4 (a pre-scaled
second table image to gather from — F_2-linearity makes scaled entries scale
the XOR-sum) times x^2 (a 6-op byte-mul) times x^(K&1) (a u16 shift). Terms
reach degree 15; both reducers were verified exact over the full 16-bit domain
first (exhaustive tests now permanent in gf2_8). Predicted 100-130 ms from the
challenge session's attribution; measured 128.

**2. Two lanes per drain iteration (−63.1 ms, 8/8).** The drain carries three
XOR chains per lane, each of depth `n_b_med` = 16. The gathers feeding them are
independent but the accumulations are serial, so one lane exposes only three
chains. Interleaving a second doubles that to six with no change in work.

**The all-ones case is the instructive failure.** Predicted ~27 ms from the
zero case's calibration; delivered 6.1. Halving a row's loads halves its
memory-level parallelism at the same time, and the remaining dependency chain
goes latency-bound -- the same mechanism that sank the LD1 x4 attempt. Whole-row
elimination avoids it because no chain survives. That result is what motivated
the lane unroll, which then outperformed the win that inspired it.

**Two censuses that closed leads without any code:**

- `a`-side pinned zeros are *exactly* the same 15 positions as `b`'s (union 15,
  a-only 0) -- the padding rows where both operands vanish. An a-side check
  would add nothing.
- The zero words cluster into the block tail (parity 1, b_med 14-15), giving
  only one fully-zero `(parity, b_med)` group of 32, worth ~1% of drain
  gathers. Whole-`b_med` elimination is not there.

**Combined effect, measured directly.** The two wins together, against the
pre-zero-skip commit in a single session, paired n=8 with alternating arm
order:

  round1 URM  base median 1477.30 ms -> head median 1403.07 ms
              -74.2 ms (-5.0%), head 8/8, ranges disjoint
              (base min 1469.55 > head max 1433.81)

That is less than the 42.4 + 63.1 = 105 ms the individual measurements suggest,
and the combined figure is the one to trust: it is the only one where both arms
ran under the same conditions. The individual runs were taken in different
sessions, and cross-run drift on this machine is large enough to swamp the
difference -- identical code measured 1420 ms in one run and 1483 ms in another.

**Methodology that made these findable.** Earlier attempts were measured on the
end-to-end headline, where a 40 ms effect is ~1% and sits under the noise. These
were measured on `round1 URM` directly via `FLOCK_ZC_TIMING`, taking the min
across the ~5 zerocheck calls in a run, 8 alternating-order pairs per verdict --
about 3x faster per sample and aimed at the phase actually being changed. Note
cross-run drift remains large: the same code measured 1420 ms in one run and
1483 ms in another, so only within-run paired deltas are trustworthy.

## Not attempted, and why

- Anything GPU-gated (`partial_fold_packed_z_best_gpu_split`,
  `ranked_lincheck_fold_gpu_shape`) — out of scope by request.
- The `Round1AbInner` staged pipeline, `c_fold4` mask tables, static-B
  specialization and its 839-line generated kernel. This is where the round-1
  win actually lives, but it is ~6000 production lines across
  `univariate_skip_optimized.rs`, its NEON kernels, and `zerocheck.rs`.
- `build_eq_table_optimized` in lincheck. The geometric-medium trick is
  prototyped in `benches/eq_build_probe.rs` but never landed; lincheck is only
  3.9% of ST here and its cross-repo gap is partly re-attribution (see above),
  so it was not the best next move.

## The cross-repo round-1 comparison was never like-for-like

This is the most important correction in this log. The challenge repo's
round-1 figure **excludes its AB precompute**. `commit_with_round1_ab_precompute`
in its `prover.rs` runs

    rayon::join(commit_arm, precompute_ab_arm)

so `precompute_round1_ab_inner_packed_padded` — the same
`shift_reduce_inner_ab` work that is 56% of our round 1 — lands in its *commit*
bucket, and its `t.commit_s` wraps the whole join. Comparing their 314 ms
round 1 against our 1444 ms was comparing a drain against a prep-plus-drain.
Combined:

| phase | ours ST | theirs ST |
|---|---:|---:|
| commit | 1277.4 | 1796.7 |
| zc round 1 | 1444.3 | 314.1 |
| **commit + round 1** | **2721.7** | **2110.7** |

The honest gap is ~611 ms, not ~1130 ms. An earlier claim in this log that
"our commit is already ~1.4x faster than theirs" was wrong for the same
reason — they do strictly more work in that phase.

We implemented the same architecture to check whether it is a speedup or an
accounting choice, and it is the latter. The transform really is
challenge-independent (the challenge reaches round 1 only via `eq_lo_scaled`
and the convert table, both owned by the drain), a
`[x_outer][b_med][64]` buffer lets the drain consume it by borrowing with no
copy, and the result is bit-identical. But:

- **ST is a wash.** `rayon::join` is sequential on one thread, so the locality
  won by no longer interleaving AB with the C transpose and the 64 KB
  convert-table drain is spent again on a 512 MiB write plus 512 MiB read the
  interleaved version never did. Non-temporal `stnp` stores, which skip the
  read-for-ownership on that write-once surface, did not change it.
- **8T is a wash.** Both join arms compete for the same saturated pool, so
  there is no idle capacity for the overlap to fill.

Measured three ways (ST paired n=8 order-alternating with NT stores: base
53288 vs 53037, base 5/8; 8T paired n=6: 325007 vs 320655, base 4/6; ST
paired n=10 without NT stores: indistinguishable once warm). Reverted.

**Methodology note worth keeping: check the power source before measuring.**
Two grand-total runs were invalidated in one evening by power state. On low
battery, macOS caps frequencies (one base arm carried a sample ~20% low); while
fast-charging a nearly-empty battery, it is even worse -- the charger's power
budget is shared with the SoC and a paired run swung -13% to +57% per pair
with a 51% base-arm spread. Run `pmset -g batt` first: measure only on AC with
the battery above ~60%, where the charge rate has tapered. The alternating
paired design protects the sign test through slow drift (arms sit within ~80 s
of each other), and min-of-5 within a run rejects transient dips -- which is
how the round-1 results cross-validated and survived -- but end-to-end
magnitudes from a throttled window are unusable.

**Methodology note worth keeping.** An earlier version of the paired script
always ran the base arm first. Throughput declines monotonically across a run
as the machine heats (342707 → 315100 over six 8T pairs), so a fixed order
biases the comparison by roughly the size of the effect being measured. Always
alternate which arm runs first, and discard a warm-up of each arm — an
apparent +3.3% win for the hoist evaporated once both were done.

## The idea behind their `accumulate_convert` win

Worth recording even though it did not transplant, because the algebra is the
interesting part. Their C-side drain does **no table gathers at all**.

`convert[b][v] = γ^b · φ_8(v)`, and **γ = X** — the comments confirm it, the
rows are built by `mul_by_x` doubling. So `Σ_b γ^b · (bit_b)` is *literally* a
16-bit mask in the field's coefficient representation: `F128 { lo: mask }`.
Since `φ_8` is F2-linear, the per-lane C contribution decomposes over the 8 bit
positions of the byte into 8 such masks — the "eight-bank C drain" — each
accumulated with pure bit operations. The only field work left is one multiply
by `eq`, and `F128 { lo: m } * eq` is itself F2-linear in the mask's 16 bits,
so even that becomes `T_lo[m & 0xff] + T_hi[m >> 8]` from tables built **once
per prove** and shared read-only (8 MiB at their shape; building per call would
cost ~4 GiB of L1 stores).

Why it does not transplant on its own: profitability depends on data layout,
not just the algebra. Building the masks needs one lane's bytes gathered
*across* `b_med`, but `chunk_c_bytes` is `[b_med][lane]`, so extracting them
costs exactly the 16 strided loads the trick was meant to remove. Their
pipeline gets the transposed layout for free from the `Round1AbInner`
precompute pass. The AB side is handled separately by the tensor split
`eq.lo[(w << s) | u] · D^-1 == eq_top_scaled[w] · eq_bot[u]`, pre-scaling the
convert tables by `eq_top` so the inner loop is pure XOR with no per-lane
multiply, keeping `2^s` bank accumulators and applying `eq_bot` once at the
end.

Any future round-1 attempt should start from the layout, not the algebra.

## Note on the geometric trick

The three layered optimizations in `univariate_skip_optimized.rs` — geometric
small-eq + shift_reduce, geometric medium-eq + 64 KB convert-table lookups, and
D^-1 absorbed into eq_lo — are **already in this tree**; the doc headers are
byte-identical to the challenge repo's. They are inherited upstream code, not a
Yukon addition, and nothing there needs porting.

## Commit phase, ST: decomposition and the refuted butterfly rewrite (2026-08-25)

Commit-phase breakdown at m=32, single-threaded (`FLOCK_COMMIT_M=32
FLOCK_NTT_SPLIT=1 RAYON_NUM_THREADS=1`, bench `pcs_commit`):
alloc/pad ~90 ms (prefault-hidden in the real prover), **NTT 858 ms** (top 9
fused-2 layers 329 ms, deep 11 blocked layers 529 ms), **merkle 417 ms** —
merkle is at the SHA-256 silicon floor (~2.6 GB/s) and closed.

**Refuted experiment — "q-resident 3-PMULL karatsuba butterflies"** (reverted,
this commit). The premise was a misdiagnosis: `butterfly_row_pair` /
`butterfly_fused_2layer` have no aarch64 dispatch arm, so I read the portable
fallback as "scalar, 6 PMULLs through the struct/GPR interface". Wrong — the
portable butterfly is generic over `F128` ops, and `F128::mul` on aarch64
inlines `ghash_mul_binius` (gf2_128.rs:104-115, where the comment records that
M-series picked binius over karatsuba). The baseline was already running the
M1-tuned mul, register-resident after inlining.

Measured, same session, same machine state, m=32 ST:

| arm | top 9 | deep 11 | NTT total |
|---|---|---|---|
| baseline (binius via generic path) | 328.6 ms | 529.5 ms | 858 ms |
| karatsuba + q-resident kernels, GPR half-sums | 540 ms | 873 ms | 1410 ms |
| same, half-sums moved to NEON (veor+vext) | 525 ms | 835 ms | 1360 ms |

+58% regression, reproduced across two runs pre-fix and confirmed post-fix;
the GPR-vs-NEON sum was worth only ~4 points of the 58. Fewer PMULLs lost to
binius's shape: karatsuba's mid-term chain plus a per-butterfly vzip/reduce/
vunzip repack costs more than the three PMULLs it saves. This is the sixth
confirmation that re-encoding fixed work never pays on this machine (0/6), and
it extends the rule to PMULL count itself: **binius's 6-PMULL mul beats
3-PMULL karatsuba in situ on M1, not just in the latency microbench**.

Kept from the episode: the `FLOCK_COMMIT_M` bench knob and the temporary
`FLOCK_NTT_SPLIT` probe (strip the probe when the commit campaign closes).
Remaining commit-ST headroom candidates, unmeasured: aarch64 fused-4 for the
top layers (`fused4_ok` is currently x86-only; top layers are full-buffer
sweeps, so deeper fusion removes memory passes — the win category with the
best track record), and nothing else obvious; cross-tree, our commit was
already at parity or ahead.

## Cross-tree commit is measured parity, not inferred (2026-08-25)

The earlier claims ("our commit is 1.4x faster", later corrected, then
"parity or ahead") all came from in-prover bucket timers, which are
confounded: their `commit_s` wraps a `rayon::join` that includes their
round-1 AB precompute. Today: direct primitive-level A/B, m=31 packed
breakdown, ST, alternating arms, 3 pairs, same minute, both trees' own
`pcs_commit` bench (byte-identical bench code; theirs got the same
FLOCK_COMMIT_M knob temporarily and was restored after; both merkle
defaults are SHA-256; FLOCK_NO_GPU_COMMIT=1 on their arm):

| arm | NTT (3 runs) | merkle | total |
|---|---|---|---|
| ours | 418 / 421 / 419 ms | 212.8 / 212.6 / 212.4 | 692 / 692 / 690 |
| theirs (c576e68) | 422 / 437 / 420 ms | 212.2 / 213.8 / 212.2 | 690 / 706 / 684 |

Identical to within ~1% on every bucket. Their NTT source files DO differ
from ours (5 files), so this is a measured null, not shared code: whatever
they changed there is performance-neutral at this shape, and there is no
commit-phase port pool. Caveats: run on battery power with an active Zoom
call (a real >15% gap could not hide in data this tight, but treat the
third decimal as weather); and their tree has a Metal `gpu_commit.rs` path
we did not exercise — CPU-vs-CPU is parity, GPU-on is untested and out of
scope for this campaign.

## Open campaign, ST (2026-08-25): the combine port, and a full two-tree reconciliation

Protocol parity first: both trees produce byte-identical proofs at n=65536
BLAKE3 (395,919 bytes) with identical verify times, so open comparisons are
clean. ST decomposition (PCS_TRACE + LIG_PROVE_TRACE / their
FLOCK_OPEN_TIMING, same day, same machine):

| sub-phase | ours (pre) | theirs | ours (post-port) |
|---|---:|---:|---:|
| combine (b_combined fold+prime) | 94.4 | 64.4 | **78.5–79.4** |
| initial sumcheck | ~36 | 44.6 | ~36 |
| recursive commits (NTT+merkle) | 19.0 | 14.9 | 19.0 |
| induce_sumcheck_poly | 7.5 | 5.6 | 7.5 |
| ring_switch / folds / glue / OOD | ~3.7 | ~3.7 | ~3.7 |
| **open TOTAL** | **~160** | **132.7** | **~145** |

**The port (commit 6baccea): composed-table fold.** `fold_one_slot(·, T)` is
F₂-linear, so `lo ↦ fold_one_slot(lo·e_hi, T)` collapses into one composed
byte table per claim per block (x-ladder monomial walk + subset-sum
doubling), deleting the per-slot field multiply — 2·L muls — from the sweep.
Needs the coarse deferred split (eq_lo 2^15, not the balanced 2^11) so the
~4.3k-op build amortizes. Bit-identical (equivalence test + verify). Their
tree had both pieces; they never A/B'd it as a unit — it predates their
session. Prediction was −30 ms; measured −15 ms, and the sub-timer probe
explains the rest (below). MT: combine 10.9 ms at 8T (near-linear transfer).

**Corrected accounting (in-fold sub-timers + open_combine_probe micro).**
Sweeps alone: ours compose 1.2 + sweep0 22.7 + sweep1 24.5 = 48.4 ms —
EXACTLY their sweep cost (64.4 bucket − ~16 prime). The remaining bucket
difference is the tail pass: our fused prime+round-1-lookahead costs 24.5 ms
vs their plain prime ~16 ms, and the lookahead buys ~12 ms back in initial
sumcheck (36 vs 44.6). Tail+initial: ours 60.5, theirs 60.6 — **the
lookahead placement is a wash**, another instance of "moving fixed work
between buckets is not a speedup." It stays only because it is inherited
code (zero new lines to keep). Earlier probes that "showed lookahead free"
were wrong: LIG_LOOKAHEAD_DISABLE only gates the ligerito consumer, never
the combine's producer pass.

**Nulls, measured.** (1) EOR3 depth-3 fold tree: flat (79.5 vs 79.1) — LLVM
already fuses XOR pairs into EOR3 under target-cpu=native, and the sweep is
load-port bound (32 loads/slot). Not kept. (2) Fusing both claims into one
sweep (two live 64 KiB composed tables, single store, no RMW read-back):
55.6 vs 48.4 ms in the micro — the doubled gather footprint thrashes L1.
Validates the claim-sequential design note in the challenge tree.

**Residual vs theirs, and why it is closed for now**: ~6 ms structural
(recursive commits 19 vs 15, induce 7.5 vs 5.6) comes from their
sparse/windowed transpose-NTT + truncated-final-NTT machinery — thousands
of lines whose four kill-switches all measured null at this shape in their
own tree (peer-session test; several gate on their ranked 2^18 shape and
cannot fire here). Fails the bloat bar decisively at ~6 ms. The remaining
~7 ms is unattributed noise; Fiat-Shamir grinding lives inside "initial
sumcheck" and swings 1.8–8.4 ms per sample (their measurement, same bucket
convention in both trees).

Instrumentation kept (strip at campaign close): `open_combine_probe` bench +
`pcs::combine_probe` module, and the `b=` field in the combine trace line.
Conditions caveat: battery power + active Zoom all afternoon; every kept
number is an internal same-run comparison or reproduced across ≥3 samples.

**Addendum (peer measurement, same day): their truncated-final-NTT is a real
null even at its designed shape.** At the ranked 2^18 config (the exact shape
`is_ranked_induce_truncated_final_ntt_shape` pins: log_msg_cols=19,
n_queries=218), 7 paired ST samples with the production switch
`FLOCK_NO_LIG_INDUCE_TRUNCATED_NTT`: lig-prove 132.39 ms ON vs 132.79 OFF,
induce 12.23 vs 11.80 — flat. So the truncation contributes nothing anywhere;
whatever induce/commit edge their tree holds (~6 ms at our shape) rides on
the always-on sparse transpose-NTT, not this. The not-ported decision stands
with their own numbers behind it. Their run also showed the familiar
environmental spike (two trailing samples at ~2× on BOTH arms identically) —
same all-week pattern, comparison-safe, logged for the record.

## Witness gen, ST: streamed full-write builder (2026-08-25, kept)

Focused bench (`genwitness_phase`, n=65536, m=30): ours 95.9 ms best /
123 avg; their default 61.4/62.4; their scalar path (FLOCK_NO_WITGEN_SIMD=1)
73.8/76.1. So their edge decomposed as: streamed full-write + unrolled Gs
(−22 ms) then SIMD quad lockstep (−12 ms more).

Ported the first, natively: three `PackedWordWriter`s publish complete u64s
sequentially (rows are contiguous through USEFUL_BITS), killing BOTH the
driver's per-group memset and the OR path's read-modify-write on every
store; the 56-G sequence unrolls with literal state/message indices. The
one out-of-order region (out_lo, 256-bit aligned) is reserved and
overwritten at the end. Bit-identical through the whole driver (new test:
real + padding slots, both values of the prefix carry bit).

Paired alternating A/B, best-of-12 per invocation: **ST 87.1–90.0 →
64.8–68.9 ms (−24%, 3/3 disjoint ranges); 8T 18.0–18.3 → 14.4–15.8 ms
(−20%, 2/2)**. Avg variance also fell (113 → 88 ms ST). In-prove witness
bucket: 89.5 → 67.3 ms. Our streamed scalar now BEATS their scalar (73.8);
their remaining SIMD-quad edge is ~3–7 ms ST for ~400+ lines of NEON
lockstep + NT-drain + scratch-provenance machinery — fails the bloat bar.

## Re-baseline: full ST cross-tree table after the open + witness ports

Same day, same machine, GPU off on their arm (their Aug 22 binary),
n=65536: ours witness 67.3 / commit 300.8 / zerocheck 396.9 / lincheck
56.4 / open 146.7 / **total 967.4 ms (67.7k comp/s)**; theirs 30.4 /
424.8 / 217.3 / 42.2 / 132.7 / **total 847.1 ms (77.4k comp/s)**. Headline
gap **1.46× (start of day) → 1.14×**. Two buckets remain confounded, both
in their favor's appearance only: their commit carries their round-1 AB
prep (known since the zerocheck campaign) — commit+zerocheck combined is
697.7 vs 642.1 (1.09×, matching the ~10% kernel-quality verdict) — and
their in-prove witness bucket (30.4) is HALF their own focused bench
(61.4), so something (seed-pipe speculation or the rate2-codeword fusion)
moves witness work out of that bucket; under investigation. Honest
remaining real gaps: lincheck 1.34×, their witness accounting, ~9% kernel
quality in zerocheck+commit.

## Lincheck: two-stripe word-load fold reaches the load floor (2026-08-25, kept)

The prior "no reachable headroom" verdict on the lincheck fold examined
blocking strategies and table geometry; the challenge tree's newer asm
kernel wins differently: the inner loop is load-port bound, and their
kernel grabs each stripe's 8 index bytes as ONE paired load (UBFX
extracts) while folding two stripes per iteration with EOR3. Ported the
idea as intrinsics (u64 load + shift extraction, two stripes per
iteration, XOR pairs LLVM fuses to EOR3): 16 loads/stripe -> ~9,
bit-identical XOR multiset. Paired A/B: **partial_fold_z ST 40.7-40.8 ->
27.8-28.0 ms (-32%, 3/3 disjoint, at the ~28 ms computed floor); 8T
6.0-6.1 -> 4.3-4.4 ms (-28%)**. Lincheck bucket 56.4 -> 42.6 ST — parity
with theirs (42.2). One measurement-hygiene note for the record: two
paired runs were invalidated before the real one — a stale-binary
overwrite refusal (aliased interactive cp) and a stash left behind by a
failed && chain built both arms from the same source; both caught by the
disjoint-range check and a binary cmp before trusting any numbers.

## End-of-day cumulative (n=65536, m=30): 1.46x -> ~1.10x

ST: witness 71.6 / commit 300.7 / zerocheck 379.7 / lincheck 42.6 /
open 144.6 — **936 ms, 70.0k comp/s**. 8T: 152.9 ms, **428.7k comp/s**.
Vs their same-day CPU-only 847 ms ST: 1.10x, with their commit+zerocheck
bucket confounds unwound this is within the ~9% uniform-kernel-quality
band established by the SHA-256 control. Today's three kept ports:
composed-table open fold (-15 ms), streamed witness builder (-22 ms),
two-stripe lincheck fold (-13 ms) — ~50 ms ST total, all bit-identical,
all paired-decisive, all transferring to 8T.

## Round 1, final pass: the "bigish gap" was mostly an estimation error (2026-08-25)

Skip-arm probes (FLOCK_R1_SKIP_PREP / _DRAIN, since stripped; the stripe-fold
timer under FLOCK_ZC_TIMING was kept) split our round 1 at m=30 ST:
**AB prep 160 + AB drain 39 + stripe fold 28 ≈ 230 ms** — the stripe fold
already carries today's two-stripe lincheck kernel (was ~41).

The peer session then measured their prep arm directly (their
FLOCK_PHASE_TIMING probe inside the commit rayon::join, 7 ST samples,
GPU off): **~140 ms**, not the ~105–125 my commit-bucket subtraction
estimated. Corrected comparison:

| piece | ours | theirs |
|---|---:|---:|
| AB prep | 160 | ~140 (measured) |
| drain + fold | 67 | 72.5 |
| **round 1 total** | **~230** | **~213** |

So round 1 is ~7% apart, we are AHEAD on drain+fold, and the prep delta is
12.5%, not 40%. Their prep mechanism (their read): `fused_apply_one_k_fast`
— identical gather structure, but unreduced PMULL/Horner accumulation with
one fused BCAX reduction per step instead of a full reduced GF(2^8)
multiply per K-row. Arithmetic-only; our multiply tail is ~17 ms of the
160 (gathers ~90%), so the port ceiling is ~8–15 ms for ~100 lines of
kernel restructure — below the bloat bar. Their other two prep levers
(DIRECT_AB_ROWS zero-copy views, AB_COMPACT_STORE) address the
materialize-then-read-back architecture ours doesn't have: our prep is
fused into the drain and never writes the 128 MB buffer at all.

**Round-1 verdict, this time with both sides measured: closed.** The
remaining zerocheck delta decomposes as r1 arithmetic ~8–15 (priced, not
taken), r2 compact format ~7–10 (priced, not taken), tail parity.

## Official end-state grid: clean-conditions, no-timer, both trees (2026-08-25 night)

Machine quieted to just the two Claude sessions, AC power, 100% charged.
15 runs: 3 per config, interleaved between trees, bare `blake3_proof`
n=65536 (no instrumentation env), best-of-3 proves per run. Ours at HEAD;
theirs the Aug 22 binary at c576e68.

| config | ours (best, spread) | theirs (best, spread) | gap |
|---|---|---|---|
| ST CPU | 930.4 ms / 70.4k c/s (0.3%) | 843.7 ms / 77.7k c/s (0.9%) | 1.10x |
| 8T CPU | 156.1 ms / 419.8k c/s (0.4%) | 131.3 ms / 499.3k c/s (2.0%) | 1.19x |
| 8T GPU-on | — (no GPU path) | 131.0 ms / 500.3k c/s (2.3%) | 1.19x |

Findings: (1) no-timer headlines match the instrumented runs within noise —
instrumentation overhead confirmed ~zero, all bucket analyses stand;
(2) run spread at 0.3–0.9 % ST confirms every prior "day-mode"/spike
anomaly was ambient load, not code; (3) their GPU is worth nothing at 8T
(131.0 vs 131.3) — their CPU path caught up to their own Metal offload;
(4) the MT gap is 1.19x under clean conditions (1.23x on battery), and it
is scheduling (helper threads / epool P+E / allocator recycling), not
kernels — ST stands at 1.10x with every bucket at parity or priced.

## Their GPU, resolved: an ST-only, ranked-shape-only effect (2026-08-25 night)

The clean grid showed GPU-on worth ~nothing at n=65536, contradicting the
campaign-era "+10.7% ST" table. Both were right — different cells. Full
GPU value map (their Aug 22 binary, clean machine, AC, paired same-minute):

| shape / threads | GPU-on | GPU-off | GPU worth |
|---|---:|---:|---:|
| m=30 ST | 832 ms | 847 ms | +1.7% |
| m=30 8T | 131.0 ms | 131.3 ms | 0 |
| m=32 ST | 2.63 s | 2.87 s | **+9.2%** |
| m=32 8T | 413.5 ms | 406.1 ms | −1.8% |

Mechanism: their heavy offloads are shape-pinned to the ranked m=32
geometry (dormant at m=30 — Metal initializes but the cpu= telemetry shows
all threads busy doing the work), and the GPU only adds value when the CPU
is starved (ST). At 8T the CPU saturates the same memory system, the GPU
graph "finishes with 0.00 ms host wait" (their comment), and sync overhead
turns it slightly negative. In the threaded production configuration the
GPU is worth nothing on either shape; the +10.7% campaign figure was the
m=32 ST cell (reproduced tonight at +9.2%), not a general advantage.

## CORRECTION + the ranked-config picture: the GPU verdict was a config artifact (2026-08-25 late)

**Retraction.** The "their GPU is worth nothing threaded" section above was
measured with the SHA-256 merkle default — which silently fails their
ranked GPU gates (`merkle_hash == Blake3` is a hard condition on the big
offload paths). The user's suspicion that "a flag needed to be turned on"
was correct: with FLOCK_MERKLE_HASH=blake3 at m=32, their GPU is worth
**+35%** at 8T on this M1 Max. The +9.2% ST figure earlier is also
understated for the same reason.

Ranked-config grid (m=32, this machine, clean, same half-hour):

| m=32 8T | ours | theirs |
|---|---:|---:|
| SHA merkle, CPU | 433.6k c/s | 645.6k c/s |
| Blake3 merkle, CPU | 410.3k (no fast blake3-merkle kernel here) | 663.4k |
| Blake3 merkle, GPU 8T | — | 897.9k |
| Blake3 merkle, GPU 10T | — | **942.7k** |

Consequences:
1. The MT gap is SHAPE-DEPENDENT: 1.19x at m=30 but **1.49x at m=32
   CPU-vs-CPU** — their scheduling stack (seed-pipe, epool, allocator
   recycling, AB-prep overlap) is gated on the ranked m=32 geometry and
   never fired in the m=30 comparisons. Their throughput scales +29%
   from m=30 to m=32; ours +3%.
2. Full scored-config gap on this machine: 433.6k vs 942.7k = **2.17x**
   (their reported 600k/900k reproduced here as 663k CPU / 898-943k GPU).
3. The ST kernel campaign remains validly closed (1.10x, same-hash,
   same-shape); what it never measured is the ranked-config stack:
   MT scheduling at m=32, GPU offload behind the Blake3 gate, 10-thread
   epool, and a fast BLAKE3 merkle kernel. Those are the remaining
   campaign, in descending order of measured value.

**10-thread addendum (2026-08-25 late).** All-core (8P+2E) CPU-only runs:
ours m=30 414.3k (−1.3% vs 8T) / m=32 443.1k (+2.2%); theirs m=30 502.1k
(+0.6%) / m=32 595.3k (−7.8% vs their 8T). E-cores are ~worthless for
CPU-only proving on both trees — their 10T mode only pays with the GPU
overlap (943k). Best-CPU-vs-best-CPU at the ranked shape: 443.1k vs
645.6k = **1.46×**, all scheduling, not thread count. Ours reproduced to
5 digits across runs (414,339 vs 414,337 c/s).

## BLAKE3 merkle: the neon8 idea in 290 intrinsics lines (2026-08-25, kept)

Their blake3 merkle edge is a 2.6k-line generated-asm 8-wide kernel; the
mechanism is just ILP (the crate's 4-wide NEON state is latency-bound on
the G chain). Re-derived as intrinsics: two transposed 4-wide states
interleaved G-for-G, dispatched from blake3_hash_many for groups of 8,
crate path for tails, bit-identical by equivalence test.

merkle_tree ST: blake3 1.63→2.30 GB/s at 512 B leaves (+41%, now 1.08×
faster than SHA-256), 1.71→2.42 GB/s at the ranked 1 KB leaves (+42%,
parity with SHA silicon; their asm ≈2.58, i.e. within 6% for 9× fewer
lines). E2E ranked config m=32 8T blake3-merkle: 410.3k → 431.5k c/s —
the −5% blake3 penalty is erased and the ranked hash choice is now free
for this tree. LLVM handled the 32-register pressure without measurable
spill cost; the asm fallback (their .S) was not needed.

## MT campaign, night 1 (2026-08-26): one keeper, seven nulls, a map of what's left

Target (user directive): CPU-only MT within 10% of the challenge tree,
no GPU. Start: 1.19× at m=30 8T.

**Kept — AB hoist v2 (commit d963445):** prep under the commit via
rayon::join, with the two defects that nulled v1 fixed: the ab_pre buffer
comes from the scratch pool uninitialized (fresh vec![0u8] zero+fault cost
was eating the entire gain) and the join window runs on the all-core (P+E)
pool while the rest of the prove stays on P-cores. The E-cores are the
active ingredient: prep is gather/PMULL compute they can add without
stealing the DRAM bandwidth the NTT saturates. Paired 3/3 at m=30
(149.6–151.3 vs 152.5–156.3), 2/2 at m=32 (best clean pair −57 ms).
Best production number: 149.6 ms / 438.1k c/s.

**Nulls/inversions, all paired, all on this M1 Max:** P-pool-only join
(wash, third confirmation); NT stores in witness writers (~0 — M1 ignores
the stnp hint, third confirmation of the model); lincheck-stripe transpose
on E during commit (NEGATIVE: bandwidth task in a bandwidth-bound window,
commit +7 ms); FLOCK_ALLCORE combine (0); NTT fused-4 on aarch64 (+19–26%,
16 live F128s spill — the old code comment was right); two-block scalar
witgen interleave (+40% ST, GPR blowout); quad-lite SIMD witgen (state
math 4-wide, scalar packing — null even after removing 4.5k lane
extractions: the packing is the cost, not the G math).

**Decisive ablations on their tree (same day):** their witgen SIMD is
worth 2× IN-PROVE (24.9→12.1 at 8T) but their SIMD-without-elision
(16.7) ≈ our streamed scalar (16.2) — i.e. the entire remaining witness
gap is their scratch-provenance CONSTANT-REGION ELISION (−4.6 ms their
tree; ceiling probe on ours: −4.8 ms paired, degraded-machine caveat).
The focused genwitness bench measures only their scalar path (the SIMD
gate lives in their prove method), which earlier mislead this log.

Standing at checkpoint: ~1.15× at m=30 (149.6–152.4 vs 130.5–130.9
same-minute). Queued with measured ceilings: witgen constant-region
elision via pool provenance tags (−3..5 ms, ~120 lines), zerocheck
round-2 compact format (−2 ms, ~150 lines, previously priced). Those two
land ≈1.10–1.12×; anything past that is their 2.6k-line lane-wise
vectorized packing. Measurements paused: machine degraded after ~6 h of
continuous benching (witness bench 14.3→37.9 ms both arms) — resume
after cooldown per the discipline.

## MT campaign, morning session (2026-08-26): elision kept; the last 4–6 ms priced

Post-cooldown conditions verified (witness bench back to 14.2 ms from the
degraded 37.9). **Kept — witness constant-region elision (4dc8742):**
scratch-pool provenance tags, derived independently at give (from BlockR1cs)
and take (from the encoder constants), gate skipping b's MAX prefix /
reserved words and all three zero tails on a hit; any other custody clears
the tag. Paired kill-switch A/B, production config: 4/5, mean −2.5 ms,
best 146.98 ms / 445.9k c/s. Byte-identity test drives a tagged give/take
cycle vs a fresh run. (The −4.8 ms ceiling probe from last night was
degraded-machine-inflated; −2.5 is the clean value, in line with their
−4.6 on the larger constant share their layout elides.)

**Final standing, same-minute paired:** m=30 production 149.3–150.6 vs
their 130.2–130.9 = **1.146×** (campaign start 1.19×; best single run
146.98). m=32: ours 572.6 (457.8k, +5.6% on the shape since yesterday) vs
theirs 394.9 = 1.45× — the ranked-shape residual is their m=32-gated
machinery.

**The remaining ~4–6 ms at m=30, priced:** (1) compact round-2
anchor+delta (−1.8 ms @MT measured from their r2 8.9 vs ours 10.7) — ~800
lines in their tree, resurfaces through our three tuned r2/r3 kernels,
worst lines-per-ms of the campaign; their newer symbolic lookahead+cascade
(rounds 3+4 and 5+6 collapsed into earlier passes) sits on top of it,
m=32-gated there, several hundred more lines. (2) Their generated
lane-wise vectorized packing network (−2–3 ms; our quad-lite probe
confirmed the packing, not the hash math, is the cost — vectorizing it is
their 2.6k-line codegen). Both exceed the standing bloat bar; parked for
an explicit call rather than taken unilaterally.

## The m=32 shape, decomposed and probed (2026-08-26)

First m=32 bucket decomposition, both trees (8T-class, same minute):
ours witness 71.2 / commit 260.3 / zc 131.2 / lincheck 27.9 / open 93.6
(sum 584, best total 572.6); theirs 19.2 / 271.6 / 142.4 / 15.9 / 84.9
(sum 534, best total **394.9** — 139 ms of phase OVERLAP that exists only
at m=32, where their ranked stack's gates open). Notably our commit AND
zerocheck buckets are BETTER than theirs at m=32 — the kernel campaign
transferred; the 1.45× lives in witness (−52: their m==32-gated deferred
stripe + witgen hetero drains), lincheck (−12: their round-1 stripe-fold
reuse), open (−9), and the wholesale pipeline overlap.

**Probed and rejected:** deferring our lincheck stripe into the commit's
all-core join window as a third arm, gated to n_blocks_log ≥ 17 —
NEGATIVE 3/3 at m=32 (628.9–641.6 on vs 616.4–633.8 off). Their own
m==32 gate on the same idea works only inside their epool/GPU-window
architecture; re-streaming 512 MB from DRAM into our already-saturated
join window loses to the L1-fused eager transpose both at m=30 (measured
earlier) and m=32. Reverted.

m=32 conclusion: closing it means porting the pipeline architecture
(phase-overlap scheduling), not any single mechanism — same class of
decision as the r2 complex and the packing network. Parked with the rest.

**Correction to the m=32 entry above (blake arms).** The 1.45× figure
compared SHA-merkle arms — which silently disables the blake-gated half of
their ranked stack (the deferred stripe requires HashKind::Blake3, and the
witness attribution above is accordingly wrong: that gate was closed in the
SHA runs). With blake arms (their true ranked config, CPU verified via
util telemetry): theirs 920–950k c/s vs our best (SHA) 457.8k —
**~2.1× at the ranked shape**. What the blake gates open, bucket-level:
their lincheck 15.9→7.6 ms, open 84.9→21.9 ms, plus the deferred-stripe
witness path. Their dev-bench blake-CPU (950k) also exceeds their
worker-scored GPU-off number (630.8k), so ranked scoring overhead is
large; cross-methodology caution applies. Conclusion unchanged in kind
but bigger in degree: the m=32 gap is the integrated blake+m32-gated
pipeline architecture, a deliberate port-project, not a mechanism list.

**RETRACTION of the blake-arms correction above.** The 920–1036k "CPU-only"
figures were GPU-contaminated: the challenge tree's GPU merkle paths
(recursive merkle, L1 overlap) are BLAKE3-only — GPU shaders hash blake,
not SHA — and sat outside the kill-switch list used here; and the GPU-
utilization "verification" sampled only the bench's 3-minute setup window,
killing the process before any timed prove ran (worthless both arms). The
tree's owner reproduced with an airtight fresh-build kill: **their true
CPU-only ceiling at m=32+blake(+blake FS, the worker's hardcoded config)
is 638.9k c/s — consistent with their worker-scored GPU-off 630.8k**,
which is the cross-check that settles it. Their GPU at the ranked config
is worth +57% dev-warm / +43% scored.

Corrected m=32 standing: ours 457.8k (SHA; blake costs us ~4%) vs theirs
639k CPU-only = **1.40×** — in line with the original SHA-arms 1.45×, so
the earlier pipeline-architecture analysis stands as written; the "2.1×"
interlude is void. Also noted from the owner: the ranked worker hardcodes
Blake3 for BOTH merkle and Fiat–Shamir (no env); dev-bench defaults are
SHA — worth +3.7% on their tree; our FS hash config at the ranked point
is an open item. Lesson for the log: a kill-switch list is only as
airtight as the tree's owner says it is, and GPU telemetry must bracket
the timed region, not the process.

## Repo default switched to Blake3 (merkle + Fiat–Shamir), 2026-08-26

Matches the ranked worker's hardcoded config (surfaced by the peer session:
BENCHMARK_HASH = Blake3 for both, no env). HashKind::default(),
FsChallenger::new, and all embedded ligerito TOMLs flipped; SHA-256 remains
selectable per component and the cross-hash tests now exercise it as the
non-default arm. Same-binary A/B: blake-vs-sha −1% at m=30, +2.3% at m=32
— neutral-to-positive thanks to the neon8 merkle kernel. All future
default-config numbers are now at the scored hash point; historical log
entries above used SHA defaults unless marked otherwise.

## m=32 blake, CPU-vs-CPU, finally clean (2026-08-26) — and the real root cause

**The contamination mechanism was a shell bug, not a switch-coverage hole:**
`GPUOFF="A=1 B=1 ..."` as a STRING does not word-split in zsh, so
`env $GPUOFF cmd` set one garbage variable and none of the kill switches —
verified by a 1 s GPU trace showing 94–98% utilization through an "all-off"
run. The array form (used by the original official grid) and inline env
lists were always valid; every blake "GPU-off" cell used the string form.
The earlier "blake-gated GPU-merkle paths escaped the kill list" hypothesis
is withdrawn — the switches were simply never delivered. (Same zsh footgun
as the `for arm in $order` loop earlier in this campaign; now twice bitten.)

**Clean paired comparison, m=32, Blake3 merkle+FS (the ranked hash point),
CPU-only both arms, inline envs:**

| pair | ours (default config, prod) | theirs (8T, all kills) | ratio |
|---|---:|---:|---:|
| 1 | 592.3 ms / 442.6k c/s | 392.1 ms / 668.6k c/s | 1.51× |
| 2 | 629.0 ms / 416.8k c/s | 413.4 ms / 634.1k c/s | 1.52× |

Their arm agrees with the owner's airtight fresh-build (638.9k) and their
worker-scored GPU-off (630.8k) — three independent methodologies within
5%. **Verified standing at the ranked shape and hash: 1.51× CPU-only.**
(Slightly above the SHA-arms 1.40–1.45× because the blake hash point
benefits their tree ~4% and ours ~2%.) The composition of that gap is the
previously-logged one: their m=32-gated pipeline (phase overlap, deferred
stripe, round-collapsing r2 complex) plus their blake-tuned kernels.

## The "pipeline architecture" was an accounting mirage (2026-08-26)

Stage 1 of the pipeline port (Merkle leaf hashing fused into the NTT deep
pass's sub-group tasks, ~200 lines, bit-identical, kill-switched) measured
3/5 pairs, mean −0.6% at m=32 — a null (the deep pass is PMULL-compute-
bound, so hash compute doesn't ride stalls; only the codeword re-read
saving survives). REVERTED.

The null prompted re-examining the "139 ms of phase overlap" that motivated
the pipeline theory — and it dissolves: their multi-run breakdown
(BLAKE3_BREAKDOWN_RUNS) shows per-run buckets summing to ~471 ms against
~400 ms headline runs, which is exactly their prove_fast_TIMED wrapper
being ~15% slower than the untimed path (documented in week one and
forgotten). Their buckets are self-consistent within the timed path; the
"overlap" was timed-buckets-vs-untimed-best. WITHDRAWN.

**The real m=32 (blake, CPU-only) gap, timed-vs-timed, finally solid:**

| phase | ours | theirs | delta | mechanism |
|---|---:|---:|---:|---|
| witness | 71.2 | 29.5 | −41.7 | their witgen SIMD packing (2× in-prove) + scaling |
| commit(+prep) | 260.3 | 231.0 | −29.3 | window packing/alloc details, NTT+merkle parity |
| zerocheck | 131.2 | 113.6 | −17.6 | their r2 lookahead+cascade (m==32-gated) |
| lincheck | 27.9 | 16.2 | −11.7 | their round-1 stripe-fold reuse |
| open | 93.6 | 79.8 | −13.8 | ranked open machinery |

No scheduling magic — five kernel/structure ports, all previously priced,
with m=32 values now attached. The big mover is the witgen SIMD packing
network: worth only ~2–3 ms at m=30 (why it was declined) but ~40 ms at
m=32. Revised menu, m=32 value per effort: witgen packing (−40, 2.6k-line
class), r2 complex (−18, ~800 lines), lincheck stripe-reuse (−12,
unscoped), open ranked pieces (−14, partially GPU-adjacent), commit misc
(−29, undecomposed). Ceiling if all land: ~600 → ~490 vs their ~400
untimed — the last ~90 is their untimed-path leanness itself.

## The SIMD packing port that became an allocation fix (2026-08-26)

The witgen SIMD packing network — the full lane-wise design: u32-granular
writers whose pending word lives in a vector register, every push one
vsli with compile-time constants, an L1 stage per stream, vld4-deinterleave
contiguous dump — was implemented via a build.rs generator (committed
artifact = the ~200-line generator, not the 2.6k-line unrolled output) and
was bit-identical on the first full test run. It then measured a NULL
in-prove at both shapes, because the real in-prove witness cost was never
the builder: **the lincheck stripe buffer was a fresh zeroed 128–512 MB
allocation every prove**, faulted during the transpose. Pooling that one
buffer (c8d36b6): witness m=30 18.5→8.0 ms, m=32 74→33 ms, both 2/2
decisive — more than the SIMD port's entire predicted value. The quad
kernel was reverted unmerged per the bloat rule; its full design and the
generator live in the commit history via c8d36b6's message.

Their in-prove witness advantage is now INVERTED at m=30 (ours 7.9 vs
their 12.1) and mostly closed at m=32 (ours ~33 vs their timed 29.5).
Third lesson of this genre in the campaign: measure the allocation story
before porting a kernel (elision, ab_pre pooling, and now this).

## Certification grid: MT target met at m=30 (2026-08-26)

Final verification grid, both trees end-to-end untimed-best, blake3
merkle+FS both arms, CPU-only both arms (their five GPU kill switches as a
zsh array, verified by comp/s sanity), 8T, interleaved same-minute pairs
with alternating order. Ours = HEAD (feb41428 build: pool fix c8d36b6,
blake defaults fc78188); theirs = their fresh Aug-26 build (5d405d46).

| pair | ours m=30 | theirs m=30 | ratio |
|---|---:|---:|---:|
| 1 | 141.85 | 133.89 | 1.059× |
| 2 | 144.69 | 137.35 | 1.053× |
| 3 | 143.28 | 132.08 | 1.085× |

Best-vs-best **141.85 vs 132.08 = 1.074×**, every pair ≤1.09×: the
"MT within 10% of yukon" target is CERTIFIED at the prod shape
(462.0k vs 496.2k comp/s). Our best buckets: witness 7.6 / commit 64.7 /
zc 33.7 / lincheck 8.1 / open 27.9.

m=32 pairs: 542.22/397.56 = 1.364× and 541.32/405.82 = 1.334×
(483.5k vs 659.4k comp/s best) — matches the priced-menu projection;
the residual is the five parked ports (r2 complex, lincheck stripe-reuse,
open ranked, commit misc, witgen tail).

Conditions: AC power, ambient GUI load (WindowServer ~55%, Texifier ~30%)
— absolute numbers mildly inflated vs quiet-window bests (their m=32 read
384-386 in a quieter window an hour earlier; ratios are interleaved and
robust). Two grid attempts discarded first: one ran a stale pre-pool
binary left newest by the A/B stash build (witness bucket 18/68 ms gave
it away), one was launched under bash where `$ARRAY` expands to its first
element — kill switches silently dropped, their GPU came alive (1.02M
comp/s tell). Both hazards now in the protocol notes.

## Commit-bucket decomposition: the gap was absorption economics (2026-08-26)

FLOCK_COMMIT_TIMING splits our m=32 commit bucket (270ms): replicate-fill
13ms solo, NTT-from-layer-2 124ms, merkle (neon8) 54ms = **191ms of commit
work** — plus the hoisted AB prep (85ms solo) absorbed at nearly full
cost. The join window is thread-THROUGHPUT-bound: both arms scale threads
near-perfectly, so wall = (191+85 work)/(pool) = 276 predicted, 270-278
measured. Sequencing the fill before the join just moved the contention
onto the NTT (124→230 beside prep; fill 90→13): thread-work is conserved,
order can't matter. Paired A/B 3/3 old-schedule (best 558.4 vs 584.8,
noisy window); reverted with numbers in the commit message.

Peer anatomy (clean window, verified): their commit is ONE fused pipelined
pass — replicate+NTT+merkle-LEAVES prints as a single 223-298ms number
(min 223, cluster 223-242) + merkle-parents 0.1ms. Their AB-prep arm
(116.9ms) rides the join FREE because the fused pass is bandwidth-bound
and leaves idle thread-time. Their zc bucket (120.4) contains no prep
(r1 41.3 + r2 49.4 + r3+ 29.7 sums exactly). So: commit work ours 191 vs
theirs 223 — WE are ahead on work; bucket ours 270 vs theirs ~231 —
they win on absorption. The earlier "-29 commit misc" menu item is
re-attributed: it is join-contention, not kernel deficit.

Consequence: the m=32 commit lever is DELETING THREAD-WORK from the join
window, not scheduling. Two candidates: (1) fuse the replicate-fill into
the first computed NTT layer-block (read z directly per replica; deletes
the 2GB fill write + its re-read, ~25-30ms thread-work); (2) retry
NTT→merkle-leaf fusion — measured null SOLO earlier (deep pass
compute-bound) but under a thread-bound join, thread-work cuts pay even
when solo wall doesn't. Together ≈ bucket parity with their 223-231.

Our zc r1/r2/r3+ split at m=32: attempted, contaminated (user interactive
on the machine; mins r1 38.7 / r2 70.1 / r3+ 64.7 exceed the known-clean
134ms zc total — upper bounds only). Redo in a quiet window; their
r2 49.4 vs our clean r2 (TBD) prices the r2-complex port properly.

## Fill→NTT fusion: NULL, reverted (2026-08-26)

Implemented `forward_transform_interleaved_from_message`: the first fused-2
top pass copies its four input rows straight from z into the codeword rows
and butterflies them in place L1-hot, deleting the standalone 2GB
replicate pass (bit-identical: garbage-start equivalence test over 5
shapes, NTT oracle, prove/verify roundtrips ×3 circuits; kill switch
FLOCK_NO_FILL_FUSE=1). Paired same-binary A/B, 8 pairs at m=32 (ambient
noisy, user interactive): commit-bucket sign test 6-2 AGAINST fusion,
totals 4-4, min-vs-min commit 308.3 fuse vs 297.9 nofuse. Reverted;
diff preserved at scratchpad/fillfuse.patch (566 lines) and in this entry.

MODEL REFINEMENT (the valuable part): the fill's 90ms under the join was
QUEUEING, not work — a memcpy pass contributes few thread-ms, so deleting
its DRAM traffic doesn't shorten a thread-bound critical path, and the
per-row copies added overhead inside the butterfly tasks. The commit
bucket is compute-limited: NTT ~124 + merkle ~54 + prep ~85 ≈ 263 of
thread-work ≈ the measured 270-278 wall. Corollary: the planned
NTT→merkle-leaf fusion retry is ALSO downgraded — it deletes a 2GB READ
(bandwidth, not thread-work) and keeps all the hashing compute; the
earlier solo null likely stands under the join too.

Surviving m=32 commit levers, by the compute model: make PREP cheaper
(unreduced-PMULL Horner arithmetic, priced ~100 lines at m=30 and
declined at ~8-15ms; prep is 85ms at m=32 so the same idea re-prices to
an est. −20-30 bucket) — everything else in the window is already at its
measured floor (fused-4 top NEON: tried, register spill, +19-26%).

## RETRACTION: the "prep Horner" menu item was already banked (2026-08-26)

Before implementing the recommended unreduced-PMULL Horner port, archaeology
killed it: the headline win behind that name is ALREADY MERGED as e1398be
(Aug 24, "accumulate round-1 prep products unreduced (pmull + weight
split)") — the −128 ms / 8-of-8 ST result, §pmull of the writeup, one of
the seven kept changes. What the menu item actually referred to was the
RESIDUAL after that merge, priced at round-1 closure (74802fc): ~8-15 ms
ceiling at m=30 ST for ~100 lines through the hottest kernel — declined
then, and the decline stands.

Decisive at today's target shape: at m=32 8T under the join, OUR prep arm
measures 85-101 ms vs THEIR prep arm's 116.9 (their own clean sample).
Our prep is already faster than theirs in the current architecture; there
is nothing left in their tree's prep worth porting. My "−20-30 ms"
estimate from earlier today was an error — I re-priced the menu label
without checking that the mechanism behind it was already in.

Corrected m=32 menu (nothing cheap left in commit): zc r2 anchor+delta
complex (−15..30, ~800 lines), open ranked pieces (−10..20), lincheck
stripe-fold reuse (−5, unscoped). The commit bucket's remaining −40 vs
theirs is absorption economics (their bandwidth-bound fused pass hides
prep free); by the compute-limited model it has no sub-800-line lever.

### Amendment (same day, prompted by Benedikt)

"Our prep is already faster than theirs" overstated an ARM-WALL comparison
into a kernel claim. Scope is symmetric (both arms = the full
challenge-independent AB transform; the challenge-dependent drain is
outside both, pinned by Fiat-Shamir), but the contexts aren't: their 116.9
runs beside a bandwidth-bound pass (near-solo), our 85-101 beside
compute-saturating passes (contended). Scaling the clean ST closure
numbers (160 vs 140 at m=30, post-e1398be) to m=32 8T: theirs ~80 solo vs
ours ~91 — their prep kernel is likely still ~12% cheaper in isolation.
The port stays dead for the corrected reason: the reachable ~10-11 ms has
no named mechanism left (unreduced accumulation, zero-copy rows, and
dead-row-fill skip are all banked here; their BCAX fold vs our
shift+x2-byte absorb is a few vector ops in a gather-dominated kernel) —
it's the campaign's unattributed uniform-kernel-quality band.

## 8-P-core pool pinning retested at m=32: still correct (2026-08-26)

Benedikt asked whether the deliberate 8-thread (P-core) global pool is now
a slowdown at the ranked shape. Paired A/B, default-8 vs
RAYON_NUM_THREADS=10, 3 pairs m=32 (noisy window): sign 2-1 for t8,
cleanest pair dead even (574.2 vs 576.1). Per-phase on the clean samples:
zc +14 ms and lincheck +6 ms on 10T (E-core stragglers at barriers),
witness/open unchanged. The early-campaign "8 beats 10 on NTT-shaped
phases" verdict holds with today's kernels. E-cores remain harvested
selectively (join window, NTT deep pass, open combine — the phases where
they add compute to bandwidth-bound windows) and nowhere else. Pool
pinning is NOT part of the m=32 residual.

### Epool priced by the peer's own kill-switch A/B: below bar (2026-08-26)

The peer added FLOCK_NO_EPOOL to their tree (it didn't exist; I'd assumed
it from naming) and ran 3 alternating pairs at ranked CPU config: epool
is worth ~6.6% / ~26.5ms total THERE, 3/3 clean — but decomposed:
commit −10.7 (their AB-prep-hetero, the analog of the all-core join we
ALREADY have), zc −6.7, lincheck −0.6, open +3.1 (reversed, ~noise).
Portable NEW value for us = zc+lincheck ≈ −7ms for a queue primitive +
call-site conversions (~200 lines): below the bloat bar and priced-and-
parked (zc-only variant listed on the menu at −6..7). The "uniform
kernel-quality band" is NOT hidden E-core scheduling; the m=32 menu
stands: r2 complex (−15..30, ~800 lines), open ranked (−10..20),
lincheck stripe-reuse (−5), epool-zc (−6..7, ~200 lines).

### epool: NULL-TO-NEGATIVE on our kernels, reverted (2026-08-26)

Paired kill-switch A/B, 3 pairs m=32: tail 3/3 WORSE with helpers
(49.7-51.4 off vs 53.6-56.2 on), zc bucket 3/3 worse (131-149 off vs
142-171 on), r2 2/3 worse (42.2-47.4 off vs 45.3-46.7 on). Reverted (the
implementation survives in history, commit "zerocheck: bounded-tail
E-core chunk queue"). Mechanism: our r2/tail are bandwidth-heavy
streaming passes; background-QoS E-cores add DRAM pressure the P-cores
need — 4th confirmation of the bandwidth-on-bandwidth rule. Their epool
pays on THEIR kernels plausibly because anchor+delta/compact formats
made those passes compute-relative. RETEST epool after the anchor+delta
port lands.

Also learned from the off arm: our incumbent r2 is 42-47ms at m=32 —
already FASTER than their timed 49.4 (≈ parity with their scored ~43).
The r2-side of the anchor+delta port is ≈0; the port's value is
concentrated in the TAIL (ours ~50 vs their timed 29.7): the r3
table-combine, lookahead, and cascades. Task #3 rescoped accordingly.

## Cascade tail: built, verified, staged opt-in — coupled to anchor+delta (2026-08-26)

Three A/B rounds at m=32 told the full story:
1. Scalar lookahead passes (as the AG tail ships them): tail 57-59 vs
   classic 52-54 — the historical "lookahead = 13-15% regression" verdict
   reproduced, and diagnosed as KERNEL QUALITY (generic scalar F128 muls
   vs the classic path's q-resident NEON kernel), not scheduling.
2. NEON lookahead kernel + fold1 entry: tail 50.4-51.7 vs 50.5-51.4 —
   WASH, explained by pass accounting: the fold1 entry (the largest pass)
   saves no traffic and pays the full product bill.
3. Integrated r2 lookahead (r2 emits the 8 sums; all tail passes 4→1):
   tail 35.0-37.4 vs 49.6-53.1, 3/3 DISJOINT (−15) — but r2 67-80 vs
   42.6-47.8 (+24). The surcharge is the mul-count floor (8 mul_q + 8
   wide per group vs classic's 4+4 = 36 extra PMULL/group × 2^24); a
   two-sweep de-spill restructure changed nothing. Net zc ≈ +9. 

CONCLUSION: cascade and anchor+delta are COUPLED — their tree affords the
surcharge only because their compact r2 pays it from a lower base
(deferred odd-element folds + cheaper unreduced product accumulation).
Staged opt-in (FLOCK_ZC_LOOKAHEAD=1, transcript byte-identical by test);
anchor+delta port is the remaining piece, anatomy question out to the
peer: where do the odd folded values for THEIR Q products come from —
paid delta-gathers in r2, or a product formulation in anchor/delta space?

## Cascade CERTIFIED default-on via one-weight-per-group products (2026-08-26)

The peer read their actual r2 kernel and surfaced the missing formulation:
all eight lookahead products share the group's eq weight → pre-scale the
four a-rows once, every product becomes a single unreduced multiply
(52 vs 72 PMULL/group). Also confirmed from their kernel: anchor+delta is
a STORE-side cut only (their r2 pays the same gathers) — decoupled from
the cascade after all, re-priced as a separate −5..8 item (compact stores
+ r3 byte-table combine).

With w-scaling in both kernels: m=32 tail 33-37 vs classic 53-66 (4/4
disjoint; better than their timed 29.7 after wrapper deflation), r2
surcharge +12 (= its PMULL floor), zc net −8..13 (sign 3/4; −7.7 on the
clean pair). m=30: net −1.0, 2/2, best-of-run sample was a cascade arm
(142.5) — no regression at the certified shape. DEFAULT ON,
FLOCK_NO_ZC_LOOKAHEAD=1 kill switch, transcript byte-identical by test.

Projected m=32 standing: zc 134 → ~121-126 vs their timed 120 — the
zerocheck kernel gap is essentially closed. Remaining m=32 items:
anchor+delta compact stores (−5..8), open ranked (−10..20, parked for
M3/M4 per Benedikt), commit absorption economics (−40, structural).

## m=32 ST gap measured: 1.30x — the MT hypothesis refuted (2026-08-26)

Benedikt asked whether the m=32 gap is multithreading. Both trees ST
(RAYON_NUM_THREADS=1, blake3, CPU-only, same-minute): ours 3.76s vs
theirs 2.89s = 1.30x, vs 1.36x MT (ours 542.1 / theirs 398.3 same
evening, cascade engaged — after fixing ANOTHER stale-bench-binary
incident: the default-on flip was committed without rebuilding the
bench, so the first "quick pull" measured the old default; hazard rule
extended: rebuild the BENCH after every lib change, verify engagement
by a phase signature before comparing). Parallel scaling ours 6.9x vs
theirs 7.3x — MT contributes only ~0.05x. The m=32 deficit is
kernel/path-level, present on one core.

Why m=32 ST is 1.30x when m=30 ST closed at ~1.10-1.15x: their m==32
allowlist — machinery that fires only at this shape (deeper
cascade+anchor+delta forms, ranked-open pieces, SIMD witness packing
whose value concentrates at m=32; our ST witness bucket is 227ms with
no pool overlap to hide the scalar builder). Our ST buckets (cascade
on): witness 227 / commit 1277 / zc 1559 (pre-cascade ~1820) /
lincheck 126 / open 600.

### m=32 ST attribution completed — half the gap is their scored-only path (2026-08-26)

Peer's ST breakdown + our sub-phase split, prep-aligned (their AB prep
sits in their commit at ST — sequential join, a straight +551 tax; ours
sits inline in our zc/r1, ~620): like-instrumented per-phase gaps are
MODEST — witness +22, ntt+merkle +159, r1-sans-prep parity (270 vs ~260,
theirs on their slow r1 path), r2 404 vs ~350, tail 226 vs ~250 (OURS
AHEAD — the cascade), lincheck parity, open +66; our commit-with-prep is
AHEAD of theirs at ST. Sum ≈ +400 of the measured 870ms ST gap. The
other ~470ms: their own bucket sum (3.39s) vs their scored run (2.88s) —
their scored prove_fast is ~18% leaner than their instrumented
prove_fast_timed ("bypasses several ranked-specific warm-path
optimizations", r1 stripe path confirmed as one instance). Ours: ~2%.
THE BIGGEST m=32 ITEM IS NOW ENUMERABLE BY CODE-READ: the diff list
between their prove_fast and prove_fast_timed paths — requested from the
peer. Everything else we have compared is within ~15% per phase.

### Their scored-path enumeration: one real mechanism (= our adjudicated
### fill-fusion), 470ms still unattributed (2026-08-26, late)

Peer's wrapper-diff enumeration: the one substantial scored-only
mechanism is their "from-message commit" (synthesize both codeword
replicas from z during the NTT first layer, deleting a ~1GiB replica
store) — MT-gated (threads>1), and ARCHITECTURALLY IDENTICAL to our
fill-fusion (fillfuse.patch), which measured null-to-negative here:
it pays on their bandwidth-bound commit window, not on our
compute-saturated one. Items 2-6 minor/inapplicable. Their honest
bottom line: the wrapper list does NOT account for the ~470ms
scored-vs-instrumented delta; they are now diffing the timed-vs-untimed
INNER prover fns. Our hypothesis handed over: their timed figure is a
single sample vs scored best-of-3 with warm pools — if the timed inner
fn defeats warm-path buffer reuse, the 18% is allocation economics
(the genre of our two biggest wins). Same 3-untimed+1-timed harness
shape here shows only ~2% skew, so the asymmetry is in their prover fns.

### The 470ms resolved: their instrumentation artifact, nothing to port (2026-08-26, close)

Peer located both mechanisms in their timed path, with line numbers.
(1) A commit-tail-fill hook: their scored path stages round-1's C-fold
prefix in the commit join's idle tail (their commit arm finishes first —
their from-message commit shortened it; the hook fills the window while
their prep arm runs). Their timed core hardcodes the hook to None.
DOESN'T MAP HERE: our join has no idle window — our commit arm is the
long pole and rayon keeps every thread busy (the same thread-bound
economics as the absorption story); at ST it's work-reordering, not
work-deletion. (2) Their scored path's ranked_lincheck_c_reuse (stripe-
based C-derivation, gated to EXACTLY the ranked blake3 shape) — which
our tree runs in ALL paths already (StripeC). Bottom line, their words:
"not something you'd port — it only affects the fairness of MY
diagnostic function against MY real path."

FINAL m=32 ST accounting: the 870ms gap = ~400ms of modest per-phase
deltas (witness +22..53, ntt+merkle +100..160 vs old parity — worth one
re-check, open +66..146, tail OURS ahead) + ~470ms that was their
timed-path artifact inflating every per-phase comparison we made against
their instrumented numbers, of which the real scored-path content is
C-reuse (we have it) and the tail-fill staging (doesn't map). The
cross-tree kernel gap at m=32 is materially smaller than today's bucket
tables suggested; the honest scored-vs-scored per-phase decomposition
would require them instrumenting their scored path, which they may do
for their own tooling honesty.

## FIRST HONEST CROSS-TREE PER-PHASE TABLE (2026-08-26 night)

Their timed prover fixed (tail-fill hook + ranked C-reuse threaded
through; honesty check 0.4% vs scored). Both sides m=32, blake3
merkle+FS, CPU-only. Ours from tonight's clean window (their fresh
3-run averages; our matched re-run was ambient-contaminated — Chrome/
ModelCatalogAgent — so our clean-window singles stand, ±3ms):

| phase    | ours MT | theirs MT | Δ    | ours ST | theirs ST | Δ (prep-aligned) |
|----------|--------:|----------:|-----:|--------:|----------:|-----:|
| witness  |    30.2 |      29.3 |   +1 |     227 |     205.8 |  +21 |
| commit   |   ~274  |     223.6 |  +50 |    1277 |  1673 (incl prep 551) | +155 sans-prep |
| zerocheck|  ~120-128 |    99.9 |  +25 |    1559 (incl prep ~620) | 770.5 | +169 sans-prep |
| lincheck |    19.6 |      16.2 |   +4 |     126 |     117.6 |   +8 |
| open     |    96.5 |      24.9 | +72  |     600 |     122.6 | +477 |
| total    |   ~542  |     393.8 | +148 |    3760 |    2889.8 | +870 |

THE HEADLINE: OPEN is half the gap at both thread counts, and was
mis-priced all campaign by their broken instrumentation (their old timed
open read 79.8-86.6 MT / 534 ST — 3-4x inflated). Their open is FLAT
across m=30→m=32 (~132→~123 ST) while ours scales linearly (145→600):
an algorithmic difference behind their ranked/m==32 gate, not tuning.
Open anatomy requested — loop 1 of the joint phase. After open, the
residuals are commit-sans-prep +155 ST (ntt+merkle — vs the old m=31
parity result, recheck), zc-sans-prep +169 ST, witness +21 ST,
lincheck +8. Their flagged caveat: their MT witness/commit may still be
slightly pessimistic (an MT-only from-message fast path not threaded
through their timed fn).

### Open anatomy received: sufficient-statistic combine (loop 1 target)

Their m==32 open eliminates the O(L) combine sweep outright: b_combined
(size 2^(m-7) — the quantity that 4x's m=30→32) is never allocated at the
ranked shape; ring-switch round-0/1 messages derive from per-claim
FIXED-SIZE sufficient statistics — DirectFold8Factors (64-bank pair),
DirectFold4Factors (16x16 product matrix H[e,d]=Σ_h f[16h+e]·B_k[16h+d]),
DirectFold2Factors (16 products) — none scaling with L. Statistics are
computed where data is already resident (AB from lincheck's z_vec_pre —
cost lands in lincheck; C from zc r1's URM extraction — lands in zc);
we ALREADY have that relocation (our s_hat_v captures skip fold_1b_rows).
What we lack is the elimination itself. CPU-only confirmed (no gpu_commit
refs in their ring_switch). Their isolation switches exist
(FLOCK_NO_OPEN_DIRECT_{FOLD8,FOLD4,AB}) for a confirming measurement.
Port = re-derive the sufficient-statistic construction onto our combine
(our composed-table fold path still does the O(L) sweep). This is the
campaign's largest remaining item: open +72 MT / +477 ST at m=32.

### Loop-1 port plan: sufficient-statistic open (scoped, ready to build)

Inventory: OUR combine (pcs.rs compute_combined_basis_and_target — same
fn lineage as theirs) always materializes b_combined (O(L), composed-
table fold) feeding ligerito::recursive_prover_with_basis, and already
precomputes round0_prime + round1_lookahead in the combine. We have the
128-slice s_hat_v captures (AB from lincheck z_vec_pre, C from zc r1)
but NO banked (retained-coordinate) variant and no factor structs.

Port, three stages, each behind FLOCK_NO_OPEN_DIRECT (default off until
certified), transcript-identical by proof-bytes test at every stage:
1. PRODUCERS: fold8-banked s_hat_v variants of our two captures — keep
   the low 6 suffix coordinates unfolded (64 banks; their bank index =
   little-endian 6-bit retained coordinate matching build_eq(suffix[..6])).
   Their producer refs: ring_switch.rs:3205-3320 (factor assembly from
   w.s_hat_v_fold8), 2545-2558 (struct semantics: A[b,e] = transpose of
   banks, bit-major so pair-fold kernels bind coords in place; W[b,d] =
   Φ(low_eq[d]·x^b); round0 cached at construction).
2. FACTORS + COMBINE BYPASS: when both claims carry factors at the gate
   shape (L == 2^25, two RS claims, no packed-direct), skip b_combined
   entirely (their pcs.rs:1286-1307) and hand the factor pair to the
   recursive prover. γ baked into banks at construction (their
   RingSwitchBatchOutput comment).
3. BANKED SUMCHECK INTAKE: a recursive_prover entry whose rounds 0..6
   run on the A/W factor states (fold banks in place per bind; after six
   binds W collapses to the byte-map generator vector) then rejoin the
   incumbent path. Their consumer ref: ring_switch.rs:4399 (
   "sixty-four-bank intake"), fold4 fallback = 16x16 H[e,d] product
   matrix (H[e,d] = Σ_h f[16h+e]·B_k[16h+d]).
Value: open 96.5→~25 MT (−72), 600→~123 ST (−477) at m=32 if the full
mechanism transfers. Gates: shape-exact like theirs initially, widen
after certification. Estimated size: 400-700 lines, the campaign's
largest port; algebra to re-derive, not copy.

### The algebra (for the port; doc gets it on certification)

Claim k: ⟨f, B_k⟩ = t_k with B_k[i] = γ_k·Π_j eq(u_j, i_j) — RANK-1.
Split i = (e,h) ∈ {0,1}^c × {0,1}^(ℓ−c): B_k = γ_k·lo_k(e)·hi_k(h).
Statistic: G_k[e] = Σ_h f[(e,h)]·hi_k(h) ∈ F^{2^c} — computed free where
f already streams (lincheck z-fold for AB, zc r1 extraction for C; our
s_hat_v captures are the c=0 case). Target t_k = γ_k Σ_e lo_k(e)G_k[e].
Rounds r<c fold G bank-wise (G'[e'] = G[0e'] + ρ(G[1e']+G[0e'])) and
update lo_k's eq factor; messages are the usual quadratics over 2^(c−r)
terms instead of L/2^(r+1). After c binds the surviving bank collapses
into the byte-map generator (their W[b,d] = Φ(low_eq[d]·x^b)); rejoin
incumbent. Deleted: the basis' O(L) life ≈ 4L element-ops + traffic
(alloc + γ-sweep + c rounds of length-L folds). f's own folds remain
(needed downstream). Same field elements throughout ⇒ transcript
bit-identical ⇒ proof-bytes equality is the port's correctness test.
Their fold4 H[e,d] = Σ_h f[16h+e]B_k[16h+d] = the same object with the
basis low factor pre-multiplied. Conceptually: the table-vs-fold rule's
endpoint — one resident rank-1 object ⇒ contract once, never
materialize the map.

### Direct-open derivation, verified against our conventions (implementation basis)

Our ring switch: suffix S = rank-1 eq tensor over word coords (x_outer[1..]);
basis B[i] = φ(S[i]) with φ(u) = Σ_b bit_b(u)·E[b], E = build_eq(r''):
φ is F2-LINEAR (fold_b128_elems). Identities the port rests on:
1. ⟨f,B⟩ = Σ_β x^β·φ(s_hat_v[β])  (F2-linearity pulls φ out of the
   bit_β(f)-weighted sum) — consistent with sumcheck_claim =
   ⟨transpose(s_hat_v), E⟩.
2. BANKED: M_e[β] = Σ_h bit_β(f[(e,h)])·hi(h) (banked s_hat_v, low c word
   coords retained; Σ_e lo(e)·M_e = s_hat_v — the reconstruction test).
   W[b,d] = φ(lo(d)·x^b) (basis-side state; r''-dependent, built at open).
3. Round r<c message at eval point x₀: both states fold at the SAME
   challenges (A on the f-side lag, W on the basis-side lag — same ρ);
   g_r(x₀) = Σ_{e'} Σ_β x^β · Σ_b bit_b(A-partial(x₀,e')[β])·W-partial(x₀,e')[b]
   — O(128·128·2^(c-r)) XOR-dominated, sub-ms; NO O(L) touch.
4. Exit: after all c binds, W's sole bank = the byte-map generator vector
   G[b] = Σ_e lag(ρ)(e)·W[b,e]; b¹[h] = ψ_G(hi(h)) materialized at
   2^(ℓ-c) via the existing composed-byte-table machinery. f folds through
   the same rounds with existing kernels (f-only variant of the lane fold).
Producers are SMALL: AB banks from lincheck's z_vec (2^k_log elements —
banked s_hat_v_from_z_vec, trivial); C banks from the zc capture
analogously. Deleted O(L) work per claim: fold_b128_elems (rs_eq_ind),
the b_combined build, and its c rounds of folds (~4L total).
Implementation order: (1) ring_switch banked structs + reference
producers + W + reconstruction/claim identity tests; (2) round-message +
state-fold fns, oracle-tested vs the dense SumcheckProver at m=13-16;
(3) ligerito lane-fold intake (f-only folds + direct messages + b¹ exit);
(4) pcs gate + banked captures from z_vec/zc; kill switch
FLOCK_NO_OPEN_DIRECT; proof-bytes identity test at every stage.

### Direct open: wired end-to-end, proof-bytes identical (loop-1 units 1-3)

The basis-free opening is complete behind FLOCK_OPEN_DIRECT=1:
prove_batched emits banked claim bundles (RsEqInd::Direct; flat s_hat_v
reconstructed from banks for the transcript), the combine skips
rs_eq_ind + b_combined entirely, and recursive_prover_direct runs the L0
lane folds with f-only array folds + banked round messages, materializes
the residual basis from the exit generators at the level-1 boundary, and
rejoins the incumbent flow. One wiring bug shaken out (banked messages
reached the challenger but not the proof's sumcheck transcript;
prefixed). Also found pre-existing rot in the ignored hand-rolled-config
pcs roundtrip test (fails on HEAD; unrelated).

Verified: proof-bytes identity through the full pcs stack at m=22
(production config shape) + verify; production-glue bundle test; oracle
tests at 4 shapes incl. (15,6); full prove/verify roundtrips on all
three circuits with the direct path ON; 347 core tests green.

REMAINING before the m=32 A/B (unit 4, producers): the v1 gate builds
banks with the serial reference scan — unusable at m=32. Plumb
banked_s_hat_v_from_z_vec (exists, tested) for the AB claim from
prover.rs's z_vec_pre, and generalize the zc r1 s_hat_v_c capture to its
banked form for the C claim. Target: open 96.5→~25 MT / 600→~123 ST.

### Direct open: producers landed, port complete; certification pending
### a clean window (2026-08-26, late night)

Unit 4 (producers) landed: AB banks from lincheck's z_vec_pre
(banked_s_hat_v_from_z_vec, carried through ProveCore), C banks captured
free inside the zc stripe fold (round1_c_banks_from_stripe_with_banked —
same outer partial fold, middle fold stopped c dims early, flat banks
bit-identical by test), plumbed via open_batch_..._banked with per-claim
fallback to the reference scan. Full validation green (proof-bytes
identity, banked-C equivalence, roundtrips with FLOCK_OPEN_DIRECT=1,
348 core tests).

First m=32 A/B attempts hit a heavily contaminated window (bests
836ms-1.3s vs clean 541; Chrome bursts): magnitudes unusable. Signal
that survives: sign 3/3 for direct on usable pairs, and one direct open
sample at 80.8ms — BELOW the clean dense floor (96.5), proving
engagement with real producers and sub-incumbent cost. Certification
A/B (target: open 96.5→~25 MT, 600→~123 ST) deferred to a quiet window;
the gate stays opt-in (FLOCK_OPEN_DIRECT=1) until then.

## DIRECT OPEN CERTIFIED, DEFAULT ON — campaign record (2026-08-26 23:09)

Quiet window (the self-arming gate fired as the machine went idle).
3 MT pairs + 1 ST pair, alternating, kill-switched same binary:

| | direct | dense | verdict |
|---|---|---|---|
| MT open | 59.5-72.1 | 94.3-100.2 | 3/3 disjoint, −33 avg |
| MT zc | 121.6-126.7 | 127.0-133.0 | no regression (3/3 better) |
| MT best | 484.3-518.4 | 527.4-543.1 | 3/3, avg −32.5 |
| ST open | 210.0 | 614.2 | −404 |
| ST best | 3.37s | 3.74s | −370 |

**Best prove 484.25 ms / 541,336 c/s — the campaign's best m=32 reading.**
Headline vs their scored ~397: ≈1.22×, from 1.36× this morning (the
cascade + direct open together). DEFAULT ON (FLOCK_NO_OPEN_DIRECT=1
kills). The residual open gap (+39 MT / +87 ST vs their honest 24.9/123)
= the unported ranked-open extras (their exact-shape-gated truncated
final NTT + lazy OOD eq) — parked per Benedikt for the M3/M4 revisit,
now cleanly the next item if unparked. Loop-1 of the joint phase is
CLOSED: mechanism identified from their honest numbers, algebra
re-derived, ported in four verified units, certified same-day.

## Open residual, loop 2 opened: our own floor first (2026-08-26, midnight)

Peer's honest ST decomposition of their open (113-126 total): ring-switch
tail 0.5 / combine 0 / initial sumcheck 38-51 / recursive commits ~58 /
induce 11.2 / OOD 0.8. Verdict: their commits are NOT near-zero (parity
with ours per-thread); their edge concentrates in the small stages —
which on OUR side were self-inflicted. Three local fixes, all
transcript-identical (identity test green each step):
1. W-state via fold byte table (was a 128-bit scan/element): tail 8.7→6.6
2. COMPOSED f-fold — unique to the direct path: messages never touch f,
   so all initial_k challenges bind in one 2^k→1 pass (~2L→~1.03L
   traffic): initial sumcheck 26.4→19.6
3. Parallelized bank transposes + reconstruction: tail 6.6→1.4 (their
   floor: 0.5)
OPEN: 64.1 (certified) → 47.6 ms at m=32 8T. Remaining deltas vs their
MT-scaled ~25: initial ~−10 (drill the 19.6), induce −4.5 (their
truncated-final-NTT — exact, requires log_inv_rate==1, fused round-msg
variant), OOD −2.8 (lazy split-eq + glue fused into the next fold),
commits ≈ parity. Their two extras' full anatomy is on file (their
message, ligerito.rs refs).

### Micro-fixes PAIRED-CERTIFIED (correcting the single-sample claim above)

Benedikt asked whether 64.1→47.6 was measured or estimated — it was
single-sample traces across different windows. Proper two-binary paired
A/B (3 pairs, alternating, head rebuilt after): open 45.5-50.4 vs
59.7-61.3 — 3/3 DISJOINT, −12.1 avg (−20%); totals 492.5-511.1 vs
516.0-538.2, 3/3. The honest certified delta for the three micro-fixes
is −12 (not −16.5: the pre-fix arm reads 60.4 in this window, not the
earlier quiet-window 64.1). Best this window: 492.53 ms / 532.2k c/s.

### Initial-sumcheck drill: half is PoW grinding at its floor (2026-08-27)

Fine timers on the direct L0: fold grinds 9.3-11.8ms (block-parallel
lowest-nonce already — protocol security work, high-variance, NOT a
kernel target; explains both trees' initial-sumcheck noise), composed
f-fold 4.5-4.7, b1 5.7-6.4, boundary 0.2-0.6. b1 split-fold variant
tried: SLOWER (9.2-9.4; per-slot mul > saved tensor build at 2^20),
reverted. Grind-adjusted kernel content of our open ≈ 36ms vs their
grind-adjusted ~15-20: remaining targets = induce truncated-final-NTT
(−4.5) and lazy OOD (−2.8), then commits/misc at parity.

### FLOCK_NO_GRIND: grinding removed from the measurement protocol (2026-08-27)

Benedikt: grind time is nonce-search luck (9.3-11.8ms MT of the open at
m=32 L0 alone) and adds variance paired A/B can't cancel. Both trees now
carry the same knob: `FLOCK_NO_GRIND=1` coerces grinding bits to 0 in
`grind_pow` (nonce 0, zero hashing; ~5 lines, LazyLock env check).
Default unchanged (grind ON). Deliberately NOT mirrored in the verifier
— grind-free proofs FAIL verification (the bench's final verify panics
after timings print; scripts tolerate the exit). Yukon confirmed with
Benedikt and added the identical knob. ALL grind-free numbers are a new
baseline family — not comparable to anything certified earlier.

### Port wave 3: truncated-final-NTT induce + lazy OOD (2026-08-27)

Both peer-anatomy items, both transcript-identical (dense-equality unit
tests + full lib suite + m=22 proof-identity green):

1. **Fused low-half final-3 NTT tail** (their `..._fused_final_3layer_
   low_half`): `induce_sumcheck_poly_via_ntt` computed the full 2^21
   transpose then `truncate(n)` — at rate 1/2 the retained half never
   needs the last three layers' full sweeps. Fused strided kernel
   (8-gather, 3 butterfly levels in registers, 4 low writes, in place
   via split_at_mut): last layer's kept output is a plain XOR, ~6n
   traffic → 1.5n. Gated `log_inv_rate == 1` inside the sparse
   transpose. Trace attribution: induce 6.8-7.6 → 6.0-7.2ms MT (~−0.8,
   consistent with the traffic math; peer's −4.5 bundled other diffs).
2. **Lazy OOD** (their `introduce_new_ood_factorized`/`glue_factorized_
   ood` analog): `introduce_ood` splits eq(z,·) = eq_lo ⊗ eq_hi
   (build_eq_table is LSB-first) — round msg + eval read only f, the
   2^n table is never built; glue defers (α·eq_hi, eq_lo) and the next
   basis glue drains all samples fused in its one read-modify-write
   pass (fold paths carry a flush no-op as insurance). Trace: OOD
   samples (5) 2.6-3.0 → 0.4-1.0ms MT; basis glue +0.1-0.6 (the
   drain). Net ≈ −2.

**Wave 3 CERTIFIED** (paired grind-free A/B, quiet-window checked): open
bucket 40.8-41.3 → 36.8-37.8 ms m=32 8T, **−9%, 3/3 disjoint**; totals
2/3 (a ~4ms effect inside ±10ms window noise — bucket is the signal).
Committed d87cb4b. Window hazard logged: three consecutive windows read
528-556 both arms with no foreign process >50% CPU — thermal inflation
from hours of sustained benching; grind-free family best remains the
first quiet run (469.39 ms / 558.5k c/s). Remaining open residual ~37
vs their grind-adjusted ~15-20; unattributed ~20 (recursive
commits/opens inside the bucket) is the next drill.

### Open drill wave 4: boundary join + blocked transpose (2026-08-27 overnight)

Peer's grind-free MT decomposition (their bench, 4 samples): open 21.46
total = initial 5.19 / recursive commits 9.88 / induce 3.17 / rest ~2 —
so the "15-20" carry was right, and our residual concentrates in three
buckets. Two fixes, certified together (paired A/B: open bucket sign
3/3, −2.8/−3.2/−8.0, avg −4.7ms; totals 2/3 — ~4ms effect in window
noise; pair 3's window degraded mid-run, pairing absorbed it):
1. **Boundary join**: the composed f-fold (DRAM-bound, full-witness
   read) and the residual-b1 build (L1-gather-bound byte-table folds)
   ran back-to-back; they're data-independent at the boundary →
   rayon::join (compute-under-bandwidth, the AB-hoist pattern).
   Initial sumcheck 11.3 → 8.5-9.8 (their 5.19 — asked how).
2. **Blocked dense transpose** in the sparse induce: the dense remainder
   ran one full 32MB sweep per layer (10 layers); layers whose blocks
   fit 2MB now run together over L2-resident chunks (blocks nest, so a
   chunk at the run's lowest layer contains whole blocks of every
   higher layer): ~640MB traffic → ~130MB. Induce 6.6-8.5 → 5.4-6.7
   (their 3.17). Equality test extended with a rate-1/4 blocked shape.
NULL: twiddle-width census (hypothesis: fused-2 loses the 3-PMULL
half-width path on mid layers) — only the 2 deepest layers are
half-width at dim 18/24; the static gate is already right. No change.
Recursive commits 14.3-15.5 vs their 9.88: our L1 NTT deep is at the
PMULL floor and merkle at the blake3 floor per component math —
anatomy question sent to peer (different decomposition / overlap /
config?). Open now ~30-33 vs their 21.5.

### Recursive fill fusion CERTIFIED (2026-08-27 overnight)

Peer per-level anatomy: their recursive commits skip the replicate fill
— first NTT pass writes the codeword straight from the compact message
(fill 0.00, their L1 ntt 3.1-3.4). Our tree already HAD the mechanism:
forward_transform_interleaved_from_message, built for the main commit,
adjudicated NULL there (compute-limited join window, 6-2 against), and
reverted into scratchpad/fillfuse.patch. Resurrected the NTT-side
machinery verbatim and switched ONLY ligero_commit (recursive path) to
it — the main commit stays on replicate (null verdict stands; the
recursive commits are standalone + bandwidth-bound, which is why it
pays here). L1 fill 0.5→0.00, all-in NTT 5.2-6.9→5.0-5.2; recursive
commits 14.3-15.5→12.9-13.3 (their 9.88; residual = their radix-8
kernel shape, parked). PAIRED A/B: open sign 3/3 (−2.35/−1.60/−0.56,
avg −1.5), totals 3/3. FLOCK_NO_FILL_FUSE=1 kills (same-binary A/B).

### Boundary FUSION (join → single sweep) — phase-certified (2026-08-27)

Read their materialize_direct_fold8 directly (Benedikt owns both trees):
one witness sweep produces f1 AND b1 AND the round-0 message — the b1
gathers + factored eq products run in the witness stream's stall slots,
fold accumulates unreduced (fold_banked_slot). Ported the idea as
fold_boundary_fused_par: replaces the rayon::join; suffix eq kept
factored (eq_lo⊗eq_hi, never built); lag fold via mul_unreduced + one
reduce. The standalone split-fold null INVERTS under fusion (logged
where the null was recorded). Boundary pair 7.3-8.4 → 5.5-5.7; initial
sumcheck 8.5-9.8 → 5.9-6.0 (their 5.19). CERTIFICATION: bucket-level
A/B inconclusive (−0.2/+10.1 outlier/0.0 in a noisy window — a ~2ms
effect under ±5ms noise); per-phase paired A/B (same binaries, same
window, init_min per run) 3/3 DISJOINT: 5.85/6.04/6.59 vs
7.52/8.40/8.13, avg −1.9. Proof-identity + full suite green.

NULL: pooled densify buffer in the induce (scratch take + parallel zero
+ right-sized copy-out vs fresh vec![ZERO; 2^21]) — induce 6.1-7.1 vs
5.4-6.7 baseline windows, the 16MB copy-out eats the fault savings at
32MB scale. The allocation rule pays at witness scale (128-512MB), not
here. Reverted.

### JOINT GRIND-FREE CROSS-TREE TABLE (2026-08-27 ~05:00, m=32 blake CPU-only 8T)

First fully-honest table of the campaign: both trees grind-free
(FLOCK_NO_GRIND=1), GPU off, strict slot alternation in one window
(theirs #N then ours #N), both sides' timed paths previously fixed.

  pair1: theirs 392.44 ms / 667,992 c/s   ours 477.93 / 548,497   1.218x
  pair2: theirs 388.75 ms / 674,319 c/s   ours 469.02 / 558,920   1.207x
  pair3: theirs 391.35 ms / 669,849 c/s   ours 475.33 / 551,497   1.215x
  best-vs-best: 469.02 vs 388.75 = 1.206x

Per-phase mins across slots — open: theirs 20.6-21.7 vs ours 29.7-31.1
(initial ~parity 5.05-5.19 vs 5.73-5.87; recursive commits 9.5-9.9 vs
11.8-12.5 = their radix-8 kernel shape; induce 2.5-2.6 vs 5.05-5.24 =
their fused densify/arena/intro-msg mechanisms, each sub-ms, below our
bloat bar — anatomy on file). Their other buckets: witness 29.1-29.9 /
commit 223.7-235.8 / zc 101.0-103.5 / lincheck 16.1-17.4. Campaign
arc: 1.36x (yesterday morning, with-grind) → 1.206x honest grind-free;
whole campaign vs origin baseline ≈ 723.9 → 469.0 ms 8T (1.54x) /
4781 → ~3370 ms ST.

### Full grind-free breakdown snapshot @ af90ba1 (2026-08-27 morning, m=32 blake 2^18)

1T: 3312.30 total (78,269 c/s) = witness 244.1 / commit 1267.8 / zc-r1
895.9 / zc-r2 408.0 / zc-r3+ 226.9 / lincheck 124.6 / open 145.0.
8T: 472.76 total (553,362 c/s) = witness 30.1 / commit 273.4 / zc-r1
35.1 / zc-r2 54.3 / zc-r3+ 31.9 / lincheck 18.8 / open 29.2.
Open sub-phases 8T: boundary fused pair 5.5-5.6, initial SC 6.1-6.2,
recursive commits 13.2-14.0 (L1 = ntt 5.2 + merkle 3.1-3.3, L2 ~2.3,
tail ~1.2), induce 5.9-7.4, intro+glue 0.7, OOD(5) 0.4, opens 0.2.
Open ST: 145 = commits 68.1 / initial 41.9 (pair 40.2) / induce 24.4 /
misc ~5. Campaign totals vs origin baseline: ST 4781→3312 (1.44x), 8T
723.9→472.8 (1.53x). vs peer grind-free same morning: witness parity,
commit +38..50, zc +18..20, open +7.5..8.6, lincheck +1.4..2.7.

### Commit attack, experiment 1 NULL: prep ∥ merkle staging (2026-08-27)

Hypothesis: pair the PMULL prep with the Merkle stage (BLAKE3 =
integer-SIMD) instead of the PMULL NTT — disjoint execution ports.
REFUTED decisively: staged join wall 306-327 vs 273 baseline; the
merkle arm inflates 55 → 150-157 beside prep (~fully additive). On M1
BLAKE3 and PMULL both issue on the NEON pipes — there is no disjoint
port pool, the compute-additive window model holds. Scheduling reshuffle
reverted (prover.rs); the commit.rs encode/merkle stage split kept
(used by experiment 2). Also measured: NTT truly solo = 158 vs 128-145
in-join windows — arm-wall readings continue to be context-dependent.

### Commit attack, experiment 2 NULL: leaf hashing fused into the NTT deep pass

Built the peer-shape fusion (per-sub-group hook in the parallel
interleaved NTT; each 2MB sub-group's leaves hashed cache-hot in the
task that finished its butterflies; merkle's 1GB codeword re-read
deleted; bit-identical root/tree/codeword by test at two shapes).
MEASURED NULL in a cooled window (kill-switch pairs, min-of-run): join
wall fused 269.5-281.5 vs staged 266.1-276.3 (sign 1/3), commit bucket
2/3, totals 1/3 — all inside ±6ms noise. The old "deletes a read, not
compute" pricing is now a measured verdict; no prep absorption
materialized either (the M1 compute-additive window model holds — their
absorption comes with a pass that is SLOWER alone by ~13ms, and
adopting slower-alone shapes only pays if absorption exceeds the
slowdown, which our kernels' shape does not exhibit). Reverted; patch
preserved in scratchpad/leaffuse.patch. COMMIT VERDICT so far: our
kernels win solo (fill 22.9 + NTT 844.7 + merkle 413 ≈ 1280 ST — NTT
repeatable to 0.13ms), the ~40ms MT bucket gap is their prep riding a
stall-rich fused pass; closing it means adopting their whole commit
architecture with uncertain net on our faster kernels. ST cross-tree
comparison pending (peer running grind-free ST decomposition).

### Commit attack, experiments 3+4 NULL: NEON fused-4 and radix-8 fused-3 top

The "aarch64 fused-4 top layers" menu item is now measured and DEAD,
with the model corrected in the process:
- NEON fused-4 (16-point rows, paired-PMULL vec2 muls): top layers
  316-323 → 794-860 ST (2.5× WORSE) — sixteen ~32MB-strided streams per
  row group break the M1 prefetcher.
- Radix-8 fused-3 (8 streams, the challenge tree's choice): 586-601 ST
  — still 1.9× worse than fused-2.
- MODEL CORRECTION: the ST top was never at the bandwidth floor. Per
  fused-2 pass: ~6.7e7 muls ≈ 67-98ms compute vs 79ms measured — the
  top layers sit AT the compute≈bandwidth balance point single-threaded,
  so wider fusion trades traffic nobody is waiting on for access
  patterns that stall. (At 8T the top IS bandwidth-bound, but the MT
  ceiling is ~10-15ms and both wider kernels regress compute.)
Both reverted. COMMIT ATTACK VERDICT: kernels are near-parity ST (our
encode+merkle 1280 vs their 1207, −6%; merkle ±6; prep ~parity); the
~40ms MT bucket gap is their single-pass commit architecture absorbing
prep in its stalls, and four replication attempts (prep∥merkle staging,
leaf fusion, fused-4, fused-3) all measured null-to-negative. Remaining
option = porting their full streaming radix-8 replicate+NTT+leaf pass:
large rewrite of the tuned NTT, uncertain net (their pass is faster ST
by ~73 but slower MT-solo by ~13 than our staged pipeline) — priced for
Benedikt's call.

### Streaming commit port (their full architecture) — MEASURED UNSUCCESSFUL, reverted

Benedikt-directed full port (no kill switch; his correct note: my
"slower MT-solo" claim was bad accounting — their 235 was prep-contended
too). Built faithfully from their source (commit.rs:489-830): deep-pass
NTT publishes ~1MiB leaf jobs (leaf hashing + aligned local parent
subtrees, hashed hot) into a bounded channel drained by utility-QoS
helper threads (E-cores); queue-full → inline; tail drained on the main
pool; shared top after (merkle top 0.03ms — locals worked). Join moved
to the P-pool when the pipeline engages (E-cores reserved for helpers).
Bit-identical (root/tree/codeword equality test at 2 shapes; suite
350 green). MEASURED, same window as a staged baseline re-run
(287-294): shallow queue (cap 4) 294-307 — inline fallback
re-serializes leaves onto P mid-transform; deep queue (no inline)
350-354 — codeword cold by drain time, staged merkle's DRAM read comes
back plus overhead. Intra-window structure: ntt+leaves 194-239 vs
staged ntt+merkle 183 — the pipeline never beats staged even inside
its own window. WHY THEIRS WINS AND OURS CAN'T (this hardware): the
architecture monetizes E-core silicon for leaf hashing; their host has
4 E-cores, ours 2 — and our all-core join ALREADY monetizes those 2
E-cores for prep compute (certified AB-hoist v2). Switching the same 2
E-cores from prep-absorption to leaf-absorption is conserved-compute
zero-sum, minus channel/QoS overhead. Patch preserved:
scratchpad/streaming_commit.patch (542 lines incl. equality test).
COMMIT PHASE FINAL VERDICT: closed-structural on M1 Max — kernels ST
near-parity, all five architecture/scheduling experiments
null-to-negative; the ~40ms MT bucket gap is E-core-count-bound, not
code-bound.

### STREAMING COMMIT CERTIFIED — the earlier verdict was wrong (2026-08-27)

Benedikt challenged the "E-core-count-bound" conclusion: their 1.2x was
measured on THIS machine. He was right — the "4-E-core host" was a
comment in their code about the ranked contest hardware, misread as the
measurement box. The real missing piece was visible in our own trace:
our pipelined ntt+leaves (194-205) was FASTER than theirs (235), but
our prep had already burned its overlap window beside a 100ms
standalone replicate-fill that their architecture doesn't have — their
from-message top eliminates the fill, so prep rides the ENTIRE window.
The main-commit fill-fusion null (staged shape: "deletes a read, not
compute") does NOT transfer to the pipeline shape, where the fill
fusion is load-bearing: it buys the prep its hiding window.

v2 = pipeline + from-message fill fusion (hook threaded through
forward_transform_interleaved_from_message_with): join wall becomes ONE
line (ntt+leaves, prep inside, merkle top 0.03ms). PAIRED A/B
(two-binary, 8-min cooldown, 3 pairs): join wall 252.5/254.7/252.8 vs
262.6/260.5/261.1 — 3/3 DISJOINT, avg −7.4; totals 482.5/463.1/456.9
vs 502.8/479.8/489.6 — 3/3 DISJOINT, avg −23. Bit-identical
(root/tree/codeword equality at 2 shapes; proof-identity; suite 350).
NEW CAMPAIGN RECORD: 456.95 ms / 573,681 c/s ≈ 1.175x vs their 388.75.
"Commit closed-structural" is RETRACTED; the architecture works on 2
E-cores once the fill is fused. Lesson for the log: a null verdict is
scoped to the architecture it was measured in.

### Ranked radix-8 top with E-core hetero tiles CERTIFIED (2026-08-27)

Second half of the streaming-commit architecture (their split-ranked
top, read from their ntt source): layers 1..9 in THREE radix-8 passes
replacing four fused-2 sweeps — layer 1 fused with the fill via the
dual-destination from-message kernel (one 512MB witness read → BOTH
replica blocks; block 0 on the XOR-only zero-root chain; outputs staged
in 8KB L1 tiles, emitted as sequential stnp bursts — the staging is
what my earlier fused-3 lacked: per-lane scatter stores defeat the
streaming-store detector and pay RFO on the fresh 1GB destination,
their note records −4% for that exact mistake); layers 4/7 in-place
radix-8 with q-resident butterflies (field lib's mul_q, no GPR
crossings; block 0 zero-root). All three passes distribute 128-row
tiles over the rayon pool AND two utility-QoS E-threads
(run_hetero_chunks, one atomic counter) — the E-cores assist the top,
then switch to leaf hashing when the deep pass publishes. Gated to the
rate-1/2 shape with n_top≤10 guard (huge shapes keep fused-2; skipping
layers 10..n_top would corrupt). Bit-identical: from_message equality
(shape (16,32,2) runs the ranked path), pipelined-commit
root/tree/codeword test, full suite. PAIRED A/B (cooled, 3 pairs):
commit join wall 256.6/259.4/266.7 vs 273.1/273.2/275.5 — 3/3
DISJOINT, avg −13.1; commit bucket 3/3; totals 2/3 (pair-1 new-arm
total was a Chrome-window outlier, its commit wall still won).
Commit window now ~256-267 warm ≈ ~240s cool-window basis vs their
224-236. Fused-3 verdict CORRECTED: the kernel shape (q-resident +
staged NT stores), not the radix, was what failed before.

### JOINT TABLE 2 (2026-08-27 afternoon): ratio ~1.2, UNCHANGED within noise

  pair1: theirs 389.83 (672,460)   ours 467.88 (560,278)   1.200x
  pair2: theirs 378.15 (693,226)   ours 465.22 (563,485)   1.230x
  pair3: theirs 402.44 (651,389)   ours 485.22 (540,253)   1.206x
  best-vs-best: 465.22 vs 378.15 = 1.230x  (morning: 1.206x)

Their spread alone is 6% (378-402, commit 219-241) with NO code changes
on their side (verified: their last commit Aug 18, same 7-file diff) —
the per-pair ratio noise floor is ±3%. HONEST RECONCILIATION of today's
afternoon commit work: bucket-level cross-window truth = commit 273.4
(morning calm) → ~263 (best current samples) ≈ −11, NOT the −36 the
stacked within-window certs implied. The streaming-commit A/B's totals
line (−23, 3/3) overstated its bucket line (−7, 2/3) — the bucket was
the truthful number, and part of the totals margin was window texture
that survived alternation. Ranked top's −13 wall partially overlaps in
the bucket view. Rule reinforced: certify on the phase/bucket the
mechanism lives in; totals inherit too much window. Current calm-basis
gap decomposition: commit +35-40 (ours ~263 vs theirs 219-241 — their
remaining edge = deep pair fusion + whole-prove epool depth), zc +20
(r2 anchor+delta priced −5..8/800 lines), open +7, lincheck +3,
witness +1. Ours slot-3 zc bucket read 174.2 in its traced run —
polluted sample, excluded from analysis.

### Whole-prove epool pass: 1 keep, 2 reverts (2026-08-27 evening)

Three compute-bound sites hetero-tiled (rayon pool + 2 utility-QoS
E-threads via run_hetero_chunks[_stateful]; streaming zc r2/tail and
lincheck excluded per the standing bandwidth-on-bandwidth null). Cooled
paired A/B, per-site buckets:
- KEEP zc r1 gather drain: r1 line 34.38-34.67 vs 34.98-35.14, 3/3
  disjoint, −0.53ms. The fold/reduce became run_hetero_chunks_stateful;
  values order-free.
- REVERT open boundary pass + recursive-commit merkle leaves: open
  bucket 1/3 with negative lean (+3.0/−1.1/+0.5) — per-call helper
  spawn/QoS churn exceeds what 2 E-cores add on 3-6ms passes. The
  epool pattern needs passes ≥ ~30ms to amortize on this host.
Cool-window state: HEAD (old arm) 461.1-461.8; new arm best 458.64 ms
/ 571,574 c/s — NEW RECORD. run_hetero_chunks gained the ST guard
(inline at 1 thread, parity preserved) and a stateful variant.
### zc COMPACT VARIANT K ported (r2..r5 in two passes) — CERTIFIED, DEFAULT ON

Full port of their compact round-2 architecture (read from source;
~700 lines): (1) PRODUCER uni_skip_fold_round23_compact_padded — one
sweep over the packed rows yields the 48B/pair compact state (anchor =
fold(row0); delta = packed-byte XOR, still in the bit domain), round
2's wire message, AND round 3's as six deferred quadratic coefficients:
products parity-split over pair index (eq2(2y+1) = r1·eq3(y)), one
odd-lane weight per group, κ=(1+r1)/r1 rebalance, single r1^-1 unscale.
Replaces our two-sweep lookahead r2 (fold+store then L2 re-read) with
ONE sweep and 25% smaller stores. (2) DEGENERATE-B fast path (theirs):
all-ones packed b rows skip their 16 table lookups (fold(0xFF..) is a
per-table constant), delta_b = 0, and the G(∞)/e/o products carry
provably-zero factors — value-preserving, targeted-tested with b ≡ 1
half-domains. (3) CONSUMER fold2_compact_round45_into — binds ρ1 AND
ρ2 through two λ-scaled byte tables (λ1 = ρ1(1+ρ2), λ3 = ρ1ρ2; the
anchors need only an ordinary ρ2 fold), emits round 4's message,
materializes the quarter level, and defers round 5's message in ρ3 by
the same parity trick — after which OUR cascade tail resumes its
ordinary cadence (entry i=3, ρ3/ρ4 deferred; the loop invariant holds
unchanged). NEON arms throughout (q-resident folds, 8 wide unreduced
accumulators, u64-pair delta stores); scalar references kept as the
non-aarch64 arm and oracle. Transcript-identical by two tests (random
+ degen-b targeted) at m=16/18; full suite green. Gates: n_mlv ≥ 7,
n_out ≥ 1024, r[k_skip+1] ≠ 0, r[k_skip+3] ≠ 0 (parity unscalings;
each fails w.p. 2^-128) — fallback = incumbent cascade route.

CERTIFICATION (same-binary env arms, 8-min cooldown, 3 pairs): producer
(r2 line) 47.96/50.50/50.97 vs classic 54.29/55.27/55.88 — 3/3
disjoint; K double fold 23.7-24.6; tail 8.2-8.9 vs 31.6-32.0.
MECHANISM SUM (r2 + K + tail): 79.9/83.2/84.5 vs 85.9/87.2/87.9 —
3/3 DISJOINT, avg −4.9ms (on the −5..8 pricing). zc bucket 2/3 with
one polluted K sample (141.5 vs its own min-sum ~118); totals 2/3 —
bucket rule applied. DEFAULT ON; FLOCK_NO_ZC_COMPACT_K=1 kills
(cascade precedent). zc sub-line sums now ≈117 vs their ≈116 —
ZEROCHECK AT PARITY on like instrumentation. Their remaining r2
number includes GPU-arm machinery we correctly skip (CPU-only rule).

### FULL CROSS-CERTIFICATION, ST + MT (2026-08-27 evening, grind-free,
GPU off, strict alternation; cooldown skipped per Benedikt — warm
absolutes, paired-valid ratios)

MT (m=32 blake 8T):
  pair1: ours 457.58 (572,895)  theirs 385.17 (680,587)  1.188x
  pair2: ours 460.04 (569,832)  theirs 384.91 (681,058)  1.195x
  pair3: ours 463.93 (565,045)  theirs 380.21 (689,479)  1.220x
  best-vs-best: 457.58 vs 380.21 = 1.203x
  our buckets: witness 30.0-30.4 / commit 262.9-263.8 / zc 115.4-121.1
  / lincheck 18.6-22.7 / open 29.0-30.6; theirs: 29.0-29.2 /
  221.9-251.5 / 100.4-102.2 / 16.1-16.2 / 20.6-21.2.

ST (1T):
  pair1: ours 3.25s (80,728)  theirs 2.82s (93,005)  1.152x
  pair2: ours 3.25s (80,621)  theirs 2.81s (93,248)  1.157x
  our buckets (repeatable to 0.2%): witness 230-231 / commit
  1266.5-1266.9 / zc 1486-1487 / lincheck 124.8-125.3 / open 143.6-146;
  theirs: 204-214 / 1630-1650 (their prep inside) / 757-759 / 115.5 /
  101.5-101.8. Bucket-boundary caveat: at 1T their r1 prep lives in
  commit, ours in zc — only totals and like-for-like compare.

READING: ST 1.15-1.16x vs MT 1.19-1.22x — the kernels are within ~15%
single-threaded (their generated-asm/16-bit-Horner margins, adjudicated
below bar), and the extra MT spread is the commit-window scheduling
residual (five experiments adjudicated null; their absorption exceeds
ours by ~30-40ms at 8T). Compact-K visible in both axes: zc MT 115-121
(was 121-134), zc ST 1487 (was 1531). CAMPAIGN TOTALS at certification:
MT 723.9 → 457.6 ms = 1.58x; ST 4781 → 3250 = 1.47x; cross-tree gap
1.49x (ranked discovery) → 1.36x (yesterday) → 1.19-1.22x MT /
1.15-1.16x ST certified.

### r1 prep structured-b shortcuts CERTIFIED (2026-08-27, no-cooldown protocol)

What "the 16-bit thing" turned out to be: the shift_reduce 16-bit
geometric-eq machinery is ALREADY OURS (identical mechanism, ported at
r1 parity long ago) — the stale "16-bit Horner as remaining margin"
attribution in earlier entries is hereby CORRECTED. The real adjacent
gap was their prep kernel's structured-b runtime shortcuts, now ported
(generic dispatch only; their ranked-census constants stay adjudicated
as noise): (1) all-ones 8-K b-block → a-only kernel (extension of the
constant-one row is constant one ⇒ y_K = ntt_a: no b gathers, no
product muls; our x4-table/x2-mul weight decomposition preserved);
(2) single-live-K0 block → one dual transform + one lane multiply
(zero rows contribute nothing). Two integer compares per block; exact.
Naive-oracle test with CRAFTED b (all-ones + single-K0 quarters —
random data never hits these paths) + suite 353 + proof-identity.
CERTIFIED under the new no-cooldown protocol (6 pairs, add-pairs-not-
minutes): join wall 5/6 for new, avg −5.7 (new 252.5-259.7 tight vs
old 257.5-271.4); r1-drain and zc lines flat (correct controls — the
kernel runs in the hoisted prep). NEW RECORD pair-5: 454.70 ms /
576,521 c/s. SECOND CORRECTION: the earlier "zc sub-line parity ~117
vs ~116" compared our cool numbers to their warm sub-line medians —
their zc BUCKET is 100-102, so zerocheck retains +14-19 MT post
compact-K; the parity claim is withdrawn.

### FINAL CAMPAIGN TABLE (2026-08-27 evening, grind-free, strict
alternation, no cooldowns — both sides ~1% warm, ratios paired-valid)

MT (m=32 blake 8T, GPU off):
  ours   469.66 / 464.50 / 467.86 ms   (best 564,362 c/s)
  theirs 381.77 / 382.60 / 383.30 ms   (best 686,648 c/s)
  pairs 1.230 / 1.214 / 1.221 — best-vs-best 464.50/381.77 = 1.217x
  buckets ours:   witness 30.3-30.7 / commit 261.7-266.0 / zc
  119.5-124.7 / lincheck 19.2-19.3 / open 28.3-30.1
  buckets theirs: 28.9-29.1 / 220.9-224.0 / 99.5-100.2 / 16.2-16.5 /
  21.1-22.2

ST (1T):
  ours 3.24 / 3.23 s (buckets repeatable ≤0.3%: witness 230.6-231.0 /
  commit 1265.9-1269.8 / zc 1465.2-1468.3 / lincheck 124.0-124.5 /
  open 147.1-147.6)
  theirs 2.82 / 2.84 s (204.0-204.6 / 1660-1690 incl. their prep /
  761.5-765.3 / 115.7-116.4 / 100.7-100.8)
  pairs 1.149 / 1.137 — best-vs-best 3.23/2.82 = 1.145x (campaign-best
  ST; zc 1465 = compact-K + structured-b at 1T, was 1531 this morning)

MT+GPU (theirs only; we are CPU-only by campaign rule):
  theirs 261.44 / 246.74 ms (1,002,687 / 1,062,409 c/s) — GPU absorbs
  commit (220-224 → 124-152) and assists zc r2 (100 → 86-88); vs our
  CPU MT best: 464.50/246.74 = 1.883x.

CAMPAIGN CLOSE: baseline → final: MT 723.9 → 454.70 record / 464.50
this table (1.56-1.59x); ST 4781 → 3230 (1.48x). Cross-tree CPU gap:
1.49x (ranked discovery) → 1.36x → 1.217x MT / 1.145x ST certified.
Kept optimizations: 20+ certified with paired sign tests; nulls: 20+
measured and reverted with written verdicts; every remaining delta
attributed (commit MT scheduling ~35-40, zc +14-19 bucket, open +7,
kernel-generation ST margins) and adjudicated below the bloat bar.

### Repo cleanup: production flags & TEMP probes stripped (2026-08-27)

Removed per Benedikt (features are certified default-on; production
never toggles them): env kill switches FLOCK_NO_ZC_LOOKAHEAD,
FLOCK_NO_ZC_COMPACT_K, FLOCK_NO_OPEN_DIRECT, FLOCK_NO_FILL_FUSE,
FLOCK_NO_AB_HOIST, FLOCK_NO_WITNESS_ELIDE, FLOCK_NO_PREFAULT,
NTT_DEEP_NOFUSE(+static), NTT_DEEP_PCORES_ONLY(+static),
MERKLE_PCORES_ONLY(+static), PCS_COMBINE_PCORES_ONLY,
LINCHECK_PCORES_ONLY, LIG_LOOKAHEAD_DISABLE; TEMP probes: combine_probe
module + open_combine_probe bench + Cargo entry, FLOCK_NTT_SPLIT
top/deep timers, LIG_COMMIT_TRACE per-level commit probe. KEPT: test
oracles as atomics only (OPEN_DIRECT_DISABLE, RS_TAIL_LOOKAHEAD_DISABLE,
ZC_COMPACT_K_DISABLE — the compact-K transcript tests had gone VACUOUS
at the default-on flip [both arms ran compact]; converted FORCE→DISABLE
so the oracle arm is real again); instrumentation used by committed
tooling (FLOCK_PHASE_TSV, FLOCK_ZC_TIMING, FLOCK_COMMIT_TIMING,
*_TRACE family); FLOCK_NO_GRIND (the measurement protocol itself);
FLOCK_ALLCORE (hardware-topology escape hatch, not a feature toggle).
Consequence for future work: same-binary feature A/B is gone for these
— use two-binary git-arm A/Bs. Post-cleanup sanity run: 472.75 ms warm
with all certified defaults engaged; suite 353 + proof-identity +
compact-K oracles green.

### Streamed witness→prep NULL — the strongest bandwidth-on-bandwidth
confirmation of the campaign (2026-08-27 evening)

Built the streamed round-1 prep (Benedikt-directed try): witness groups
publish completion on a channel; two utility-QoS E-threads run the
matching prep chunks (contiguous a/b regions — prefix-friendly, unlike
the commit's strided first pass, whose literal streaming caps at ~4ms
and was ruled out on inspection); the commit-window prep arm finishes
the remainder. IT WORKED MECHANICALLY: prep arm 97→18 (80% streamed),
commit join wall 261→182-194, witness bucket flat, roundtrip+verify
green. BUT totals 465→690-710 (+240!). Discrimination: zero consumers
→ normal; no-op consumers (threads+channel, no work) → normal;
prefaulter QoS raised → no change. VERDICT: the prep's memory work on
2 E-cores during the WRITE-SATURATED witness window collapses the
P-pool's streaming throughput — ~60 thread-ms of E work costs ~240
wall-ms (reads of freshly-written a/b lines add coherence traffic on
top of 512MB of competing stores). 5th bandwidth-on-bandwidth
confirmation, and the sharpest: the same E-cores+prep pairing that WINS
inside the commit window (AB-hoist, certified) is catastrophic inside
the witness window. E-core value is entirely window-boundedness-
dependent. Reverted; patch (631 lines incl. chunked prep API) in
scratchpad/streamed_prep.patch.

---

## MAIN-MERGE PHASE (2026-08-28 →)

Directive: adopt main's protocol (676 commits: recursion_circuit, f256
two-point OOD, ag-union, lagrange-const-denom, legacy-hardening, bloat
phase 1, tower-split), merging mainline PRs one at a time, keeping our
kernel/substrate performance work wherever it still has a live call
path, and re-porting the rest as measured follow-ups.

### Merge step 1: PR #26 recursion_circuit (600a901) — 2026-08-28

**Scope surprise:** the upstream feature branches cross-merged, so
#26's tip already carries the f256 configs, the two-point-OOD f256
split opening, the union (lane-major, integer-lane) commit transport,
per-challenge grinding, and the sparse (M6) zerocheck tail. The
protocol jump lands HERE; later steps should be much smaller.

**53 conflicts.** Resolution policy: main wins protocol semantics
(configs, grinding schedule, opening structure, drivers); ours wins
kernels/substrate. Per-file outcomes:

- 42 config TOMLs: theirs (f256, queries 279, fold_grinding 0,
  claim/consistency batch grinding, ood_samples). Our blake3 hash
  default REVERTED to main's sha256 (generator + derivation test are
  main's; flipping the default is a one-commit DECISION ITEM for
  later — benches select blake3 via FLOCK_MERKLE_HASH/FLOCK_FS_HASH,
  which survived). hash.rs default likewise back to Sha256; our two
  default-hash tests updated.
- scratch.rs: hand-merged. Theirs' pool upgrades kept (4x capacity
  window, class-aware eviction, POOL_TRACE, f256 view pool, zero pool,
  prewarm budget/union) + our provenance-tag API re-applied on top
  (tagged tuples, take/give_f128_tagged). Their U8_POOL replaces our
  duplicate POOL_U8.
- additive_ntt_f128.rs: hand-merged. Theirs' integer-lane + live-lane
  (dead-lane skip) machinery + parallelism floor kept; our on_sub
  per-sub-group hook, from_message fusion, ranked radix-8 top and
  interleaved_n_top helper kept (helper updated to ceil_log2 for
  integer lanes). Both sides' tests kept.
- pcs/commit.rs: hand-merged. Theirs' lane-major commit
  (commit_lane_major, dense_lanes, finalize_commit, cap-based
  Commitment) + our streaming pipelined commit + stage split kept;
  pipelined path now emits the cap; pipeline gated to pow2 lanes
  (integer-lane goes lane-major/staged); pow2 msg-len asserts
  loosened to msg_len_f128().
- pcs.rs, ligerito.rs, ring_switch.rs: THEIRS WHOLESALE (f256-split
  opening is a rewrite; our f128 direct-open, boundary fusion,
  composed-table sweep, lazy-OOD glue have no call path in it).
  RE-PORT CANDIDATES, assessed against the new opening.
- zerocheck.rs: THEIRS WHOLESALE (grinding + sparse-M6 driver).
  multilinear.rs hand-merged: theirs' generic kernel restructure +
  runs/sparse machinery kept, our integrated-lookahead round 2 and
  compact-K section re-appended (adapted to single-run
  round2_pair_skip). Driver wiring for cascade/compact-K NOT yet
  re-grafted — RE-GRAFT ITEM (transcript-identical, so it layers
  cleanly).
- prover.rs, r1cs_hashes/blake3.rs: THEIRS WHOLESALE (union driver;
  auto-merge garbled ours in). Lost for now: stripe-C zerocheck entry,
  AB-hoist (gate returns false with note — its consumer is the
  stripe-C entry), tagged witness give-backs, streamed full-write
  witness builder + provenance elision. All RE-GRAFT ITEMS. prove_fast
  now runs the UNION prover; prove_fast_timed decomposes the legacy
  direct path (bench labeled accordingly; union breakdown via
  PCS_TRACE=1).

**Tests:** flock-core 563 green, flock-prover 114 green, ignored
roundtrips green (batch-major prove_fast, prove_fast_ag, ligerito
roundtrip, const-pin). Two auto-merge stitch bugs caught at compile
(fused_2layer_row_op arity, build_b_med_counts arity); one caught by
tests (pipelined commit cap). Lesson re-confirmed: auto-merged regions
of co-evolved files are STITCH HAZARDS — the failures were loud, but
only because main's test additions (derivation test, l0-matches-full)
covered them.

**Step-1 bench snapshot** (m=32, grind-free, blake3 via env, warm,
single session — indicative, not certified):
- Union headline prove_fast: best 879.5 ms (runs 1.13 s / 879.5 ms /
  1.11 s) vs 465 ms pre-merge = 1.89x. Peak memory 8.93 GB. Proof
  567.31 KiB (was 427.30).
- Union phases (PCS_TRACE, warm run): commit 259 ms (staged lane-major
  — no streaming pipeline on this path yet), boolean zerocheck +
  lincheck 284–458 ms (noisy; no compact-K/cascade), open 266–297 ms
  (f256 split: W build ~40, merged sumcheck ~18–37, inner ligerito
  ~193–222), witgen ~free (batch-major partial + zero pool).
- Legacy direct path (prove_fast_timed, one cool run): witness 51 ms,
  commit 193 ms, zerocheck 1.56 s (!!), lincheck 84 ms, open 393 ms.
  The 1.56 s zc is an OPEN ITEM — smells like the generic run-list
  (scalar) round-2 kernel or a mis-gated fast path on the direct
  spec; investigate during the zc re-graft.

**Re-port/re-graft queue** (each as its own measured commit, after the
remaining merge steps land): (1) zc cascade + compact-K driver wiring
onto the grinding driver; (2) streaming/pipelined commit for the
lane-major union L0; (3) stripe-C + AB-hoist onto the union/direct
drivers; (4) streamed witness builder + provenance elision for the
batch-major generator; (5) direct-open/boundary-fusion ideas vs the
f256-split opening (assess — the opening changed shape); (6) direct
path zc 1.56 s anomaly; (7) DECISION: flip default hash to blake3
repo-wide (generator + TOMLs + tests) or keep sha256 default with env
selection for benches.

### Merge step 2: PR #32 f256-lookahead (372d323) — 2026-08-28

Clean merge, ZERO conflicts (step 1 absorbed the cross-merged bulk).
+1459/-120 over 11 files, dominated by ligerito/extension.rs (f256
tower extension fold machinery + fold-lookahead tests). Suites green
(core 564, prover 117). m=32 grind-free blake3, 3 cold-start runs:
1.90 / 1.17 / 1.15 s (step 1 same protocol: 1.85 / 1.34 / 1.19) —
PARITY within session noise; proof size unchanged (567.31 KiB), peak
8.52 GB.

### Merge step 3: PR #33 ag-union (6ee974f) — 2026-08-28

Clean merge, zero conflicts. +4727/-555 over 20 files: AG-skip union
integration (genus95 round 1, ag_skip driver, tower.rs expansion,
union verifier work, 128-bit grinding audit doc). Suites green (core
572, prover 118). m=32 grind-free blake3 cold-start: 1.66 / 1.12 /
1.79 s, best 1.12 — parity (run-3 spike looks environmental); proof
size unchanged.

### Merge step 4: PR #10 lagrange-const-denom (via abe3df1 fmt tip) — 2026-08-28

3ff5d29 was already in ancestry (absorbed by the cross-merged
branches); this merge added only the fmt commit's merkle_path.rs delta
(+52/-19). Suites green (core 572, prover 120). Bench cold-start:
2.27 / 2.15 / 0.948 s — best 948 ms, parity; first-two-run spikes look
like memory-pressure churn from back-to-back 8.5 GB bench sessions
(the certified comparison comes after the re-grafts, with proper
alternation).

### Merge step 5: PR #34 legacy-hardening (2bb04a9) — 2026-08-28

Clean merge, zero conflicts (+186/-1: verifier/r1cs hardening checks,
small blake3/sha2 additions). Suites green (core 574, prover 122).
First bench session read 2.26/2.32/2.28 s across the board — rerun
gave 1.85/1.07/0.970 s: the slow session was machine state (788K
pageouts; back-to-back 8.5 GB bench sessions), not the PR. PARITY,
best 970 ms.

### Merge step 6: PR #37 bloat-phase1 (86d5fd5+61cff5f) — 2026-08-28

Two trivial merkle.rs hunks (kept main's MERKLE_PCORES_ONLY knob).
Main's own purge: -13,121 lines over 66 files, INCLUDING the legacy
direct-path prover (prove_fast_timed / prove_fast_ligerito_timed) —
the bench's timed breakdown block is gone with it; per-phase
attribution is now PCS_TRACE=1 only, matching main's bench. This also
retires the "direct path zc 1.56 s" open item (the path no longer
exists) and re-scopes re-graft targets to the UNION driver only.
Suites green (core 545 — count reflects deleted legacy tests, prover
124). Bench: 1.92 / 1.72 / 0.919 s, best 919 ms — parity.

### Merge step 7: PR #38 tower-split (b310f35) — 2026-08-28

Clean merge, zero conflicts (tower.rs split into
query/real_walker/tape modules; ~20K moved lines). Suites green (core
545, prover 124). Bench 1.58 / 1.18 / 1.73 s, best 1.18 — parity
(session noise; the machine has been paging all afternoon).

**MERGE COMPLETE: merge-base(HEAD, origin/main) == origin/main tip
(b310f35).** All 7 mainline PRs are in. Union prove_fast at m=32
grind-free blake3 sits at ~0.92–1.18 s best-of-3 cold-start across
steps (vs 465 ms pre-merge) — the gap is the re-graft/re-port queue
from step 1, now re-scoped to the union driver: zc cascade+compact-K,
streaming lane-major commit, stripe-C/AB-hoist, streamed witness
builder, direct-open ideas vs the f256 opening.

### Default hash → BLAKE3, repo-wide (Benedikt-directed) — 2026-08-28

Restores our pre-merge default on top of the merged tree, this time
through main's own machinery: HashKind::default() → Blake3, the
security-config generator emits hash = "blake3", all 98 embedded
config TOMLs flipped (derivation test keeps generator and TOMLs
locked), FsChallenger::new() → Blake3 transcripts. Tests updated to
be default-agnostic where they pin consistency (23 test cfg literals
pinned to explicit Sha256) and flipped where they pin the default
(challenger default pin, params-inherit test, m29 TOML load, m22
roundtrip now exercises the sha256 arm + blake3-mismatch reject).
Proof-byte fixtures re-pinned by design (union_element 7,
union_m6_fixtures 6; two deterministic print runs agreed each).
Suites green (core 545, prover 124) + ignored roundtrips green with
grinding on. Bench sanity: defaults print blake3/blake3 with no env;
times in the machine's current paging-degraded band, hash config
identical to prior runs.

### Merge step 8: bloat-phase2/3 (8a36c91) + first branch-vs-main A/B — 2026-08-31

Main moved 10 commits past b310f35; merged cleanly on top of the
blake3 flip. Resolutions: the 28 retired `*_fast128`/`*_slim128`
TOMLs deleted (main's rename — Fast/Slim now CARRY the 128 schedules,
proof-IO v22; our `hash = "blake3"` line auto-merged into the
surviving files), gf2_128 wide-NEON tests moved to the shared
`test_rng::Rng` (`next_f128` → `f128`), 13 proof-byte pins
regenerated (both sides stale: v22 × blake3 default; two agreeing
deterministic print runs each). Suites green (core 557, prover
green). merge-base == origin/main tip again.

**Branch-vs-main paired A/B (m=32, grind-free, blake3/blake3 pinned
via env on BOTH arms; mainline worktree carries two measurement-only
patches: the FLOCK_NO_GRIND knob in grind_pow and the bench
verify-skip).** Totals run (8 pairs, alternating): 5/8 branch, but
the run split early-late (pairs 1–5 all branch −20..−140 ms, pairs
6–8 all main +40..+83) with mediaanalysisd at 64% CPU and the battery
fast-charging at 25–29% while DRAINING under load — both named
hazards; unadjudicable. Phase-paired run (6 pairs, PCS_TRACE
min-per-phase): commit 2/4 med +10 ms, zc+lincheck 4/2 med −7 ms
(pair swings ±150 ms — foreign load), open 3/3 med 0, total 4/2 med
−2 ms. **Verdict: no bucket certifies a difference; branch ≈ main
end-to-end within today's noise.** Consistent with the liveness
audit: the live survivors (blake3 hash8 leaf kernel, lincheck NEON
rewrite, round1 E-core drain) are tens-of-ms items under a
±50–150 ms floor; every big-ticket optimization (compact-K/cascade
zc, streaming pipelined commit, stripe-C/AB-hoist, streamed witness
builder) is present but DORMANT — no driver calls. Re-certify on a
quiet, charged machine; the real move is the re-graft queue.
Trace note: `bind statement` is ~815 ms on the first prove only
(cached after) — cold totals ~2.4 s are not a regression signal.

### ST/MT phase table, branch vs main — 2026-08-31

m=32 grind-free blake3 both arms, one invocation per cell (min per
phase over 5 proves; ST = RAYON_NUM_THREADS=1). ms:

| phase   | br MT | mn MT | ratio | br ST  | mn ST  | ratio |
|---------|-------|-------|-------|--------|--------|-------|
| commit  | 232.0 | 227.6 | 1.019 | 1690.1 | 1718.9 | 0.983 |
| zc+linc | 226.7 | 257.7 | 0.880 | 1556.7 | 1745.7 | 0.892 |
| open    | 264.0 | 264.7 | 0.997 | 1582.2 | 1606.2 | 0.985 |
| sum     | 722.7 | 750.0 | 0.964 | 4829.0 | 5070.8 | 0.952 |

witgen/compact 0.0 everywhere (pooled-zeroed / aliased). The ONE
surviving live win that resolves: **zc+lincheck −12% MT / −11% ST** —
the ST agreement pins it on the lincheck NEON block-kernel rewrite
(load-port fix), not scheduling; the round1 E-drain adds little
beyond it at MT. commit and open are parity (hash8 leaf kernel gain
is inside the ±2% floor here). Sum-of-buckets ≈ prove TOTAL −~12 ms
unlabeled residue. `bind statement` one-time ~790 ms in all four
cells (identical arms). ST/MT scaling ≈ 6.7× on 8 P-cores. Caveats:
single cell per config, machine on 24% battery with interactive load;
earlier trace-parse footgun fixed (open_batch/open_merged also print
"TOTAL" — match "[prove_union] TOTAL").

### Re-graft #1: zc compact-K + cascade wired onto the union grinding driver — 2026-08-31

Ported the f035ddb round-2 branch (compact-K rounds 2..5 | integrated
r2 lookahead | classic) and the 4→1 cascade tail into main's grinding
driver: unified while-loop with `rho_prev`/`pending2` deferred-
challenge state, one `sample_rho!` grind-or-sample per round message
(nonce cadence identical on every route), two-challenge final binding.
All kernels had survived in multilinear.rs; zerocheck.rs re-gains the
two test oracles (ZC_COMPACT_K_DISABLE, RS_TAIL_LOOKAHEAD_DISABLE) and
the three transcript-identity tests. New `PaddingSpec::
effective_single_run()` — as_single_run modulo trailing all-zero runs
(the fast path's implicit gap) — lets the single-type union's
gap/useful/tail list serve the single-run kernels.

**Measured verdict — sparse keeps the bench.** The union boolean
region at m=32 is ~19% occupied (useful_cols/128), so main's
support-proportional sparse path fires; with cascade priority forced,
compact-K's producer pays a full-domain pass: r2+K+tail 145–190 ms vs
sparse r2+tail 93–170 (K fold 31–48 + cascade tail 11–15 ARE cheaper
than the sparse tail's 44–103, but the producer eats it). The old −5
certification was on a ~fully-occupied witness; main's sparsity
banked that win differently. Priority reverted: sparse dispatches
first, the cascade serves dense single-run flows (where the sparse
occupancy gate fails and the 2026-08-27 certification applies).
Dispatch is visible under FLOCK_ZC_TIMING ("gates:" line).

Suites: core 560 (3 new oracle tests), prover 76, all 13 proof pins
UNCHANGED — transcript identity of every route, enforced. Takeaway
for the queue: today's zc cost is round1 URM ~120 ms — stripe-C/
AB-hoist (item #3) is the zc money now, not round 2/tail.

### Re-graft #3 attempt: round-1 AB hoist onto the union commit — NULL, reverted — 2026-08-31

Wired the certified 2026-08-27 join (prep beside commit on the
all-core pool; `Round1AbPre` threaded through
`prove_packed_padded_capture_s_hat_v_c_with_grinding`) onto the union
driver. Mechanism buckets moved exactly as designed — round1 URM
120 → 53–56 ms (6/6, −61 median), zc+lincheck −60 (6/6) — but the
commit wall paid +52 (0/6): **net total 5/6 at only −3..−11 ms, a
wash.** Second variant split `commit_lane_major` into fill/finalize
and joined the prep beside NTT+Merkle only (the fill is the
streamed-prep null's write-saturated phase): the fill ran clean
(82 → 24 ms) but the NTT absorbed the same contention (90 → 150–160)
— identical wall.

The arithmetic says why, and it generalizes: the prep is ~480
thread-ms; the union commit window is ALREADY all-core dense (the
deep NTT pass recruits the E-cores itself, unlike the 2026-08-27
tree), so overlap is zero-sum — wall grows by work/threads ≈ 48 ms
for the 60 the zerocheck saves. There is no idle-slack window in this
prover for challenge-independent prep to hide in; a future hoist only
pays off if some phase leaves cores (not bandwidth) idle. Reverted
per the measured-revert rule; suites stayed green throughout and the
transcript never moved (prep is bit-identical by construction).

### Re-graft #2: lane-major streaming commit — UNCERTIFIED, parked — 2026-08-31

Ported the leaf pipeline to the union's integer-lane commit: factored
`commit_into_pipelined`'s queue/helper engine into `leaf_pipeline_run`,
added a lane-major variant (staged transpose fill, then the live-lane
NTT deep pass publishing whole-position blocks — non-pow2 lane counts
align by construction, so the pow2 gate is not needed there), plus a
staged-vs-pipelined byte-identity oracle at non-pow2 t. Mechanically
correct: merkle-top 116 → 0.2 ms, smoke commit 239–249 vs staged
~250–270.

**12 paired A/B vs 9c119e5, two rounds: commit bucket 6/12 med +2 ms,
totals 6/12 med +3 — consistent with zero.** Round 1's apparent zclc
+19 side effect vanished in round 2 (machine texture, not the change).
Reading: the queue fills and most leaf hashing lands inline on the
publisher anyway — the work relocates into the NTT window; the real
saving is only the codeword's DRAM re-read (~25 ms theoretical at
m=32) and pipeline overhead eats most of it. Conditions were the worst
of the day (19% battery, net-draining on AC, interactive load).

PARKED on branch `regraft-2-lane-pipeline` (pushed) with the oracle
test — re-run the A/B on a quiet charged machine, and at m=30/34
where the codeword (and the re-read) is larger. Not merged: does not
earn its ~120 lines on this evidence.

### Zerocheck route A/B: sparse vs compact-K+cascade, SAME BINARY — 2026-08-31

`FLOCK_SPARSE_GATE` (the existing env override on `SPARSE_TAIL_GATE`)
makes the round-2 dispatch a same-binary A/B: gate=1 (default) takes
main's sparse route, gate=huge forces the dense route, which now fires
the re-grafted compact-K + cascade. 8 pairs, alternating, min-of-5 per
invocation, m=32 grind-free blake3. ms:

| bucket        | sparse | dense/cascade | sign |
|---------------|--------|---------------|------|
| round1 URM    | 117.6  | 118.0         | 5/8 (unaffected — same code) |
| round 2       |  43.4  | 101.7         | 8/8 sparse |
| compact-K fold|   —    |  30.3         | — |
| rounds 3+ tail|  44.9  |  10.6         | **0/8 sparse — cascade 4.2× faster** |
| r2+K+tail     |  89.3  | 142.7         | 8/8 sparse |
| zc+lincheck   | 239.3  | 288.7         | 8/8 sparse |

**Both halves are 8/8 disjoint, in OPPOSITE directions.** Sparse owns
round 2 (−58 ms: it never materializes the dead region the compact
producer sweeps); the cascade owns the tail (−34 ms: one 4→1 pass
serves two rounds, and our dispatch currently forces an UNFUSED tail
whenever round 2 went sparse — `tail_cascade` requires
`dense_single_run`, zerocheck.rs:842). Neither route is optimal. A
hybrid — sparse (or sparse-compact) round 2 handing a compacted state
to a SPARSE 4→1 lookahead kernel — targets ~54 ms where we now pay
89.3, i.e. ~−35 ms on the zc bucket. That kernel does not exist in
either tree: main has sparsity without fusion, the challenge tree has
fusion without sparsity. This is the one genuine "port the idea, not
the code" item the zerocheck has left.

CORRECTION to the re-graft #1 entry above: its "~19% occupancy" figure
was asserted from a misread of `boolean_padding_spec` and is
WITHDRAWN — the sparse route engages at this shape because
`SPARSE_TAIL_GATE = 1` makes the gate `live ≤ n` (always true), not
because the witness is very sparse. The gate's own doc describes the
intended crossover as half utilization (`live·2 > n` stays dense), so
the constant and the doc disagree; at 8/8 for sparse on round 2 the
constant is right for round 2 and wrong for the tail. Re-graft #1's
verdict (sparse keeps the bench) stands on the measurement, but its
stated cause was wrong.
