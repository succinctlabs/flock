# Two recursion variants: 100-bit and 128-bit

> **Historical-command note (2026-08-15):** the `--test circuit_merkle`
> invocations and `TOWER_PROFILE`/`TOWER_ENV_M` env vars quoted in the
> resolved-incident records below describe the pre-productionization harness.
> `tests/circuit_merkle.rs` has since moved into `src/tower.rs` — the mvp*
> test names quoted below were retired with the productionization (only
> `chain_spine_converges` and `chain_tower_e2e_with_lane` survive, run with
> `--lib tower:: -- --ignored`), and the profile/geometry env vars were
> replaced by the
> typed `TowerConfig::{Chain100, Chain128}` (`TOWER_CONFIG=chain100` for
> the 100-bit tower). The records are kept verbatim as history.

Original review context: Ron's 2026-08-12 `pow-mask-slot` session, based on
`min/recursion-128bit` at `4603c0f`. The resolution annotations and validation
status below describe the rebased `min/recursion-128bit` working tree as of
2026-08-13.

Ron wants both security levels runnable side by side while the 128-bit work
matures: **the 100-bit variant is the configuration shipped before your
branch**, and **the 128-bit variant is the configuration you are building**.
This note records how the variants are realized, the evidence they reproduce
the old cost point, their validation status on both recursion tracks, and
three pre-existing issues the chain-track runs surfaced — all three verified
against your unmodified tip, none introduced by our commits.

## The variants

> **Consolidation note (2026-08-27, bloat ledger §C, proof-IO v22):** the
> grind-free `Fast`/`Slim` with the +1/level ladder were deleted and the
> `Fast128`/`Slim128` schedules took their names. Strict `Fast`/`Slim` now
> mean: aggressive ladder (rate +2/level), 16-bit query PoW at every level
> (query term 112 bits + PoW, work-normalized to 128), larger deep-level
> batch grinding. `Fast100`/`Slim100` are unchanged and frozen at their
> historical schedules, so they now differ from the strict profiles in
> ladder, PoW and target — the "only the query target differs" statements
> below describe the pre-consolidation code and are kept as the record of
> how the 100-bit variants were derived.

| | 100-bit | 128-bit |
| --- | --- | --- |
| leaf/node track (mvp) | `TOWER_PROFILE=fast100` | default `TOWER_PROFILE=fast` |
| chain track / spine | `TOWER_PROFILE=slim100` (envelope on, m29) | `TOWER_PROFILE=slim` (envelope on, m29) |
| in code | `PcsParams.profile = Fast100 / Slim100` | `= Fast / Slim` |

The right-hand column is the strict component-security configuration. Its
Johnson query, two-point OOD, Flock-paper Appendix C.3 batching, and F256 MCA
terms each clear 128 bits. `Secure` remains the separate historical 120-bit
unique-decoding profile used as an additional regression target for every
algebraic-grinding family; its name does not make it the 128-bit profile.

Within Ligerito, `Fast100` and `Slim100` use the same Johnson accounting,
two-point OOD, Flock-paper Appendix C.3 grinding schedule, F256 transcript
shape, and m28/m29 `initial_k` exceptions as `Fast` and `Slim`, except that
the consistency-query term targets the profile's own `security_bits()` (100)
instead of `LIST_DECODING_QUERY_TARGET_BITS`. The Johnson query floor in
`LigeritoSecurityConfig::validate` is keyed off `analysis_version`
(`query128` vs `query100`), so each config family validates against its own
target; a boundary test pins the `Fast100` floor and the schedule equality.
Both variants live in one binary. Proof format v20 (head is v22 as of
2026-08-27 — the profile consolidation; v21 on 2026-08-14, `6988b62`, was the
union-only batch consolidation) additionally makes the
non-Ligerito policy distinction explicit: strict `Fast`/`Slim` enable all
algebraic grinding families, whereas `Fast100`/`Slim100` intentionally keep
those families disabled. Their public Ligerito query schedules also remain
different.

**They reproduce the pre-branch schedules exactly.** The canonical generator
at target 100 re-derives, byte-for-byte, the counts replaced by the strict
128-bit schedules:

- `m27_fast100`: per-level `[218, 106, 71, 53]`, Σq 448 (your audit's own
  "before" example);
- `m29_fast100`: Σq 491; `m29_slim100`: Σq 262 — the exact numbers in
  `circuit_merkle.rs`'s profile comment.

This doubles as a proof that the old Fast/Slim were 100-bit query targets.
Note the 100-bit variants are *slightly stronger* than the literal
pre-branch state inside Ligerito: they inherit two-point OOD binding, C.3
algebraic grinding, and F256 MCA. They do not claim 128-bit security and do
not enable the non-Ligerito grinding families.

## Validation status

The original measurements were made on `pow-mask-slot`. The rebased branch
carries the same fused PoW-mask design (one four-word row per grinding check)
at `f8093ec`, plus the soundness repair described in
`128-bit-grinding-audit.md`.

- `flock-core --lib` (incl. the config suite over all 70 embedded TOMLs) and
  the full `flock-prover` suite: green.
- 100-bit leaf/node: mvp11 node + mvp12 tower green under `fast100`.
- 128-bit leaf/node: mvp11 node + mvp12 tower green under strict `fast`.
- 100-bit chain track: `first_level_node_two_chains_fold_and_adjacency`,
  `chain_tower_e2e_with_lane`, and — notably — **`chain_spine_converges`
  green under `slim100` + the m29 envelope**. The envelope's fixed point
  held with zero re-pins, since it was iterated against exactly these
  schedules. This is the first spine run since your Ligerito parts landed.
- 128-bit chain track: strict-Slim m29 spine and chain tower are green. The
  historical Secure chain tower is also green as a compatibility regression;
  see the 2026-08-13 resolution below.

Historical same-run mvp11 comparison from Ron's review (medians of 10 online
reps, M4 Max; `secure` is the 120-bit compatibility profile, not the strict
128-bit column above):

| | 100-bit (`fast100`) | 128-bit (`secure`) |
| --- | ---: | ---: |
| online prove | 117 ms | 146 ms (+25%) |
| outer proof | 292.0 KiB | 566.7 KiB (+94%) |
| BLAKE rows / capacity | 24,827 at nu 15 / mu 23 | 38,489 at nu 16 / mu 24 |

## Issues found on `4603c0f` and current resolution

**1. MVP-7 failed to parse its own tape (`LegacyPow`).**
`mvp7_real_query_phase` dies at "op 272: expected the next cap absorb, got
`LegacyPow { bits: 9 }`" — its inner still records legacy PoW ops the
post-fusion parser never learned. Looks like mvp7 was not converted with
the rest of the mvp ladder. Repro:
`cargo test --release -p flock-prover --test circuit_merkle mvp7_real_query_phase -- --ignored --exact`.

Resolved upstream by converting MVP-7 to the fused transcript path. The exact
reproduction command is green after the rebase.

**2. The `initial_ood` walk broke on the new Slim schedule ("L0 OOD beta").**
`parse_open_levels`'s L0 OOD loop panics at the `SqueezeScalar` expectation
when walking a strict-128 slim tape — which blocks **every** envelope run
(`TOWER_PROFILE=slim` + `TOWER_ENV_M=29`), and with it the whole spine, at
the 128-bit schedule. Diagnostic that should localize it quickly: the SAME
parser walks old-schedule slim tapes fine (`slim100` spine passes end to
end), and fast/fast100 tapes fine — so it is something the new slim counts
change about the L0 region's op order, not slim's query-phase PoW or rate
per se. The diagnostic at that point dumped its surrounding op context.
Repro:
`TOWER_ENV_M=29 TOWER_PROFILE=slim cargo test --release -p flock-prover --test circuit_merkle chain_spine_converges -- --ignored --exact`.

Resolved by anchoring `parse_open_levels` at the
`flock-ligerito-basis-v0` protocol label, then checking the target and L0 cap
that follow it. The former "last cap with this byte length" heuristic could
select a later equal-sized recursive cap. The exact strict-Slim reproduction
now passes through the converged spine.

**3. The Secure chain tower failed in witness generation.**
`chain_tower_e2e_with_lane` under `TOWER_PROFILE=secure` panics with "a
connected wire disagrees with the gate output that produces it (slot 0
[= b3], class root …)" in `builder.rs`. This is your audit's "recursive
verification of the chain/Merkle wrapper proofs is out of scope" carve-out
made concrete: the chain-track tape walkers were never exercised against
grinding transcripts, and somewhere the FL/node emitters wire a chain row
under a pre-grinding shape assumption. Repro:
`TOWER_PROFILE=secure cargo test --release -p flock-prover --test circuit_merkle chain_tower_e2e_with_lane -- --ignored --exact`.

Resolved by replacing the chain lane's manual ordinary-finalize replay with
the canonical PoW-aware `fs_chain::trace_duplex`. Under Secure, the manual
loop ignored the fused PoW compression counter and therefore supplied the
recursive fold endpoint with a different challenge from the native verifier.
The exact reproduction now passes, including both lane discharges and tamper
tests.

## Chain-128 closure

1. The strict-Slim parser is fixed and its converged m29 spine passes.
2. The Secure chain lane uses the PoW-aware trace and passes end to end.
3. No count-cap re-iteration is required in the shipped design: free counts
   are unconditional. The strict-Slim spine validates the live PoW count,
   pinned lane count, fixed public layout, and steady circuit digest. The old
   `counts_bool`/`counts_el` values remain only the retired padding oracle and
   slot-declaration key list.

The 2026-08-13 validation matrix also passed strict Fast `mvp11`/`mvp12`,
Fast100 and Secure `mvp11`/`mvp12`, strict Slim and Slim100 chain towers,
Slim100's converged spine, `mvp7`, the full active `flock-core` suite (501
passed, 22 ignored), and the full active `flock-prover` suite and integrations
with no failures. Ron's three ignored
proof-byte pin tests were also run explicitly. The branch's deliberate v20
strict-profile transcript change moved the fixture digests; the new values
were identical across two print runs, were documented at the fixtures, and
the pin tests now pass normally.

## Family-H recursive closure (2026-08-14)

The ring-switch verifier's family-H arithmetic is now inside every recursive
R1CS verifier path: the direct boolean leaf, the first-level `ChildRegion`,
and the steady `RealRegion`. The circuit computes the two
transpose/equality-weight dots, derives the inverse-Moore coefficients,
replays all Frobenius powers in the $V$ recombination, adds the
packed-direct/group terms, and copy-constrains the terminal relation
`running = q_eval * V`. The native target/running replays remain test oracles
only; they are no longer the soundness mechanism.

The transpose uses a 17-IO-word boolean table over dynamic $8\times8$ tiles.
The inverse-Moore rows use the GHASH trace-dual basis's geometric tail and
seven exceptional entries. The two RS Frobenius ladders are paired in F256;
their 8,128 squarings per child fill the existing narrow F256 MAC slot and
spill into the existing F256 spine at the physical row limit. No new element
type was added. At m32 the envelope remains `nu=14`, `mu=23`, 511 gate cell
slots plus one public slot, with a 256.0 KiB recursive proof.

Same-host, same-command steady comparison against `9b94943` (x86 host,
64 logical CPUs, three runs per stage, `TOWER_PROFILE=slim128`,
`CHAIN_BLOCKS=262144`):

| online stage | before family H | after family H | delta |
| --- | ---: | ---: | ---: |
| base chain leaf | 1313.3 ms | 1282.8 ms | -2.3% (noise; path unchanged) |
| first-level wrapper | 403.9 ms | 441.0 ms | +9.2% |
| fresh internal node | 375.9 ms | 424.9 ms | +13.0% |
| steady spine node | 382.1 ms | 430.1 ms | +12.6% |
| amortized per leaf | 1706 ms | 1718 ms | +0.7% raw |
| throughput | 154k comp/s | 153k comp/s | about -0.7% raw |
| recursive proof | 255.7 KiB | 256.0 KiB | +0.1% |

Because the unchanged base-leaf measurement happened to improve by about
30 ms, the raw amortized delta understates the added verifier work. Holding
that stage at its baseline gives about 1749 ms/leaf, or a more conservative
**+2.5% amortized family-H overhead**. The internal-node BLAKE census moves
22,804 to 23,133 rows (+1.4%); family H itself is algebraic, while the BLAKE
increase comes from the enlarged recursive statement/claim surfaces. Dense
area moves 2,977,357 to 3,466,436 words (+16.4%) but remains at `dense_m=29`.

Validation includes the tile relation's honest/transcript assembly plus
mutated-output and mutated-selector rejection, the direct-leaf and standalone
mixed-child recursive proofs, the m32 headline tower, the converged
`chain_spine_converges` run, all active `flock-core` library tests (506 passed,
25 ignored), and the complete active `flock-prover` test suite.

## Rebased branch map

- `f8093ec` — the fused PoW mask row (one four-word row per grinding check;
  isolated grinding overhead 11 → 6 ms, +4 → +2 BLAKE rows). One subtlety
  relevant to your recursive PoW relation: the nonce-width rejection lives
  in the mask word's WIRE BINDING (word 2 must equal the statement's mask
  constant, whose high half is zero), not in the R1CS alone.
- `2959d88` — merge of the Ligerito parts 1–3; the new Ligerito
  `Pow` sites ride the fused row automatically via the generic tape walk.
- `3f943eb` — `Fast100`.
- `cfcfe16` — `Slim100`, spine validation, and the parser diagnostic that
  exposed issue 2.
- `be75c25` — Ron's original review report. The current working-tree fixes are
  recorded above and in `128-bit-grinding-audit.md`.

## Parked option: chain100 on the F128-only ladder (~140 ms nodes)

*Recorded 2026-08-14. Status: PARKED — a real project, not a config flip.
Revisit if Chain100 becomes a product tier where +20% throughput matters.*

Before the F256 rewrite, the 100-bit chain tower ran on an F128-only
Ligerito ladder and its nodes cost ~140 ms — vs ~199 ms today. That world
is fully recoverable from git, and it is sound for 100 bits: F256 exists
only to close the proximity-gap/MCA term that the 128-bit target needs.

**The pointers:**

- `be75c25` — the last F128 world. `configs/ligerito/m29_slim100.toml`
  there says `field = "f128"` with the certified 100-bit analysis
  (`analysis_version = "johnson_two_point_ood_query100_c3_algebraic_…"`,
  v18-era: two-point OOD + Appendix C.3 grinding). Measured on Ron's
  M4 Max at m32: leaf 423.9 | FL 194.5 | internal 142.7 ms, amortised
  593 ms/leaf -> 442k compressions/sec.
- `97cc1d2` ("feat: 256-bit field ligerito") — the in-place rewrite that
  removed it: ~687 churned lines in the `pcs/ligerito.rs` ladder core plus
  the new F256 modules (`ligerito/extension.rs`, `field/gf2_256.rs`,
  ring-switch/tensor changes), and 2,041 lines of in-circuit-verifier
  changes (then `tests/circuit_merkle.rs`, since productionized as
  `src/tower.rs` behind `TowerConfig::{Chain100, Chain128}`). The config
  loader now REJECTS `field != "f256"`
  (`LigeritoSecurityConfig::validate`), so the knob survives but the path
  behind it does not.
- Attribution of the +60 ms/node: the chain-m32 paired-trace session
  (2026-08-14) — ~+25 ms soundness-priced protocol (coordinate-split F256
  commits, two-limb folds, 128-bit PoW), ~+13 ms honest content growth
  (+12% replayed b3 rows), +4 ms was a real bug fixed as `9878c4c`; the
  serial-fill bug was fixed as `cf05a2a`. The remaining ladder is at its
  measured floor (~2x the F128 baseline in every phase = the honest F256
  cost).

**What restoring it takes** (sized from the rewrite itself):

1. *flock-core, first*: make the ladder core field-generic again (or keep
   two variants of the commit/fold/OOD/handoff phases) and un-reject
   `field = "f128"`. This is rkm0959's rewrite; a supported restore in
   core is far cheaper than forking around it downstream.
2. *The tower*: dual transcript-shape support keyed by `TowerConfig` —
   the pre-F256 parse/emit paths back alongside the F256 ones, including
   the per-prefix base-field `ResidualGate` family deleted in the stage-3
   registry diet (`babcb52`), plus a SECOND envelope registry with its own
   measured counts*/publics*/lanes* census. Est. 1.5–2.5k lines of
   circuit code.
3. Two tower-proof wire formats, and a re-certification pass that the
   resurrected 100-bit grinding budgets (`fold_grinding_bits` 17 vs 0,
   etc.) still compose with the post-v18/v20 protocol.

The permanent cost is the fork itself: every future ladder/protocol change
(the m*=28 envelope re-target, new grinding families) lands and audits
twice. The cheaper lever that helps BOTH configs is the m*=28 re-pin
(`envelope_content_probe` is its sizing instrument) — it will not reach
140 ms, but it is one protocol, one audit.

---

Also, see https://claude.ai/code/artifact/70a216b1-ecd9-4839-b1ce-cbbca24a3618 for an audit of our branch in Fable 5.
