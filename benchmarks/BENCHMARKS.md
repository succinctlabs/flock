# Reproducing Flock's benchmarks

This document gives the commands that produce Flock's headline numbers —
per-hash throughput, a fixed-point system profile, the per-phase prover
breakdown, and throughput scaling — and how to run the same points against the
competitor provers.

The reference measurements were taken on a single **Apple M4 Max (36 GB RAM,
10 performance cores)**. Numbers are machine-specific; the *method* below
reproduces them on comparable hardware.

---

## 0. Prerequisites

```bash
# from the repo root
cargo build --release            # build Flock
```

**Competitor provers** (binius64, plonky3, hashcaster) live in gitignored
subdirectories under `benchmarks/`. Each has a `setup.sh` that clones
its upstream repo (pinned to a fixed commit), applies the necessary patches, and
builds it. The orchestrator scripts below invoke these automatically on first
run; the first run of each competitor therefore includes a one-time clone+build
(several minutes).

**Conventions used throughout:**

- **Threads.** Multi-threaded (MT) runs pin to the performance cores
  (`init_perf_thread_pool` / `RAYON_NUM_THREADS=<P-cores>`, matched across
  provers); single-threaded (ST) runs use `RAYON_NUM_THREADS=1`.
- **Best-of-3.** Every prover is measured as the minimum of 3 timed runs after a
  warm-up (Flock via `n_runs=3`; competitors patched to best-of-3).
- **Result cache.** Each orchestrator caches every `(prover, size, threads)` row
  under `benchmarks/bench-<hash>-cache/` (gitignored). A plain re-run
  reuses cached rows and only runs what's missing; **naming a prover** re-runs it
  fresh; `NO_CACHE=1` forces a full fresh run; delete the cache dir to reset.
- **Cooldowns.** Pass `--cooldown N` (or `COOLDOWN=N`) to sleep N seconds between
  benchmarks so thermal throttling doesn't bias later (especially ST) runs.
  **~20 s is recommended**; default is off.

All `cargo bench` commands below run against the local Flock prover and need no
competitor setup; the `benchmarks/` orchestrators are only needed for the
cross-prover comparison tables.

---

## README hash throughput matrix

Regenerate the README table across SHA-256 and BLAKE3, at
multiple batch sizes, using one
thread and the machine's physical-core count:

```bash
benchmarks/bench_hash_throughput.sh
```

The runner prints a Markdown table ready to paste into the README. Defaults are
best-of-3 measurements at batches 2¹⁰, 2¹², 2¹⁴, 2¹⁶, and 2¹⁸. Throughput is
rendered in thousands of hashes per second (`k hashes/s`). Override them with
`LOG2S="..."`, `RUNS=N`, and `MT_THREADS=N`. Every point includes an untimed
warm-up proof that is verified before the timed trials.

---

## 1. Native hashing baseline

Per-core software-hashing throughput (the "Native" reference row):

```bash
cargo bench --bench native_hash
```

Read the per-core ops/s:
- `SHA-256 compress (scalar)` → ≈ 7.0 M/s
- `BLAKE3 compress  (scalar)` → ≈ 18 M/s

---

## 2. SHA-256 throughput

Flock vs Binius64 (Plonky3/Hashcaster have no SHA-256 circuit).

```bash
cd benchmarks
./bench_sha256.sh --cooldown 20
```

This orchestrator uses a shared sweep: `HASH_LOG2S` for the MT pass, `ST_LOG2S`
for the ST pass, with per-prover `*_MAX_LOG2` caps. Flock alone:
`SHA2_LOG2S="10 12 14 16" cargo bench --bench sha2_proof`
(add `RAYON_NUM_THREADS=1` for the single-threaded column).

---

## 3. BLAKE3 throughput

Flock vs Binius64 vs Plonky3 (no Hashcaster BLAKE3 circuit).

```bash
cd benchmarks
./bench_blake3.sh --cooldown 20
```

Flock alone:
`BLAKE3_LOG2S="10 12 14 16 18" cargo bench --bench blake3_proof`
(`RAYON_NUM_THREADS=1` for single-threaded).

---

## 4. Per-phase prover breakdown

Decomposition of Flock's fast prover into its five phases (witness gen, PCS
commit, zerocheck, lincheck, recursive PCS open) at 2¹⁴, as a percentage, for
keccak / sha256 / blake3.

```bash
cd benchmarks
./breakdown.sh                             # TARGET_LOG2=14 by default
```

It runs each `*_proof` bench's inline `[prove_fast breakdown]` (via
`prove_fast_timed`, which times the real Ligerito prover including the recursive
open) and prints a side-by-side percentage table. The single-shot breakdown is
noisy — average a few runs:

```bash
for i in 1 2 3; do ./breakdown.sh 2>/dev/null | sed -n '/phase/,/breakdown total/p'; done
```

---

## 5. BLAKE3 throughput scaling

Flock's BLAKE3 throughput vs batch size at 2¹⁰/2¹²/2¹⁴/2¹⁶/2¹⁸, single- and
multi-threaded, with the MT/ST speedup at each point.

```bash
cd benchmarks
./bench_blake3_flock.sh --cooldown 20      # flock blake3, those sizes, MT + ST
```

It prints (and caches) the per-`(size, threads)` throughput:
- single-threaded throughput = the `_t1` rows,
- multi-threaded throughput = the `_t8` rows,
- speedup at each size = MT throughput / ST throughput.

(`COOLDOWN` defaults to 30 s here since the sweep includes the heavy 2¹⁸ point;
override with `--cooldown 20` to match the others.)
