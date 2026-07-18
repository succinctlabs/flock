# Flock in Python - a pedagogical end-to-end implementation

A from-scratch, readable Python port of **Flock** - the protocol in
`../flock-paper.pdf`, implemented for real in this repo's Rust crates
(`crates/flock-core`, `crates/flock-prover`) - written as a heavily-commented
educational reference: `galois` + `numpy`, correctness over performance. Read it next to the paper to understand *what*
the protocol does; read the Rust to see *how* to make it fast.

It proves a **batch of AND-gates wired into a hash-chain**, over the paper's own
field `GF(2^128)` (GHASH polynomial `x^128 + x^7 + x^2 + x + 1`), with a **real
hash-based polynomial commitment** (Ligero/Brakedown style, SHA-256 Merkle + Reed-
Solomon). The only cryptographic assumption is SHA-256 - so, like Flock, it is
transparent and post-quantum (no pairings, no trusted setup).

## Running

Requires Python 3 with [`galois`](https://github.com/mhostetter/galois) and `numpy`:

```bash
pip install galois numpy

python3 test_flock.py     # full test suite (math, PCS, end-to-end, soundness)
python3 flock_perf.py     # wall-clock: batch proving vs K-times separate proving
```

Some environments (e.g. conda) print harmless `OMP`/`numba`/`TBB` warnings to
stderr on import; ignore them (or append `2>/dev/null`).

Running `test_flock.py` doubles as a guided tour of the protocol: each section pretty-prints the actual math it verifies - field arithmetic, the eq weights, every sumcheck round's message/check/challenge, the PCS opening, and a phase-by-phase replay of the whole verifier - using worked examples small enough to redo by hand.

## What it proves

- **Base circuit F** (one AND gate): `a * b = c` over F2, witness `z0 = [1, a, b, c]`.
- **Batch** (`K = 2^k` instances): `A = I_K (x) A0`, so `(Az) o (Bz) = Cz` holds iff
  every instance's gate is satisfied.
- **Glue G** (hash-chain): `out_i = x_{i+1}` for all `i`, i.e. each gate's output is
  the next gate's input. Enforced as `Gz = 0`.

## Module map (to the paper)

| File | Role | Paper |
|------|------|-------|
| `field.py` | `GF(2^128)` via the GHASH irreducible polynomial | S4.3 |
| `mle.py` | multilinear extension, `eq` polynomial, variable folding | S2.1 |
| `transcript.py` | Fiat-Shamir over SHA-256 (challenges in F) | S1.1 security |
| `sumcheck.py` | generic sumcheck prover/verifier | S2.3 |
| `pcs.py` | hash-based multilinear PCS (Ligero: Merkle + Reed-Solomon) | S2.4, App. C |
| `r1cs.py` | batched AND-gate R1CS (`I_K (x) A0`) + glue matrix G | S3, S4.1, S4.6 |
| `flock.py` | end-to-end: commit -> zerocheck -> lincheck -> glue -> open | S3-S4 |
| `flock_perf.py` | batch vs K-separate wall-clock comparison | S1, S5 |
| `test_flock.py` | correctness + soundness test suite | - |

## Protocol pipeline (`flock.py`)

1. **Commit** the witness MLE `z_hat` with the PCS.
2. **Zerocheck**: `sum_i eq(r,i) * (a[i]*b[i] - c[i]) = 0` where `a=Az, b=Bz, c=Cz`
   - the rank-1 (Hadamard) constraint. Sumcheck of degree 3.
3. **Lincheck**: reduce the three claims `a(r_y), b(r_y), c(r_y)` to a single claim on
   `z_hat`, batched with a random `alpha`. Uses the block-diagonal collapse (S4.1):
   the matrix marginal is computed on the small base matrix `A0`, not the K-times
   larger `A`.
4. **Glue**: prove the hash-chain `Gz = 0`, yielding another `z_hat` claim (S4.6).
5. **Open** `z_hat` at the lincheck and glue points; the PCS values must match the
   claims the sumchecks produced.

All steps share one SHA-256 transcript, so `prove()` returns a non-interactive proof.

## What is deliberately simplified

This is a teaching artifact, not the optimized Rust prover. It omits the S4
performance work whose point is CPU/cache behavior (invisible in Python):
**univariate skip** (S4.2), **friendly challenges** (S4.3 geometric progression),
**circuit walking** (S4.5), and the **ring-switching** dense->Boolean PCS transform
(App. B). The glue uses a generic linear check rather than Flock's tailored shift
argument. The protocol *structure* and soundness logic, however, are faithful.

## Implementation notes (galois pitfalls)

Two classic `galois` traps, documented where they bite in the code:

- **In-place ops mutate shared constants.** Galois scalars are 0-d numpy arrays,
  so `acc = ONE; acc *= y` silently corrupts the global `ONE`. Always accumulate
  from a fresh `GF(1)` with non-in-place ops (see `field.py`, `sumcheck.py`).
- **Field elements are not integers.** Bit tables must be lifted with `GF(...)`
  before arithmetic, and `GF([...])` construction wants plain ints, not 0-d
  galois scalars - hence the explicit `int(...)` round-trips (see `transcript.py`).

## Soundness demonstrated (`test_flock.py`)

- corrupting an AND-gate output -> **zerocheck** rejects;
- breaking the chain (valid gates, but `out_i != x_{i+1}`) -> **glue** rejects;
- tampering a PCS evaluation or a committed column -> **PCS** rejects;
- tampering a sumcheck message -> **sumcheck** rejects.

## Benchmark results

Verbatim output of `python3 flock_perf.py` (single run on an otherwise idle machine):

```text
   K | prove batch  prove sep speedup | verify batch verify sep speedup | #proofs b/s
--------------------------------------------------------------------------------------------
   2 |      23.3ms     22.7ms   0.97x |      213.6ms    284.5ms   1.33x |    1/2
   4 |      31.2ms     45.9ms   1.47x |      287.0ms    567.9ms   1.98x |    1/4
   8 |      44.7ms     88.1ms   1.97x |      355.7ms   1106.8ms   3.11x |    1/8
  16 |      80.7ms    168.4ms   2.09x |      456.6ms   2208.6ms   4.84x |    1/16
  32 |     173.5ms    363.9ms   2.10x |      593.2ms   4447.3ms   7.50x |    1/32
  64 |     463.3ms    702.5ms   1.52x |      903.3ms   8863.5ms   9.81x |    1/64

note: SEPARATE cannot prove the hash-chain (out_i = x_{i+1}) - that
cross-instance binding exists only in the BATCH witness (glue G).
```

The paper's headline survives even in unoptimized Python: one batch proof beats K separate proofs (prove up to ~2.1x, verify up to ~9.8x at K = 64), the verifier checks 1 proof instead of K, and the K separate proofs cannot express the cross-instance chain at all.

**Hardware:** MacBook Pro, Apple M1 Max - 10 cores (8 performance + 2 efficiency), 64 GB RAM, macOS 15.7.4; Python 3.8.5 (conda) with `galois` 0.4.10, `numpy` 1.24.4.

**Available memory at run time** (captured immediately before the run, 2026-07-02 11:33 +0800):

- `memory_pressure`: system-wide memory free percentage 42% (~27 GB of the 64 GB available).
- `vm_stat` (16 KiB pages): 3,734 pages free plus 875,469 pages inactive (~13.4 GB directly reclaimable).
  macOS deliberately keeps raw free pages near zero and reclaims inactive/compressed memory on demand, so `memory_pressure`'s 42% is the meaningful availability figure.
- `sysctl hw.memsize`: 68,719,476,736 bytes (64 GB) total.
