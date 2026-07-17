# Flock

A Rust implementation of the **Flock** proving system: a prover and verifier for
R1CS-over-GF(2) statements, built on a zerocheck + lincheck PIOP with a
multilinear PCS (Ligerito) over the binary field F₂₁₂₈. Tuned for
Apple silicon (M-series) and AVX-512-capable x86-64 CPUs.

It ships end-to-end provers for hash-chain and Merkle-path statements over
BLAKE3, SHA-256, and Keccak-f[1600].

## Layout

Two crates, split along the prove/verify boundary:

- **`crates/flock-core`** — the protocol library and verifier (field arithmetic,
  NTT, zerocheck, lincheck, PCS, Merkle, R1CS). Carries everything needed to
  verify; portable, with scalar fallbacks for the NEON kernels.
- **`crates/flock-prover`** — the end-to-end prover: prove orchestration, the
  hash R1CS encoders, the hash-chain / Merkle-path statements, and the
  `flock_chain` CLI. Depends on `flock-core` and re-exports it.

The heavy NEON kernels live in the shared `flock-core` layer, so the verifier
runs on the same code as the prover; `flock-core` still compiles off-ARM via the
scalar fallbacks.

## Build

```sh
cargo build --release
cargo test --release
```

Requires a recent stable Rust toolchain (edition 2024). Optimized kernels target
ARM64 NEON and x86-64 AVX-512/VPCLMULQDQ, with portable fallbacks for other
targets.

## CLI — hash-chain prover

```sh
cargo build --release -p flock-prover --bin flock_chain

# Prove an 8-step BLAKE3 chain:
cargo run --release -p flock-prover --bin flock_chain -- prove \
    --hash blake3 --steps 8 --out /tmp/chain.bin

# Verify:
cargo run --release -p flock-prover --bin flock_chain -- verify --in /tmp/chain.bin
```

`--hash` accepts `blake3`, `sha2`, or `keccak`. `--steps` must be a power of two
≥ 8. Run `flock_chain help` for the full flag list (`--mode`, `--backend`, …).

## Benchmarks

Hash proving throughput on an **AMD Ryzen Threadripper 7970X** (32 physical
cores / 64 hardware threads, 256 GB RAM), measured on Linux x86-64 on
2026-07-17. The build uses `-C target-cpu=native`; the active optimized path is
**AVX-512 + VPCLMULQDQ** (the CPU also supports AVX and AVX2). Multi-threaded
runs use the 32 physical cores, without SMT.

Throughput in thousands of hashes per second (`k hashes/s`; higher is better):

| Hash | Batch | 1T row-major | 1T batch-major | 32T row-major | 32T batch-major |
|---|---:|---:|---:|---:|---:|
| SHA-256 | 1024 | 30.3 | 30.2 | 58.7 | 85.0 |
| SHA-256 | 4096 | 34.0 | 33.5 | 135.8 | 145.1 |
| SHA-256 | 16384 | 33.0 | 32.3 | 200.3 | 240.3 |
| SHA-256 | 65536 | 32.3 | 32.0 | 249.2 | 271.3 |
| SHA-256 | 262144 | 31.5 | 31.0 | 296.9 | 305.3 |
| BLAKE3 | 1024 | 34.5 | 34.1 | 105.3 | 109.0 |
| BLAKE3 | 4096 | 54.7 | 54.0 | 217.7 | 227.1 |
| BLAKE3 | 16384 | 61.4 | 62.7 | 394.7 | 411.7 |
| BLAKE3 | 65536 | 62.4 | 64.8 | 541.0 | 540.4 |
| BLAKE3 | 262144 | 64.1 | 64.8 | 633.9 | 629.8 |
| Keccak-f[1600] | 1024 | 17.9 | 19.2 | 57.5 | 54.9 |
| Keccak-f[1600] | 4096 | 18.7 | 19.9 | 109.5 | 106.9 |
| Keccak-f[1600] | 16384 | 18.1 | 19.0 | 135.4 | 141.6 |
| Keccak-f[1600] | 65536 | 18.4 | 18.6 | 156.1 | 164.1 |
| Keccak-f[1600] | 262144 | 18.3 | 17.6 | 159.9 | 164.3 |

The figures measure the full default Ligerito `prove_fast` path, including
witness generation and proof construction. SHA-256 and BLAKE3 count compression
functions; Keccak counts Keccak-f[1600] permutations. “Batch” is the number of
independent hash operations proved together. Each value is the best of three
measured proofs after one untimed warm-up; the warm-up proof is also verified.
Row-major stores each hash witness contiguously, while batch-major groups
corresponding witness chunks across the batch. The Keccak rows use the
single-permutation encoder so the two layouts are directly comparable; the
separate 3-wide Keccak benchmark remains available for maximum Keccak
throughput.

Regenerate the complete table with:

```sh
benchmarks/bench_hash_throughput.sh
```

Override `LOG2S`, `RUNS`, or `MT_THREADS` to change the batches, trial count,
or multi-threaded pool size. There are no Criterion harnesses; each Rust bench
is a no-harness binary that prints its own results. Run an individual bench
with:

```sh
cargo bench --bench blake3_proof
cargo bench --bench e2e_zerocheck
```

Always run benches **one at a time** — concurrent benches contend for cache,
memory bandwidth, and thermal headroom on a single chip. See
[`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) for the full set and the
competitor comparisons.

## Acknowledgments and third-party code

Flock incorporates code from the projects below; see the individual file
headers for the exact upstream paths and copyright notices. Both projects are
dual-licensed under Apache-2.0 OR MIT, matching Flock's own license.

**[binius64](https://github.com/binius-zk/binius64)** — Irreducible's
binary-tower field framework; the basis for our F₁₂₈ / ring-switch design.
Dual-licensed Apache-2.0 OR MIT; Copyright 2025 The Binius Developers and
Irreducible, Inc. Derived files:

- `crates/flock-core/src/field/phi8.rs` — `PHI_8_TABLE`, a verbatim copy from
  `crates/field/src/ghash.rs`.
- `crates/flock-core/src/field/gf2_128.rs` — the default `Mul`
  (`ghash_mul_binius`) ports `mul_clmul` from
  `crates/field/src/arch/shared/ghash.rs`.
- `crates/flock-core/src/field/gf2_8.rs` — the NEON 16-wide multiplier
  (`gf8_mul_vec16` / `gf8_reduce_vec16`) ports `packed_aes_16x8b_multiply` from
  `crates/field/src/arch/aarch64/simd_arithmetic.rs`.
- `crates/flock-core/src/ntt/additive_ntt_f128.rs` — algorithm skeleton
  (iterative LCH NTT, neighbors-last ordering) derived from
  `NeighborsLastReference` in `crates/math/src/ntt/reference.rs`; the
  interleaved SoA layout, fused 2-layer butterfly, and parallelization are
  original to Flock.
- `crates/flock-core/src/pcs/tensor_algebra.rs` — port of
  `crates/math/src/tensor_algebra.rs`, specialized to `F = F_2`, `FE = F_{2^128}`.
- `crates/flock-core/src/pcs/ring_switch.rs` — the verifier's polylog
  `eval_rs_eq` helper ports `crates/verifier/src/ring_switch.rs`; the rest of
  the module is original to Flock.

**[bolt-rs](https://github.com/bcc-research/bolt-rs)** — BCC Research's Ligerito
implementation; reference for our integrated Ligerito PCS backend.
Dual-licensed MIT OR Apache-2.0; Copyright (c) 2026 Bain Capital Crypto, LP and
Ron Rothblum. Derived files:

- `crates/flock-core/src/pcs/ligerito.rs` — port of `ligerito_recursive.rs` onto
  Flock primitives.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
