# Flock

A Rust implementation of the **Flock** proving system: a prover and verifier for
R1CS-over-GF(2) statements, built on a zerocheck + lincheck PIOP with a
multilinear PCS (Ligerito) over the binary field F₂₁₂₈. Tuned for
Apple silicon (M-series) and AVX-512-capable x86-64 CPUs.

It ships end-to-end provers for batched hash statements (BLAKE3 and SHA-256
compressions) and a recursion tower that folds proofs into proofs.

## Layout

The workspace contains these crates:

- **`crates/flock-field`** — binary field types and architecture-specific field kernels.
- **`crates/flock-hash`** — shared hash types and compression primitives.
- **`crates/flock-transcript`** — Fiat-Shamir challengers and transcript recording.
- **`crates/flock-merkle`** — generic Merkle construction and optimized hash kernels.
- **`crates/flock-multilinear`** — field-generic multilinear evaluation,
  equality tables, and folds. Index order is an explicit API parameter.
- **`crates/flock-parallel`** — the shared all-core rayon pool.
- **`crates/flock-core`** — the protocol library and verifier.
  It contains the NTT, PIOPs, PCS, and R1CS machinery.
- **`crates/flock-prover`** — the end-to-end prover: prove orchestration, the
  hash R1CS encoders, the Merkle-path statements, and the recursion tower.
  Depends on `flock-core` and re-exports it.
- **`crates/flock-cuda-ffi`** — the optional interface to CUDA prover kernels.

Architecture-specific field and Merkle kernels live in their owning crates.
Other protocol kernels remain in `flock-core`. Portable fallbacks support other targets.

## Build

```sh
cargo build --release
cargo test --release
```

Requires a recent stable Rust toolchain (edition 2024). Optimized kernels target
ARM64 NEON and x86-64 AVX-512/VPCLMULQDQ, with portable fallbacks for other
targets.

## Benchmarks

Hash proving throughput on an **AMD Ryzen Threadripper 7970X** (32 physical
cores / 64 hardware threads, 256 GB RAM), measured on Linux x86-64 on
2026-07-17. The build uses `-C target-cpu=native`; the active optimized path is
**AVX-512 + VPCLMULQDQ** (the CPU also supports AVX and AVX2). Multi-threaded
runs use the 32 physical cores, without SMT.

Throughput in thousands of hashes per second (`k hashes/s`; higher is better):

| Hash | Batch | 1T | 32T |
|---|---:|---:|---:|
| SHA-256 | 1024 | 30.2 | 85.0 |
| SHA-256 | 4096 | 33.5 | 145.1 |
| SHA-256 | 16384 | 32.3 | 240.3 |
| SHA-256 | 65536 | 32.0 | 271.3 |
| SHA-256 | 262144 | 31.0 | 305.3 |
| BLAKE3 | 1024 | 34.1 | 109.0 |
| BLAKE3 | 4096 | 54.0 | 227.1 |
| BLAKE3 | 16384 | 62.7 | 411.7 |
| BLAKE3 | 65536 | 64.8 | 540.4 |
| BLAKE3 | 262144 | 64.8 | 629.8 |

The figures measure the full default Ligerito `prove_fast` path, including
witness generation and proof construction. SHA-256 and BLAKE3 count compression
functions. “Batch” is the number of
independent hash operations proved together. Each value is the best of three
measured proofs after one untimed warm-up; the warm-up proof is also verified.
The SHA-256 and
BLAKE3 encoders have shrunk substantially since this table was measured
(2026-08-14 zk.golf-derived fused adders), so current numbers run higher —
regenerate before quoting.

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

- `crates/flock-field/src/phi8.rs` — `PHI_8_TABLE`, a verbatim copy from
  `crates/field/src/ghash.rs`.
- `crates/flock-field/src/gf2_128.rs` — the default `Mul`
  (`ghash_mul_binius`) ports `mul_clmul` from
  `crates/field/src/arch/shared/ghash.rs`.
- `crates/flock-field/src/gf2_8.rs` — the NEON 16-wide multiplier
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
