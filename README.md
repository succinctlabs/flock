# Flock

A Rust implementation of the **Flock** proving system: a prover and verifier for
R1CS-over-GF(2) statements, built on a zerocheck + lincheck PIOP with a
multilinear PCS (Ligerito) over the binary field F₂₁₂₈. Tuned for
Apple silicon (M-series).

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

Requires a recent stable Rust toolchain (edition 2024). All optimizations target
`aarch64-apple-darwin`; the code compiles on other targets but the NEON paths are
gated to ARM64.

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

There are no Criterion harnesses; each bench is a no-harness binary that prints
its own table. Run one with:

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
