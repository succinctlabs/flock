# cuda-ghash — the flock GPU prover (Blackwell, `clmad`)

The CUDA side of flock. What began as a port of `flock::field::F128`'s GHASH
arithmetic (using NVIDIA's native carryless multiply-add **`clmad`**, PTX ISA
9.3; SASS `CLMAD.LO`/`CLMAD.HI` on Blackwell `sm_120`) has grown into a full
GPU prover for the direct-shape BLAKE3 statement: witness generation, commit
NTT + Merkle, zerocheck (round 1/2/tail), lincheck, the Ligerito F256
recursion ladder, the Fiat–Shamir challenger, and PoW grinding all run
on-device, validated bit-for-bit against the Rust implementation via dumped
test vectors.

## Field

GF(2¹²⁸) in GHASH form, irreducible `p = x¹²⁸ + x⁷ + x² + x + 1`, layout
`lo = x⁰..x⁶³`, `hi = x⁶⁴..x¹²⁷` — identical to the Rust `F128`. `clmad`
maps onto it naturally: a 64×64→128 carryless product is one `clmad.hi` +
one `clmad.lo`, and GHASH's pervasive cross-term/reduction XORs fold into
`clmad`'s free `^ c` operand. See `f128.cuh`. `f256.cuh` builds the F256
tower on top; `phi8_table.cuh` carries the φ8 tables the RS skip still uses.

## Layout

By prefix rather than per file (the directory is ~70 files):

| Prefix / file | Purpose |
|------|---------|
| `f128.cuh`, `f256.cuh`, `phi8_table.cuh` | field arithmetic (four F128 multiply strategies; F256 tower) |
| `*_witness.cuh`, `sha256.cuh` | on-device witness generation (BLAKE3 is the prove path) |
| `ntt_*.cuh`, `merkle*.cuh/hpp`, `challenger.hpp`, `zc_challenger_device.cuh`, `pow_grind.cuh` | commit NTT, Merkle trees/openings, FS challenger, PoW |
| `zerocheck_*.cuh`, `lincheck.cuh`, `sumcheck_ab.cuh`, `induce_sumcheck.cuh`, `introduce_glue.cuh`, `ligerito_f256.cuh` | the PIOP + Ligerito recursion kernels |
| `prove_ffi.cu` | the `extern "C"` prove entry consumed by `crates/flock-cuda-ffi` |
| `test_*.cu/.cpp` | correctness tests — most load `*_vectors.bin` oracles emitted by the Rust `dump_*` bins (`crates/flock-prover/src/bin/`); some are self-checking A/Bs |
| `bench_*.cu` | benchmarks (mirroring the Rust `benches/` where one exists) |

The Rust-side oracle generators live in `crates/flock-prover/src/bin/dump_*`;
`make -k run-tests` (what `.github/workflows/cuda.yml` runs on the Blackwell
box) regenerates the vectors it needs and runs the test set listed at the
`run-tests:` target in the `Makefile`.

## Requirements

- A `clmad`-capable `ptxas` (CUDA 13.3 build `V13.3.33`+ works).
- **Always compile AOT** (`-gencode arch=compute_120,code=sm_120`, as in the
  Makefile): `ptxas` assembles `clmad` → SASS at build time, so the GPU
  driver's PTX-JIT version is irrelevant. Do **not** rely on runtime PTX JIT.
- An NVIDIA Blackwell GPU (`sm_120`); `clmad` itself needs `sm_80`+.

## Usage

```bash
make -k run-tests   # build + regenerate Rust vectors + run the CI test set
make test           # the original F128 field correctness check
make bench          # field benchmarks
make sass           # confirm the hot loops emit native CLMAD instructions
make clean
```

Vector regeneration runs `cargo run --bin dump_*` from the repo root, so the
Rust toolchain is required. `cargo bench --bench field` gives this host's CPU
(PCLMULQDQ / NEON PMULL) numbers to compare against the GPU. The GPU↔Rust
end-to-end roundtrips live on the Rust side:
`cargo test -p flock-cuda-ffi --features gpu -- --ignored gpu_roundtrip_m`.
