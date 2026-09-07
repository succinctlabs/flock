# flock-mobile

C-ABI shim over the Flock prover for iOS. Exploration branch `mobile-ios`.

## Build

Targets `aarch64-apple-ios` (device) and `aarch64-apple-ios-sim` (simulator)
need full Xcode selected (`xcode-select -s /Applications/Xcode.app/Contents/Developer`):
the `sha2` assembly and `blake3` NEON build scripts ask xcrun for the iPhone SDK.
Without Xcode, `sha2` minus its `asm` feature plus `blake3` with `pure` also compile.

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --release -p flock-mobile --target aarch64-apple-ios       # libflock_mobile.a
cargo build --release -p flock-mobile --target aarch64-apple-ios-sim   # + flock_mobile_probe
```

The header is `include/flock_mobile.h`. Link the `.a` plus `-lc++`-free system
libs only; the crate has no C++ dependency.

## Probe

```sh
cargo run --release -p flock-mobile --bin flock_mobile_probe -- leaf 256
cargo run --release -p flock-mobile --bin flock_mobile_probe -- tower 4 256
xcrun simctl spawn booted target/aarch64-apple-ios-sim/release/flock_mobile_probe leaf 256
```

Peak RSS is process-wide, so take each number in a fresh process.

## Feasibility findings (2026-09-07, M4 Max, 14 threads)

Every workspace crate except `flock-cuda-ffi` compiles and links for both iOS
targets with no source changes. The interleaved Merkle SHA-256 kernel uses
`core::arch` NEON intrinsics, and the iOS target enables the `sha2` and `aes`
CPU features at compile time, so the hash hot path carries over.

| run | prove | peak RSS |
| --- | --- | --- |
| tower 4 x 256 | 18.6 s | 12.5 GiB |
| tower 8 x 256 | 49.7 s | 15.8 GiB |
| leaf 256 | 1.80 s | 2.42 GiB |
| leaf 4096 | 1.80 s | 2.48 GiB |
| leaf 16384 | 1.88 s | 2.85 GiB |
| leaf 65536 | 2.09 s | 4.06 GiB |

Inside the iPhone 17 Pro simulator (iOS 26.5 runtime, same Mac, via `simctl spawn`):

| run | prove | peak RSS |
| --- | --- | --- |
| tower 4 x 256 | 26.6 s | 10.2 GiB |
| leaf 256 | 2.46 s | 2.46 GiB |
| leaf 4096 | 2.72 s | 2.52 GiB |

The simulator runs on the host CPU, so it proves the iOS runtime works (threads,
allocation, the C-built hash kernels) but says nothing about phone speed.

The recursion node is a fixed m=29 dense circuit, so the tower cannot run on a
phone as it stands. A single leaf has a ~2.4 GiB floor, which fits the per-process
limit on 8 GB iPhones.

### Where the leaf floor goes (leaf 256, m = 22, traced 2026-09-07)

The proof itself adds 7 MiB. The floor is the BLAKE3 base-block R1CS,
`build_block_r1cs`: A has 27.3M nonzeros and B 17.2M, 44.4M in all, stored as
`Vec<Vec<usize>>` (8 bytes per nonzero, 16384 x 16384 block), so one copy is
339 MiB (397 MiB with Vec slack). Peak RSS along the leaf build:

| step | resident |
| --- | --- |
| `build_block_r1cs` (with its build temporaries) | 596 MiB |
| `TableType::from_block_r1cs` clones A/B/C | 945 MiB |
| `ShapeBuilder::finish` (transient, unattributed) | 1,209 MiB |
| `chain_shape_cached(..).clone()` deep-copies the registry | 1,912 MiB |
| `chain_blake_r1cs` builds a second BlockR1cs for the lincheck circuit | 2,230 MiB |
| `csc_lincheck_circuit` (u32 CSC, ~178 MiB) + walk | 2,399 MiB |
| prove, deferred verify, matrix check | 2,407 MiB |

Three full copies of the matrices are live at once (shape cache, shape clone,
lincheck BlockR1cs) plus two transient ones during the build. Levers, in order
of payoff: share one `Arc` copy across the cache, the clone, and the lincheck
circuit (about -700 MiB); store column indices as `u16` (columns fit; -75% per
copy); drop the BlockR1cs before cloning the TableType out of it. None of these
touch the proof system.

Open items: thread-pool sizing on a 2P+4E phone (the pool is `available_parallelism`; the macOS
perf-core sysctl and background-QoS hooks in `flock-core` are gated on
`target_os = "macos"` and become no-ops on iOS), and a real device number.
