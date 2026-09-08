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

Native, after PR #50 (shared BLAKE3 R1CS matrices) and PR #51 (nodes share the
cached block + CSC circuit); the "before" column is main before those PRs:

| run | prove before | peak RSS before | prove after | peak RSS after |
| --- | --- | --- | --- | --- |
| tower 4 x 256 | 18.6 s | 12.5 GiB | 10.7 s | 3.49 GiB |
| leaf 256 | 1.80 s | 2.42 GiB | 1.20 s | 0.79 GiB |
| leaf 65536 | 2.09 s | 4.06 GiB | 1.44 s | 2.44 GiB |

Inside the iPhone 17 Pro simulator (iOS 26.5 runtime, same Mac, via `simctl spawn`),
after PRs #50 and #51:

| run | prove | peak RSS |
| --- | --- | --- |
| tower 4 x 256 | 13.0 s | 2.96 GiB |
| leaf 256 | 1.75 s | 0.78 GiB |

The simulator runs on the host CPU, so it proves the iOS runtime works (threads,
allocation, the C-built hash kernels) but says nothing about phone speed.

### Cold vs warm, and thread pools (2026-09-08, after PR #52)

`flock_mobile_init_threads(global, all_core)` sizes rayon's global pool (the
NTT-shaped prover phases; 0 = the performance-core count read in-process via
`sysctlbyname`) and the all-core P+E pool (the flat parallel-for passes; 0 =
every logical CPU). The leaf demo now proves twice: cold pays the per-shape
setup (circuit emission, BLAKE3 matrices, CSC lincheck circuit), warm is the
per-statement cost over the cached shape.

| leaf | pools | cold | warm | peak RSS |
| --- | --- | --- | --- | --- |
| 256 | 10 / 14 (M4 Max default) | 1.24 s | 12.6 ms | 811 MiB |
| 4096 | 10 / 14 | 1.22 s | 30.6 ms | 882 MiB |
| 65536 | 10 / 14 | 1.48 s | 214 ms | 2.42 GiB |
| 256 | 2 / 6 (a 2P+4E phone's shape) | 1.22 s | 33.2 ms | 777 MiB |
| 4096 | 2 / 6 | 1.27 s | 78.8 ms | 899 MiB |
| 65536 | 2 / 6 | 1.87 s | 566 ms | 2.55 GiB |

The setup is single-threaded and dominates a phone-sized leaf: the pool size
barely moves the cold number and scales the warm one. A production prover
pays setup once per process. Tower 4 x 256 with the global pool at 2 / 4 / 6
threads (all-core 6) proves in 11.5 / 10.9 / 10.7 s against 10.4 s at the
default 10 / 14, again mostly setup. In the simulator (host CPU) leaf 256 is
2.1 s cold / 44 ms warm at the defaults and 1.7 s / 39 ms at 2 / 6.

What a real device changes: Apple schedules threads onto core types by QoS
class rather than by count, so a 2-thread pool is not guaranteed to land on the
P-cores and 6 threads may beat the desktop heuristic. Measure 2 / 4 / 6 on the
phone through `flock_mobile_init_threads` before choosing.

The recursion node is a fixed m=29 dense circuit. Before PR #50 the tower could not
fit any phone; at 3.5 GiB the smallest tower is now within reach of an 8 GB
iPhone's per-process limit, and a leaf at 256 compressions is well inside it.
What is left in a leaf's floor is one shared copy of the 44M-nonzero BLAKE3 base
matrices (339 MiB) plus the CSC lincheck circuit (~178 MiB); narrowing the column
index type is the next lever if the phone needs more headroom.

### Where the leaf floor went before PR #50 (leaf 256, m = 22, traced 2026-09-07)

The proof itself adds 7 MiB. The floor was the BLAKE3 base-block R1CS,
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

PR #50 froze `SparseBinaryMatrix` rows behind an `Arc` and memoized the BLAKE3
base matrices once per process, which removed every duplicate above.

Open items: a real device number (an Xcode app target with signing), and the
thread-pool choice on the device (see above). The background-QoS hook for the
codeword pre-fault in `flock-core` is still gated on `target_os = "macos"` and
is a no-op on iOS; the same pthread call exists there.
