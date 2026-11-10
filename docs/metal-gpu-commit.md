# Metal GPU commit

Status: on branch `metal-gpu`, behind the `metal` cargo feature. Ported from
the flock-challenge tree (`crates/flock-core/src/gpu_commit.rs` at
`c35c1c4`), where the GPU commit was that prover's largest single win.

## Summary

The PCS's level-0 commit (Reed–Solomon encoding by the additive NTT, then
the BLAKE3 Merkle tree) runs on the Apple GPU through Metal. On this
machine's 10-core M4 GPU, the GPU alone is slower than the 10-thread CPU
commit, so the commit is split between the two. At BLAKE3 2^18 on 10
threads:

| | Metal off | Metal on |
|---|---:|---:|
| BLAKE3 2^18: commit | 133 ms | 108 ms |
| BLAKE3 2^18: prove | 433 ms | 409 ms |
| leanXMSS 1000 signatures: commit | 127 ms | 104 ms |
| leanXMSS 1000 signatures: prove | 461 ms | 437 ms (2,112 → 2,222 sig/s) |

The output is byte-identical to the CPU commit. Proofs and the verifier do
not change.

## Switches

- **Build:** `--features metal` on `flock-core`, forwarded by `flock-prover`.
  It only compiles on macOS/aarch64; elsewhere the feature builds the CPU
  path unchanged. There is no crate dependency: Metal and libobjc are loaded
  at runtime through `dlopen` and `objc_msgSend`.
- **Runtime:** with the feature built in, the GPU takes every eligible
  commit; `FLOCK_METAL=0` keeps them all on the CPU (same-binary A/B).
- **Split:** tuned per shape, commit by commit (see *Design*). Two env
  vars pin it instead:
  - `FLOCK_METAL_GPU_FIRST16`: sixteenths of the first pass's tiles the
    GPU takes.
  - `FLOCK_METAL_GPU_BLOCKS`: layer-4 position blocks (of 16) the GPU
    takes. 16 runs the whole graph on the GPU.
- **Timing:** `FLOCK_COMMIT_TIMING=1` prints the split, the GPU-share fill,
  each phase's CPU and GPU time, and the GPU wait.

A commit is eligible when it uses BLAKE3 leaves and rate 1/2, has at most 64
lanes, and its codeword is at least 2^20 words. The GPU's buffers must also
be free: an earlier GPU commit's `ProverData` still holds them until it
drops, and a commit that arrives meanwhile runs on the CPU.

## The first commit, and warming up

The first GPU commit in a process pays three one-time costs:
1. **Compiling the kernels.** Metal compiles the MSL source at runtime
   (`newLibraryWithSource:`), once per process.
2. **The shape's buffers.** Allocating the message, codeword and tree
   buffers, and the first touch of their pages (about 1.2 GB at BLAKE3
   2^18).
3. **The check commit.** The first commit of each shape also runs the CPU
   commit and compares the two byte for byte (see *Correctness*).

On an M4 Max, the first BLAKE3 2^18 commit took ~265 ms, against ~46 ms
warm and ~64 ms for the CPU commit. The results in this document are warm
medians, so **a process that makes one proof is slower with Metal on.**

**`gpu_commit::warm_up(&[params])`** pays all three ahead of time. A
long-lived prover calls it at startup with the `PcsParams` it will commit
under:
- it loads Metal and compiles the kernels;
- for each shape the GPU takes, it allocates the buffers and runs the
  checked first commit on a synthetic (pseudo-random) message.

After it, commits of those shapes run warm. `warm_up(&[])` only compiles. It
returns `false` when Metal is off or unavailable, when the GPU disagreed
with the CPU (Metal is then off for the process), or when live prover data
holds the buffers. Commits stay correct in each case, on the CPU. The check
on a synthetic message is as strong as one on the witness: the dispatches,
the split and the kernels' control flow depend only on the shape, and a
pseudo-random message reaches every lane and every table entry.

**A precompiled metallib** is the other option for cost 1: compile the MSL
at build time (`xcrun metal`, `xcrun metallib`, which need the Metal
toolchain on the build machine), embed the library, and load it with
`newLibraryWithData:`. It is not built. `warm_up` covers a long-lived
prover, and a one-proof process would still pay costs 2 and 3.

## Buffers between commits

The buffers of the two most recently used shapes (`CACHED_SHAPES`) stay
allocated between commits; a third shape evicts the older one. A workload
that alternates two shapes, such as tower leaf and node commits, therefore
allocates each shape once instead of on every switch. (With one cached
shape it reallocated on every switch; that cost was not measured.) The
price is memory: up to two shapes' buffers stay resident, about 1.2 GB per
shape at BLAKE3 2^18. **`gpu_commit::release()`** frees every cached buffer
that no prover data holds. The next commit of a freed shape allocates again;
the compiled kernels, the shape checks and the tuned splits are kept.

## Design

**The graph.** For `log_d` = log2 positions:
1. a from-message pass for layers 0..4, which reads the message once and
   synthesizes the zero half of `[z, 0]`;
2. fused 4-layer passes over the codeword, the last one shorter when
   `log_d − 4` is not a multiple of 4. The kernels are the challenge's:
   - shared twiddle tables across four tiles while `s ≥ 2`;
   - a mixed final pass where its 40- and 20-bit twiddle bounds hold;
   - a half-footprint final pass otherwise;
3. leaf hashing, then the parent levels, three per dispatch while they hold
   whole 256-node groups.

**Any lane count.** The challenge's kernels hard-code 64 interleaved lanes
and 1 KiB leaves. The union commit trims empty lanes (BLAKE3 at 2^18
commits 46), so every kernel here takes the lane count `t ≤ 64`:
- element `(pos, lane)` sits at `pos·t + lane`;
- threads with `lane ≥ t` help build the twiddle tables but skip every load
  and store;
- a leaf is `16·t` bytes, one BLAKE3 chunk whose last block is short when
  `t` is not a multiple of 4.

**Buffers.**
- The CPU transposes the lane-major message straight into a persistent
  shared Metal buffer.
- The GPU leaves the codeword and the tree in two more, alongside the flat
  twiddle table.
- `ProverData` holds `CodewordBuf` / `MerkleTreeBuf`, which deref to slices.
  The open reads the GPU's buffers in place (unified memory), so there is no
  1 GiB copy-out.
- The two views share a lease: the buffers are free for the next commit once
  both drop.

**The hybrid.** Every phase splits work between the two processors, in the
same buffers:
1. **The first pass, by tile.** The pass is independent per tile `r` (its 16
   positions `(e << s) + r`). The GPU takes tiles `[0, r₁)`. The CPU takes
   the rest with `first_pass_from_message_tiles`: a copy into both halves for
   layer 0, then one fused 3-layer butterfly per half.
2. **The rest, by layer-4 block.** After layer 4 the 16 position blocks are
   independent transforms with independent subtrees. The GPU takes blocks
   `[0, k)`: every kernel derives its block from the high bits of `tgid`, so
   a prefix of the threadgroups is a prefix of the blocks. The CPU takes
   `[k, 16)`.
3. **The CPU suffix runs main's tuned NTT on each block.** A twiddle is
   F₂-linear in its block index, so a block's sub-transform has the full
   transform's twiddles plus one constant per layer.
   `AdditiveNttF128::sub_block` builds that transform: the same `evals` rows,
   with per-layer offsets added in `twiddle`. The deep pass's hook hashes
   finished sub-groups while they are cache-hot.
4. **The join.** The CPU hashes the last 7 tree levels after both sides
   finish.

**Balancing the split.** Each commit measures both sides of each phase: the
GPU's from its command buffer's own timestamps, the CPU's by wall clock
(contention included). Per shape it keeps an average cost per unit (per
sixteenth of tiles, per block) and moves the next commit's split to where
the two sides finish together, `x = 16·c / (g + c)`. The first commit of a
shape, which runs cold, is not used. At BLAKE3 2^18 it settles within two
commits at 8–9/16 of the first pass and 8 blocks. The output is identical
for every split.

**Overlapping the fill.** The CPU transposes only the GPU's message rows
before starting it (about 8 ms of the 15), then fills its own rows while the
GPU runs its first pass.

**Packing lanes.** With `t < 64` lanes, a 64-thread group used to leave
`64 − t` threads computing zeros. The from-message pass and the grouped
passes now run their 4 tiles × `t` lanes as flat work items, 64 per round:
3 rounds instead of 4 at 46 lanes. Consecutive items stay consecutive in
memory. GPU NTT at 2^20 × 46: 168–188 → 143–155 ms. The final pass, one
tile per group with its own twiddles, is not packed.

**No shared cache lines.** Every boundary between what the CPU and the GPU
write is a whole 256-byte span:
- block and tile boundaries are whole positions of `16·t` bytes, and the
  first-pass split is a multiple of 8 tiles;
- both sides stop hashing 7 levels below the root, where each block owns 8
  nodes.

**Correctness.**
- **First-commit check:** the first GPU commit of each shape in a process
  also runs the CPU commit and compares codeword and tree byte for byte. A
  mismatch turns the GPU off for the rest of the process and the CPU result
  is used.
- **Fallback:** a GPU error falls back to the CPU from the message buffer,
  which the GPU never writes.
- **CI:** the Metal code compiles only on macOS/aarch64 with the feature, so
  the Linux lint and test jobs never build it. `.github/workflows/metal.yml`
  runs clippy (`--features flock-prover/metal`, all targets, `-D warnings`)
  and the Metal unit tests on a GitHub-hosted Apple Silicon runner. Those
  runners are virtual machines; on one without a Metal device the GPU tests
  print `no Metal device; skipping` and pass, and any other Metal init error
  fails them.

## Measurements

All on the M4 (4P+6E, 10-core GPU), BLAKE3 2^18 (`2^20` positions × 46
lanes). Medians of each round's fastest prove, alternating rounds, ms.

**GPU time per stage** (`gpu_graph_timing`, GPU timestamps):

| stage | time |
|---|---:|
| each NTT pass | 30–46 |
| whole NTT | 168–188 |
| Merkle | 10–15 |

- **46 vs 64 lanes:** about the same time. The passes are compute-bound,
  and dead lanes run the network.
- **Memory traffic** alone would be about 60–70 ms.

**GPU-only commit, three rounds each:**

| threads | prove, Metal off | prove, Metal on |
|---:|---:|---:|
| 10 | 496 | 609 |
| 4 | 595 | 642 |

- **Commit:** the CPU's is 143–158 ms at either thread count, because its
  Merkle tree and deep NTT pass already run on all cores. The GPU's is
  about 190–210.
- **Why the challenge saw +39%:** yukon's A/B on the challenge tree was at 4
  threads with the split. Its ranked machine is an M3 Max with a 40-core
  GPU.

**Hybrid split sweep** (commit, ms):

| GPU first-pass share | GPU blocks | commit |
|---:|---:|---:|
| 6/16 | 6 | 142–147 |
| 6/16 | 7 | 121–125 |
| 8/16 | 7 | 126–132 |
| 6/16 | 8 | 127–130 |

At the default split:

| stage | time |
|---|---:|
| fill | 15 ms |
| first pass | 19–25 ms |
| CPU suffix (9 blocks) | 89 ms, the GPU's 7 finishing inside it |

- **CPU per-block cost:** about 10 ms, against 8.7 ms in the CPU-only
  commit. The two processors share memory bandwidth.
- **Tuned vs simple suffix:** the tuned NTT on sub-blocks measured the same
  as a simpler fused-pass suffix, which is consistent with bandwidth, not
  kernel scheduling, being the limit.

**Tuned and packed hybrid vs CPU, three alternating rounds, 10 threads,
quiet machine:**

| | Metal off | Metal on |
|---|---:|---:|
| BLAKE3 2^18: commit | 133 [133 133 135] | 108 [106 108 109] |
| BLAKE3 2^18: prove | 433 [431 442 433] | 409 [403 409 418] |
| leanXMSS 1000: commit | 127 [126 130 127] | 104 [101 104 106] |
| leanXMSS 1000: prove | 461 [461 467 452] | 437 [431 448 437] |

Faster in 3/3 rounds on both, and every other phase is unchanged. At the
tuned split for leanXMSS:
- GPU-share fill: 6.5 ms;
- first pass: 31 ms (CPU 21, GPU 24);
- rest: CPU 74 ms, GPU 67 ms.

**The first hybrid (fixed split, no packing), four alternating rounds:**

| | Metal off | Metal on |
|---|---:|---:|
| commit | 152 [164 151 144 153] | 131 [135 126 138 126] |
| prove | 486 [521 486 468 487] | 464 [461 467 490 450] |

- **Commit:** faster in 4/4 rounds.
- **Prove:** faster in 3/4 rounds.
- **Other phases:** equal.

The machine was shared during these runs, so the totals are noisier than
the commit.

## Not ported

- **The challenge's zerocheck and lincheck GPU arms** (`GPU_ZEROCHECK`,
  `ZC_BYTE_FOLD`, `GPU_LINCHECK`, `GPU_ZC_*`). They fold the RS round-1 C
  message. Main proves with the AG zerocheck, so they do not apply as
  written.
- **Witness streaming into the commit** (`WITNESS_GPU_STREAM`). It is tied
  to the challenge's BLAKE3 witness generator; main's union witness differs.
- **Every other switch** was indistinguishable from zero in yukon's A/B or
  tied to the challenge's benchmark harness:
  - the GPU grind, the recursive Merkle trees and keep-warm;
  - the embedded metallib, which only shortens init;
  - the cross-process warmup cache and the autotuner.

## Ligerito's recursive commits

They stay on the CPU. Each recursive level commits through `ligero_commit`
(`pcs/ligerito.rs`), which has its own NTT call and Merkle tree, not the
level-0 path above. At BLAKE3 2^18 (10 threads, `FLOCK_LIG_TIMING=1`) they
are 16 lanes at rates 1/8 to 1/2048:

| level | shape | encode + Merkle |
|---|---|---:|
| 1 | 2^16 columns, rate 1/8 | ~25 ms |
| 2 | 2^13, rate 1/32 | ~11 ms |
| 3 | 2^10, rate 1/128 | ~6 ms |
| 4–5 | rates 1/512, 1/2048 | ~3.5 ms |

About 45 ms per prove. Only level 1 is large enough for a split to matter.
At the level-0 split's ~18% gain that is about 4–6 ms, and the levels run
one at a time between sumcheck rounds, so the GPU work overlaps nothing. The
challenge's recursive-Merkle GPU switch measured as noise in yukon's A/B.

## GPU zerocheck round 1 (prototype, kept on a branch)

A Metal kernel for the AG round-1 sums over Full blocks: the 160 product
coordinates and both `c` banks, equal to
`round1_straight_line_program_packed_banks_fused_padded`. It was never wired
into the prover and is not on this branch. The code (`gpu_ag_round1.msl`,
the generated `gpu_ag_round1_gen.msl`, `scripts/gen_gpu_ag_round1.py`, and
the tests `ag_round1_matches_cpu` and `ag_round1_timing`) is kept on the
branch `metal-ag-round1-prototype` (at `03c955a`).
- **v1:** each thread transposes its 32 rows into plane words and runs the
  Paar encode. At 2^17 blocks it took 68–92 ms against the CPU's ~20 ms
  (10 threads). It was spill-bound: about 800 live words per thread.
- **v2:** only `a`'s and `b`'s 64 plane words stay in registers. Each
  encoded word is an unrolled XOR chain straight from the 160×64 matrix.
  Products go through threadgroup memory in three coordinate sets, each
  folded at once. 50–64 ms; 28–35 ms without the `eq` multiplications.

The remaining costs are the 128 live plane words and ~4,600 table lookups
per block for the `eq` multiplications, which the M4's GPU has no carryless
multiply for. A GPU prefix of ~25% of the blocks would save about 10 ms at
best, so it is not wired in. The challenge's experience agrees: only its
multiplication-free GPU arms paid off. The GKR wiring check's folds have the
same multiplications.

## Next

- **The rest of the fill.** The GPU's share is still transposed before it
  starts (6–8 ms). Reading the lane-major witness directly would need it
  in a Metal buffer (a no-copy wrap of the scratch allocation).
- **The final pass.** It is the one GPU pass left unpacked; packing it
  means several twiddle sets per group.

## Tests

- **`gpu_commit_matches_cpu`:** the GPU graph against the CPU commit,
  codeword and tree.
  - Positions 2^8–2^12 and lanes 64/61/6/1.
  - Every pass kernel, including the half-footprint final pass with the
    mixed one turned off.
- **`hybrid_commit_matches_cpu`:** the hybrid at 2^12 and 2^13 positions
  with lanes 64/61/6.
  - First-pass/block splits 0/1, 6/5, 16/8 and 3/15.
  - Stale bytes are written into the buffers first.
- **`sub_block_transforms_match_the_full_transform`:** the tile-split first
  pass plus `sub_block` transforms through the tuned NTT path equal the full
  transform.
- **`production_shape_matches_cpu`** (ignored, ~2.2 GB peak; run it alone
  with `cargo test --release -p flock-core --features metal --lib
  production_shape -- --ignored`): the production shape, 2^20 positions ×
  46 lanes (BLAKE3 2^18), against the CPU commit byte for byte:
  - the whole graph, with the mixed and the half-footprint final pass;
  - the hybrid at the initial split (6/16, 7), the split an M4 Max tuned to
    (5/16, 10), and the extremes 0/16, 1 and 16/16, 15;
  - `warm_up`'s checked first commit through `try_commit`, which must take
    the GPU.
- **`buffer_cache_keeps_two_shapes`:** the cache's order, reuse, eviction
  and `release` (which keeps the leased buffers), on a local cache.
- **`gpu_graph_timing`** (ignored): per-stage GPU timestamps at production
  size.
- **Without the feature:** the workspace tests, clippy and fmt pass
  unchanged. The sub-block twiddle offsets are compiled only with the
  feature (and in tests), so the default build's NTT is main's: single
  core, BLAKE3 2^18, its ranked NTT measured 449.1 vs 449.2 ms (median of
  3 alternating rounds) and prove 1.95–1.98 s on both. With `--features flock-prover/metal` the workspace tests pass
  too; large commits then take the GPU path, first-commit check included.
