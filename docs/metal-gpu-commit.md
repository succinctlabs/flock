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
| commit | 152 ms | 131 ms |
| prove | 486 ms | 464 ms |

The output is byte-identical to the CPU commit. Proofs and the verifier do
not change.

## Switches

- **Build:** `--features metal` on `flock-core`, forwarded by `flock-prover`.
  It only compiles on macOS/aarch64; elsewhere the feature builds the CPU
  path unchanged. There is no crate dependency: Metal and libobjc are loaded
  at runtime through `dlopen` and `objc_msgSend`.
- **Runtime:** with the feature built in, the GPU takes every eligible
  commit; `FLOCK_METAL=0` keeps them all on the CPU (same-binary A/B).
- **Split tuning:**
  - `FLOCK_METAL_GPU_FIRST16` (default 6): sixteenths of the first pass's
    tiles the GPU takes.
  - `FLOCK_METAL_GPU_BLOCKS` (default 7): layer-4 position blocks (of 16)
    the GPU takes. 16 runs the whole graph on the GPU.
- **Timing:** `FLOCK_COMMIT_TIMING=1` prints the fill, the first pass, the
  CPU share and the GPU wait.

A commit is eligible when it uses BLAKE3 leaves and rate 1/2, has at most 64
lanes, and its codeword is at least 2^20 words. The GPU's buffers must also
be free: an earlier GPU commit's `ProverData` still holds them until it
drops, and a commit that arrives meanwhile runs on the CPU.

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

**Hybrid vs CPU, four alternating rounds, 10 threads:**

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

## Next

- **Tune the split per shape and machine.** The defaults are this machine's
  at BLAKE3 2^18. A larger GPU wants more blocks; a one-shot autotune on the
  first commit (it already runs both paths) could pick them.
- **Overlap the fill.** The 15 ms lane-major transpose runs before either
  processor starts. The GPU could start first-pass tiles as their message
  rows land.
- **Leave the dead lanes out.** With `t < 64`, a quarter of the GPU's
  threads run the network on zeros. A kernel that packs two tiles per group
  at small `t` would recover that.
- **leanXMSS.** It lives on its own branch; the commit is the same
  lane-major path, so it should take the hybrid unchanged once merged.

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
- **`gpu_graph_timing`** (ignored): per-stage GPU timestamps at production
  size.
- **Without the feature:** the workspace tests, clippy and fmt pass
  unchanged. With `--features flock-prover/metal` the workspace tests pass
  too; large commits then take the GPU path, first-commit check included.
