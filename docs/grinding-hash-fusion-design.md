# Fused grinding and Fiat--Shamir squeeze

Status: implemented and exercised end to end on 2026-08-12. (References to
`tests/circuit_merkle.rs` below are historical — that file has since been
productionized into `src/tower.rs`.)

This note specifies the optimized grinding transcript, explains the security
argument it relies on, maps the construction to code, and records the isolated
benchmark. It supersedes the earlier proposal in which a grinding site used a
pre-nonce chain digest, a standalone BLAKE3 PoW hash, and a second squeeze.

## Result

A scalar grinded challenge now uses the same one recursive BLAKE3 compression
row as an ordinary scalar challenge. Grinding is not hash-free: the native
prover still searches about `2^lambda` nonce candidates. The optimization
removes accepted-proof verification rows by reusing the challenge squeeze.

| Construction | BLAKE3 rows around a scalar challenge | Extra rows due to grinding |
| --- | ---: | ---: |
| ordinary squeeze | 1 | 0 |
| former grinding verifier | 3 | 2 |
| fused grinding verifier | 1 | 0 |

The recursive circuit still has non-BLAKE constraints for the 64-bit nonce and
the leading-zero prefix. A wide vector squeeze has three challenge words in
its fused first row rather than four, so some vector widths can require one
additional continuation row.

## Exact transition

Let the live chained transcript be `(cv, pending)`, where `cv` is 256 bits and
`pending` is a 16-byte-aligned string shorter than 64 bytes. Let `w` be the
64-bit nonce and `lambda <= 128` the requested difficulty. Form one 64-byte
message block

```text
M = pending || LE64(w) || 0^64 || zero padding to 64 bytes.
```

The zero high half makes the nonce exactly 64 bits. The recursive circuit
constrains this padding rather than trusting proof serialization.

Difficulty and real message length are bound through the compression counter:

```text
L       = len(pending) + 16
counter = 0xF10C500000000000 OR (L << 32) OR lambda
O       = BLAKE3Compress(cv, M, counter, block_len = 64,
                        flags = CHAIN_SQUEEZE).
```

View the 64-byte output as four field words:

```text
O = (O0, O1, O2, O3),       Oi in F_2^128.
```

Their roles are:

```text
PoW predicate       P       = O1
first challenge words       = O0, O2, O3
next chaining value cv'     = O2 || O3
require prefix_lambda(P)     = 0^lambda.
```

For a scalar squeeze, the protected algebraic challenge is `r = O0`; `r`,
`P`, and `cv'` occupy disjoint output words. For a vector squeeze, the first
three challenges are `O0, O2, O3`, matching the repository's pre-existing
practice of exposing words that also continue the transcript state.

Further vector words come from ordinary continuation rows:

```text
Oj+1 = BLAKE3Compress(cvj, 0^512, 0, block_len = 0,
                     flags = CHAIN_SQUEEZE),
cvj+1 = low_256(Oj+1),
```

with all four 128-bit output words exposed as challenges. A zero-bit site uses
the same fused transition but accepts only the canonical nonce `w = 0`.

If appending the nonce leaves more than one block pending, ordinary absorb
rows process every block except the last. An exactly full final block is held
for the fused row; this is what preserves the one-row scalar result.

## Why the challenge is not biased

The predicate and scalar challenge must not be the same output word. Requiring
the first `lambda` bits of the algebraic challenge itself to be zero would
restrict it to `2^(128-lambda)` values and change a root-counting bound to

```text
Pr[f(r) = 0] <= degree(f) / 2^(128-lambda).
```

Instead, the implementation checks `P = O1` and uses the disjoint word
`r = O0`. Under the random-function/XOF assumption for the domain-separated
compression outputs, for any bad challenge set `B` of size at most `d`, one
nonce attempt satisfies both conditions with probability at most

```text
Pr[prefix_lambda(Pw) = 0 and rw in B]
    <= 2^(-lambda) * d / 2^128.
```

Thus searching for a nonce that both passes PoW and produces a bad algebraic
challenge retains the intended multiplicative `2^lambda` work factor. The
assumption is the same style of idealized BLAKE3-compression assumption already
used by the custom chained Fiat--Shamir transcript, extended to disjoint words
of one compression output.

The compression binds the nonce to the full transcript: `cv` commits to all
prior complete blocks and `pending` is included directly in `M`. Hashing only
the nonce would be unsound because a valid nonce could be precomputed and
reused; materializing a separate digest first is unnecessary.

## Recursive-circuit relation

The fused compression is an ordinary row of the shared BLAKE3 table. The
recursive circuit wires:

```text
cv input       <- preceding chain row (or IV)
message input  <- pending transcript words and nonce word
params input   <- fixed (counter, 64, CHAIN_SQUEEZE)
predicate      <- output word O1
next cv        <- output words O2, O3
challenge      <- output O0, then O2, O3, then continuation outputs.
```

The bit-spread table enforces

```text
nonce.hi = 0
predicate AND leading_zero_mask(lambda) = 0.
```

For `lambda = 0` it instead enforces the whole nonce word is zero. There is no
standalone BLAKE3 PoW row in the recursive circuit.

## Code map

- `crates/flock-transcript/src/challenger.rs`
  - `pow_squeeze_counter` defines the domain-separated counter.
  - `B3Chain::pow_candidate_output` defines the raw fused compression.
  - `B3Chain::apply_pow_squeeze` verifies the predicate, allocates output
    words, and advances the state.
  - `B3Chain::grind_pow_squeeze_into` performs the SIMD/parallel nonce search.
  - the four `*_pow_and_sample_*` trait methods make the protected sample an
    atomic prover/verifier operation.
- `crates/flock-transcript/src/transcript_record.rs`
  - `TranscriptOp::Pow` is the fused nonce marker immediately preceding its
    squeeze; `LegacyPow` preserves the generic non-chained fallback.
- `crates/flock-prover/src/r1cs_hashes/fs_chain.rs`
  - `FsChainSponge::finalize_pow` reproduces the row, exact challenge-word
    sources, and the high-half chaining transition.
  - `CvSource::RowHi`, `squeeze_words`, and `block_word_counts` carry those
    choices explicitly into circuit construction.
- `crates/flock-prover/tests/circuit_merkle.rs`
  - `emit_fs_chain` wires the exact row inputs and high-half chain link.
  - `emit_pow_checks` emits only nonce-width/canonicality and prefix masks.
  - `emit_recorded_pow_checks` locates every fused predicate and nonce in a
    recorded verifier transcript.

All active production prover/verifier sites use the atomic fused APIs. The
generic `grind_pow`/`verify_pow` APIs remain for compatibility and record as
`LegacyPow`; they are not used by the recursive production paths.

## Validation

The validation covers four distinct boundaries:

- native prover/verifier lockstep for scalar and multi-row vector squeezes;
- SIMD nonce-search output versus the scalar raw compression;
- recorded transcript versus `FsChainSponge`, including nonce placement,
  predicate word, challenge-word map and high-half continuation;
- recursive R1CS acceptance of a valid nonce and rejection of an invalid
  nonce, followed by full recursive proving and verification.

The merged-chain differential now checks every scalar and vector challenge
word. The independent BLAKE-row model counts each fork as a separate IV-rooted
chain and asserts equality with the generated trace.

End-to-end results under the original 448-query Johnson/list-decoding Ligerito
geometry:

- `mvp11_two_to_one_recursion_node`: passed;
- `mvp12_recursion_tower`: passed through four leaves, two level-one nodes and
  one level-two node;
- native core library: 479 passed, 22 ignored;
- prover library: 76 passed, 23 ignored;
- `circuit_merkle` default set: 5 passed, 41 ignored, including the raw-mask
  differential and valid/invalid recursive-R1CS proof test.

## Isolated benchmark

The benchmark keeps the original Fast Ligerito TOMLs in both columns. The
control uses the Fast/no-new-Secure-grinding policy; the grinding column keeps
`PcsParams.profile = Secure`, and therefore all 128-bit algebraic grinding
policies, while temporarily substituting only those original TOMLs. The final
comparison used three online repetitions over one already-built shape per
column (`TOWER_STEADY=2`), so circuit/setup construction is excluded equally.

| 2-to-1 node metric | optimized control | optimized + Secure grinding | delta |
| --- | ---: | ---: | ---: |
| median online proving | 208 ms | 217 ms | +9 ms (+4.3%) |
| median outer prove component | 195 ms | 203 ms | +8 ms (+4.1%) |
| median native verification | 10 ms | 11 ms | about +1 ms |
| outer proof | 291.5 KiB | 295.2 KiB | +3.7 KiB (+1.3%) |
| BLAKE rows | 24,451 | 24,455 | +4 (+0.016%) |
| BLAKE slot / circuit cell | `nu=15`, `mu=23` | `nu=15`, `mu=23` | no capacity jump |

The earlier, unfused Secure-grinding measurement was 27,559 BLAKE rows and a
302.0 KiB proof. Fusion and keeping PoW witness material private therefore
remove 3,104 rows (11.3%) and 6.8 KiB (2.3%) from that implementation. The
remaining four-row control delta is two extra `H(publics)` rows per child;
there is no per-grind BLAKE-row term. The census reports zero standalone PoW
BLAKE rows. The remaining time/proof-size delta comes from native nonce search,
more bit-spread relations and the extra proof fields, not hash rows.

At level two, the fused Secure/list-decoding tower used 27,468 BLAKE rows and a
306.0 KiB proof and stayed at `nu=15`, `mu=23`.

All temporary TOML substitutions were restored after the runs; the checked-in
Secure Ligerito configuration remains unchanged.

## Remaining caveats

- This is a transcript/proof-format change; old and new proofs are not meant
  to interoperate.
- The multi-output random-function/XOF assumption should remain explicit in
  any formal security write-up.
- Native prover grinding work is intentionally not reduced. The measured
  recursive-verifier savings must not be confused with PoW search cost.
- Ligerito/list-decoding and proximity-gap soundness remain the next separate
  security task; this optimization changes their transcript cost, not their
  soundness analysis.
