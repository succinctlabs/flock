# Layout probes (E0/E1 of ../witness-layout-plan.md)

Standalone crate (outside the repo workspace; `autoresearch/` is gitignored).

```sh
# E0 oracle + per-hash builder/stripe lockstep tests
cargo test --release

# E1 witness-gen producer bench — full spectrum (methodology: plan §4):
cargo run --release --bin e1_witness_gen -- \
  --hash all --m 23,26,29 --iters 5 --groups auto --tsv ../results/out.tsv
RAYON_NUM_THREADS=1 cargo run --release --bin e1_witness_gen -- \
  --hash all --m 23,26,29 --iters 5 --groups auto --tsv ../results/out.tsv
```

- `src/layout.rs` — L1′ layout descriptor: address maps, bit permutation,
  claim-point relabeling, word-transpose reference (E0).
- `src/bit_helpers.rs` — copies of common.rs's pub(crate) bit packers.
- `src/{keccak,sha2,blake3}_witness.rs` — copies of the private per-block
  builders (held byte-identical to the public drivers by
  `tests/e1_builder_lockstep.rs`).
- `src/producer.rs` — hash-generic row-major baseline vs staged L1′ producers
  (useful-chunk flush bound, fused lincheck stripe, NT stores, `auto_group`).
- `src/{keccak,sha2,blake3}_vwide.rs` + `direct_common.rs` — the C2
  **direct-write** producers (V = 8 lockstep SIMD, NT chunk-row emission
  straight to L1′, inline stripe). Validated by
  `tests/keccak_vwide_lockstep.rs`. This is the producer design of record.
- Results: §7 of `../witness-layout-plan.md`; raw TSV in `../results/`.
