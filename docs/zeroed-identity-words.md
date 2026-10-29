# Zeroed identity words: input rows out of the boolean PIOP

Status: implemented on branch `zerocheck-input-rows` (stacked on `leanxmss`).
Opt-in per registry type. Turned on for the leanXMSS gates and, behind
`BLAKE3_ZERO_INPUTS=1`, for the `blake3_proof` BLAKE3 type.

## Summary

A Boolean gate's input words carry only trivial constraint rows. For each
input bit `z_i` the square R1CS (`C = I`) has the row `z_i · z_pin = z_i`,
where `z_pin` is the type's constant-one wire. Once the lincheck's const pin
fixes `z_pin = 1`, that row holds for every witness. Yet the zerocheck pays
for it like any other row: the 7 input words of a BLAKE3 compression are
7.6% of its blocks, and in leanXMSS the gadget slots' inputs bring the total
to about 10%.

This change drops those rows from the boolean PIOP:
- `A` and `B` lose them.
- `C` becomes `diag(mask)`.
- The zerocheck skips their blocks.

The relation's satisfying witnesses are unchanged. What changes is that the
C claim is no longer a plain evaluation of `z`: it is `z` with the zeroed
columns masked out. So the opening's weight for that claim carries a
per-column mask.

## Which words

`TableType::zeroed_word_cols` lists them. `TableType::with_zeroed_identity_words`
derives the list from the type's own matrices: a word-column qualifies if
all 128 of its rows are exactly `A = {i}`, `B = {const_pin}`, `C = {i}`.

`Registry::new` re-checks every listed word. It requires:
- a const pin;
- whole useful words, ascending;
- only such rows.

It refuses anything else, so a word holding a real constraint can never be
dropped. Words that merely are gate inputs do not qualify on that basis; only
the matrices decide. The list is absorbed into `Registry::digest` after the
IO schema, under its own label, and only when non-empty, so every existing
registry keeps its digest.

For the leanXMSS gates every input word qualifies:
- compression: words 1–7;
- chain: 4 words;
- swap: 3;
- codeword: 1;
- address: 2;
- address step: 2.

For the hand-written BLAKE3 R1CS the qualifying words are cv, m and the
params word (word-columns 0–1, 4–7 and 8). The `1·1 = 1` row sits in a mixed
column and stays.

## The protocol

Throughout, `mask` is 0 on the zeroed word-columns of every boolean slot and
1 elsewhere. `UnionInstance::zeroed_columns` lists the zeroed columns in the
opening's column numbering, which is `(o_t >> (7 + ν)) + w` for word `w` of
slot `t`, the numbering of `jagged_heights`.

1. **Zerocheck.** The run-list the zerocheck reads
   (`Instance::boolean_padding_spec`) marks the zeroed words as zero runs.
   The AG zerocheck is read-exact against it: Dead blocks are never read and
   count as zero. That is exactly the relation with those rows dropped,
   `a' = b' = c' = 0` there, with no kernel change.
   - **`c` side:** `c` aliases `z` through the same run-list, so `c_eval` and
     `s_hat_v_c` come out masked.
   - **Verifier:** its zerocheck takes no run-list and does not change.
   - **Committed layout:** the global `padding_spec`, which the commitment and
     the ring switch use, keeps the zeroed words as data. They are committed
     like any other word.
2. **Lincheck.** A type's row-eq table at the zerocheck's point has the
   zeroed rows cleared before it is folded through the type's
   `LincheckCircuit` (`lincheck::union::type_eq_inner`). That folds `A'` and
   `B'` without touching any circuit implementation. Prover and verifier
   share the helper. The prover's per-gate dedupe keys on the zeroed list as
   well as the circuit.
3. **The C claim.** `c'(r_c) = Σ_x eq(r_c, x)·mask(x)·z(x)`. The mask depends
   only on the column bits, so the claim's weight is
   `eq(r_row, ·) ⊗ (eq(r_col, ·) ⊙ mask)`.
   - **Why the opening can carry it:** the opening already builds every
     ring-switched claim's weight as a per-column factor times a row tensor.
     A 0/1 mask is fixed by Frobenius and passes through the F₂-linear fold
     map, so it enters as one multiply on the column factor.
   - **Prover:** `pcs::open_batch_merged` takes `zeroed_cols` per RS claim and
     applies it in each of its paths: the block-first transport's `eq_cols`,
     the materialized weight's `bake`, the multipoint's dual values, the
     aligned closed-form `coltabs`, the lazy and dense weight builds, and the
     anchor.
   - **Verifier:** the verifier's anchor (`frobenius_statements`) sums masked
     column weights through the existing `weights_columns_at`.
   - **What does not change:** the ring-switch verification, and the AB
     claim, which is unmasked.

The verifier takes the mask from its own registry, never from the proof.

## Soundness

The relation `R'` (rows dropped) and `R` have the same satisfying set:
- Each dropped row reads `z_i·z_pin = z_i`.
- The union lincheck's const-pin term forces `z_pin = 1` on every live row,
  so `z_i·1 = z_i` holds for all `z_i`.
- The pin's own row (`i = pin`, `z_pin² = z_pin`) holds for every bit.

The PIOP for `R'` is the existing one with `A' = diag(mask)·A`,
`B' = diag(mask)·B` and `C' = diag(mask)`:
- The zerocheck proves `a'∘b' = c'` row by row.
- The lincheck reduces the `a'`, `b'` claims to a `z` claim through `A'`, `B'`.
- The opening proves `c'(r_c)` as an evaluation of `z` against the public
  weight `eq ⊙ mask`.

The opening's soundness is for an arbitrary public weight: the packed-direct
groups already open against arbitrary per-column weights. So nothing new is
assumed. The mask is registry-static and digest-bound.

## Not supported yet

- **The deferred verification paths** (the jagged-layout export and the
  lincheck's matrix-claims accumulation) describe rows and columns with
  plain eqs. With zeroed words they refuse: they assert, which a prover
  cannot trigger, since the mask comes from the verifier's registry.
- **The recursion tower** never zeroes words, so it never reaches them.

Carrying a masked claim through accumulation needs:
- a masked `JaggedRowWeight`;
- its in-circuit evaluation in the tower's fold region;
- the walkers' native replicas, which today rely on the column weights
  summing to 1.

## Measurements

201,000 BLAKE3 compressions (1000 leanXMSS signatures), 10 threads on an M4
(4P+6E). Three alternating rounds, each value the median of the rounds'
fastest proves (ms):

| | Before | After |
|---|---:|---:|
| pure BLAKE3: zerocheck live blocks | 288,880 | 266,900 (−7.6%) |
| pure BLAKE3: zerocheck | 130 | 120 |
| pure BLAKE3: prove | 451 | 438 |
| leanXMSS: zerocheck live blocks | 308,870 | 276,925 (−10.3%) |
| leanXMSS: zerocheck | 154 | 145 |
| leanXMSS: zerocheck + lincheck window | 205 | 197 |
| leanXMSS: prove | 495 | 490 |

In leanXMSS the zerocheck's first round and fold drop as expected. Its
later rounds overlap the wiring GKR, which keeps its ~37 ms of work, so they
do not get faster.

Two further prover-only savings are left:
- **The C-claim bit-bank fold** could skip the zeroed columns (about 10% of
  its 24 ms).
- **The packers** could stop writing `a`/`b` for zeroed words, which no
  one reads.
