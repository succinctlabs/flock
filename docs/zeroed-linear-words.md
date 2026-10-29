# Zeroed linear words: degree-one rows out of the zerocheck

Status: implemented on branch `zerocheck-input-rows` (stacked on `leanxmss`).
Opt-in per registry type. On for every leanXMSS gate. Behind
`BLAKE3_ZERO_LINEAR=1` (inputs and outputs) or `BLAKE3_ZERO_INPUTS=1` (inputs
only) for the `blake3_proof` BLAKE3 type.

## Summary

Every Boolean gate type is a square R1CS with `C = I` and a constant-one
wire `z_pin`. Many of its constraint rows have degree one: one factor is
the constant.
- **Input words.** Each input bit has the row `z_i · z_pin = z_i`.
- **Output words.** An output defined by a copy or an XOR reads
  `(A z)_i · z_pin = z_i`.
- **The pin's word.** In a builder gate this is word 0: the pin's own row
  plus 127 rows `0 · 0 = z_i` that force unused bits to zero.

Once the lincheck's const pin fixes `z_pin = 1`, each of these rows is the
linear identity `(A z)_i = z_i`. Yet the zerocheck pays for it like any
product.

This change drops every whole 128-row word of degree-one rows from the
zerocheck:
- **A and B** lose the rows.
- **C** becomes `diag(mask)`.
- **The lincheck** checks the dropped rows' linear identity
  `(A + I)·z = 0` instead, at the zerocheck's own point.

An input word's identity is empty, because `A_i = {i}` makes
`(A + I)_i = 0`. So the identity words of the first version of this PR are
the special case with nothing to check.

## Which words

`TableType::zeroed_word_cols` lists them. Two constructors derive the list
from the type's own matrices:
- `TableType::with_zeroed_linear_words` takes every word whose 128 rows are
  all `TableType::is_linear_row`: `C_i = {i}`, and either `B_i = {pin}`, or
  `A_i` and `B_i` both empty.
- `with_zeroed_identity_words` keeps only the words of `x·1 = x` rows.

`Registry::new` re-checks every listed word. It requires a const pin, whole
useful words in ascending order, and only degree-one rows. It refuses any
word with a degree-two row. The list is absorbed into `Registry::digest`
under its own label, and only when non-empty.

What drops, per gate:

| Gate | Words | Dropped | Left in the zerocheck |
|---|---:|---|---:|
| leanXMSS compress | 92 | 0–7 (pin, inputs), 89–91 (outputs) | 81 |
| chain | 7 | all | 0 |
| swap | 8 | 0–3, 5–6 | 2 |
| codeword | 14 | 0–1 | 12 |
| address | 4 | all | 0 |
| address step | 5 | all | 0 |
| hand-written BLAKE3 | 92 | its 7 input and 4 output words | 81 |

The chain, address and address-step gates compute nothing but copies and
XORs, so the zerocheck reads none of their words. Two kinds of rows stay
because their words mix them with products: the codeword gate's words split
64 linear / 64 product rows, and the compression gate's word 8 has 70
forced-zero bits.

## The protocol

Throughout, `D` is the set of dropped rows, the words of
`UnionInstance::zeroed_word_cols`. `mask` is 0 on `D` and 1 elsewhere.

1. **Zerocheck.** It proves `a'∘b' = c'` for `a' = diag(mask)·A·z`,
   `b' = diag(mask)·B·z` and `c' = diag(mask)·z`. The run-list it reads
   (`UnionInstance::boolean_padding_spec`) marks `D`'s words as zero runs.
   Both kernels (AG and RS) are read-exact, so they read nothing there.
   This is unchanged from the identity-word version.
2. **Lincheck.** The comb's row weights `u` are the quirky row-eq table
   `eq`, with the rows of `D` scaled by a fresh challenge `η` (they were
   cleared before).
   - **Fix-ups:** after the usual fold `α·Aᵀu + Bᵀu`, an O(k) pass adds
     `α·u_i` at column `i` for each dropped row. For rows whose `B_i` is
     the pin, it also adds `u_i` at the pin column, which cancels B's
     share.
   - **Result:** the comb is that of `A'`, `B'` plus
     `α·η·Σ_{i∈D} eq_i·(A + I)_i`. The sumcheck then proves
     `α·a'(r) + b'(r) + α·η·ρ̂(r) = target`, with `ρ` the residual vector
     `(A + I)·z` restricted to `D`.
   - **Target:** the verifier's target has no `η` term, so it forces
     `ρ̂(r) = 0`.
   - **When `η` is sampled:** right after `α`, and only when some dropped
     row is not `x·1 = x` (`UnionInstance::checks_linear_rows`). It grinds
     like a pin's `β`.
3. **The C claim.** `c'(r_c) = Σ eq(r_c, x)·mask(x)·z(x)`. The opening
   carries the mask as a per-column factor in every path, as before.

## Soundness

1. **The relation.** The PIOP proves two things:
   - the kept rows, through the zerocheck;
   - `(A z)_i = z_i` for every dropped row and every instance, through the
     lincheck.

   On a declared instance the pin is 1, so the second is exactly the
   original row.
2. **The `η` batch.** `η` is sampled after `z` is committed and after the
   zerocheck has fixed `a'(r)` and `b'(r)`. The sumcheck identity is affine
   in `η`, so a nonzero `ρ̂(r)` survives with probability at most `1/|F|`.
   That is one more `β`-like site, with one PoW bit under
   `LincheckGrinding::per_challenge_128`.
3. **The point `r`.** `ρ` is fixed by the commitment before `r` is drawn.
   A nonzero `ρ` has `ρ̂(r) = 0` only if `r` hits a bad set:
   - at the skip coordinate, the zeros of a nonzero skip slice's base-code
     word: at most 158 (AG) or 63 (φ₈);
   - per multilinear coordinate, degree one.

   These add to the zerocheck's own bad sets at the same challenges.
   `ZerocheckGrinding::with_linear_check` grinds for the sums:

   | Challenge | Bound before | Bound with the check | Bits |
   |---|---:|---:|---|
   | AG `r₁` | 474 | 632 | 9 → 10 |
   | RS skip point | 127 | 190 | 7 → 8 |
   | Each round | 2 | 3 | unchanged at 2 |

   `UnionInstance::zerocheck_grinding` applies it on both sides.
4. **Dummy rows.** A slot's dummy rows `[n_t, 2^ν)` have pin 0.
   - **The original relation:** the row `(A z)_i · z_pin = z_i` forces
     `z_i = 0` there for every input and output word. The lincheck's
     identity has no pin and does not.
   - **Height-`n_t` stacking (the default):** dummy rows are not committed.
     They are virtual zeros, so nothing changes.
   - **`UnionInstance::set_full_height_commit`:** dummy rows are committed.
     The wiring and the bus then read them. Their gathers read each
     committed word's whole column and treat dummy rows as zero. A dead
     wiring leaf is `1 + w`, so a prover who could set a dummy input word
     to `w = 1` would zero both grand products and void every copy
     constraint.
   - **The rule:** a slot that commits dummy rows (full height,
     `0 < n_t < 2^ν`) keeps all its rows. `UnionInstance::zeroed_word_cols`
     returns nothing for it, and every consumer reads the list through
     that method.

   The first version of this PR dropped the input words regardless of
   commit mode. Under full height that removed this guarantee for partial
   slots. That includes every leanXMSS kind's last slot, and the pure
   BLAKE3 bench run with `BLAKE3_FULL_HEIGHT=1` at a count that is not a
   power of two.

The mask and the dropped rows come from the verifier's registry and the
public counts, never from the proof.

## Identity words are the linear check with nothing to check

For an `x·1 = x` row, `A_i = {i}` and `B_i = {pin}`:
- the `η`-scaled fold puts `α·η·eq_i` at `i` and `η·eq_i` at the pin;
- the fix-ups add the same two terms;
- in characteristic 2, they cancel.

So with only identity words dropped, the comb equals the cleared-table comb
of the first version for any `η`.
`identity_words_are_the_linear_check_with_nothing_to_check` pins this. The
implementation does not sample `η` or raise the grinding then, so an
identity-only registry runs exactly the first version's computation.

With linear words, the extra prover work is:
- one O(k) fix-up pass per distinct gate, beside a fold that touches the
  same nonzeros as before;
- one more `η` sample;
- the tenth bit on the AG `r₁` grind, about 1 ms serial (the grind is
  documented as ~1 ms at 9 bits).

See *Measurements*.

## Not supported yet

- **The deferred verification paths** (the jagged-layout export, and the
  lincheck's matrix-claims accumulation, `MatrixAssertion::claims` and
  `check_reported`) describe rows with plain eqs. With zeroed words they
  assert. A prover cannot trigger this, since the list comes from the
  verifier's registry.
- **The recursion tower** never zeroes words.
- **Partial slots under full-height commit** keep their rows. Recovering
  them would take the C claim's weight
  `eq_row ⊗ (eq_col ⊙ mask) + (eq_row ⊙ dead) ⊗ (eq_col ⊙ (1 − mask))`.
  Its row factor is masked to the dummy rows per slot, which every opening
  path would have to carry. In leanXMSS at 1000 signatures, those slots
  hold about a quarter of the droppable blocks.

## Measurements

Setup:
- M4 (4P+6E), 10 threads, three alternating rounds.
- Phase rows are the median over rounds of each round's fastest prove (ms).
- Zerocheck sub-phases are medians over all proves.

**leanXMSS, 1000 signatures** (full-height commit, ν = 15). The middle
column is the first version of this PR: input words dropped in every slot,
which is unsound under full height.

| | before (`leanxmss`) | #67 as first pushed | this version |
|---|---:|---:|---:|
| zerocheck | 156 | 144 | 140 |
| round 1 / fold / tail | 53 / 40 / 61 | 50 / 36 / 57 | 44 / 34 / 55 |
| lincheck | 29 | 29 | 31 |
| zerocheck + lincheck window | 208 | 195 | 194 |
| prove | 505 | 494 | 494 |

- **Same total as the first version, now sound.** This version drops more
  per full slot (linear outputs and the pin's word as well as the inputs),
  but nothing in the partial slots.
- **The lincheck's extra 2 ms:** the partial slots now fold their own
  combs, one more for the compression gate and one more for chain.

**Pure BLAKE3, 2^18 compressions** (one full slot):

| | nothing dropped | inputs | inputs, #67 binary | inputs and outputs |
|---|---:|---:|---:|---:|
| zerocheck | 136 | 157 | 159 | 152 |
| round 1 / fold / tail | 59 / 46 / 31 | 54 / 41 / 63 | 54 / 41 / 63 | 52 / 39 / 61 |
| lincheck | 19 | 19 | 19 | 19 |
| prove | 445 | 468 | 471 | 463 |

- **The linear check is free.** The lincheck takes 19 ms in every arm, and
  inputs-only on this binary matches the first version's binary.
- **At full utilization, dropping words was a net loss.** Round 1 and the
  fold shrink as expected, but the multilinear tail doubles.
  - The sparse tail ran its two-rounds-per-pass lookahead only when the
    live layout was a single interval starting at 0.
  - A full slot's used words are such an interval, and dropped words split
    it.
  - leanXMSS never has that layout (many slots, gaps, partial columns), so
    its tail ran without the lookahead in every arm. At 55–61 ms it was
    its largest zerocheck phase.
- **The fix belonged to the tail.** The lookahead now runs on any live
  layout (next section), which removes the regression.

## The tail's lookahead over live spans

The sparse tail used to run the lookahead ladder only on one live interval
starting at 0. On any other layout it resolved the fold's deferred
challenge with a serial in-place fold over the whole live set
(`fold_in_place_pair`), then ran the classic one-round-per-pass kernel.
At the tail's entry:
- **leanXMSS** (1000 signatures) has 34.6M live positions in 227
  intervals, aligned only to 128 positions. A partial slot's columns are
  live on `2·n_t` of their `2^(ν+1)` positions.
- **Pure BLAKE3 with dropped words** is one interval, but it starts at
  column 11.

`fold_lookahead_sparse_into` (`multilinear.rs`) runs the lookahead pass on
live-span buffers:
- Every group of inputs (16 with two pending variables, 8 with one) is
  weighted by its global index.
- A run of groups inside one stored interval goes through the dense NEON
  kernel on the compacted slice.
- A group straddling a live/dead boundary gathers its inputs with the dead
  ones at zero.
- The output layout (`lookahead_layout`) is the group cover of the live
  set, merged where groups touch. Its intervals are multiples of 4, so the
  ladder continues on any layout.

The sums and stored outputs equal the dense pass's on the expanded
buffers, so the transcript is unchanged.
`sparse_lookahead_matches_the_dense_pass` checks this for prefixes, an
offset interval, slot-shaped runs, and intervals that start, end or meet
inside a group, for both pass widths.

The gain is multicore only:

| ms | before | resolve parallel only | live-span lookahead |
|---|---:|---:|---:|
| **10 threads**, leanXMSS: tail | 56 | 43 | 24 |
| leanXMSS: zerocheck | 146 | 136 | 110 |
| leanXMSS: zerocheck + lincheck window | 203 | 194 | 184 |
| pure BLAKE3 2^18, linear words dropped: tail | 62 | 50 | 28 |
| pure BLAKE3: zerocheck | 167 | 147 | 129 |
| pure BLAKE3: zerocheck + lincheck window | 205 | 184 | 162 |
| **1 thread**, tail: leanXMSS | 92 | | 91 |
| pure BLAKE3, linear words dropped | 112 | | 114 |
| pure BLAKE3, nothing dropped (single prefix) | 126 | | 126 |

- **Ten threads:** medians of three alternating rounds' fastest proves. The
  prove was faster in 5 of 5 rounds on both workloads (median −28 ms
  leanXMSS, −20 ms pure BLAKE3). The desktop was shared, so the window is
  the steadier figure.
- **One thread:** medians of three rounds' per-phase minimum.
- **Why one thread shows nothing.** Both routes do about `3L`
  multiplications for `L` live positions:
  - fold2 passes: 2.25 per input over `L, L/4, …`;
  - classic: 1 per input for the resolve, then 2 per input over
    `L/2, L/4, …`.

  What the ladder saves is memory traffic, and the old route's resolve ran
  single-threaded over the whole live set. Both only matter when the cores
  share bandwidth and wait at pass boundaries.
- **Two effects.** The middle arm parallelizes only the resolve. It gets
  about a third of the gain, and fewer passes get the rest.
- **Pure BLAKE3 now gains from dropping words at full utilization.** Over
  the same ten-thread rounds the zerocheck has a median of 140 ms with
  nothing dropped and 129 ms with linear words dropped, against 167 ms
  before.

The first version's pure-BLAKE3 numbers (201,000 compressions,
`BLAKE3_FULL_HEIGHT=1`, zerocheck 130 → 120) were measured in a
configuration this version refuses as unsound: a partial slot committed at
full height.

## Next

- **Partial slots under full height** (*Not supported yet*).
- **The two prover-only leftovers:** the C bit-bank fold over masked
  columns, and a/b writes for dropped words.
- **The RS zerocheck's tail.** Its lookahead cascade still runs only on
  dense single-run layouts, on aarch64. The live-span pass could serve its
  sparse route too. The RS zerocheck is not the default on aarch64, so
  nothing above uses it.

## Tests

- `schedule`: `zeroed_words_are_whole_identity_words` and
  `zeroed_word_with_a_real_row_is_rejected`. The second refuses a word with
  a degree-two row.
- `lincheck::union`:
  `identity_words_are_the_linear_check_with_nothing_to_check` checks the
  comb identity above, and the exact `α·η·eq⊙(A + I)` term for an XOR word.
- `tests/union_lincheck.rs`: `dropped_linear_rows_are_checked_by_the_lincheck`.
  A witness whose XOR output bit is wrong passes the zerocheck once the word
  is dropped and is rejected by the lincheck. With only the input word
  dropped, the zerocheck rejects it. Both are run without grinding and with
  the strict schedule.
- `r1cs_hashes::blake3`: `zeroed_words_roundtrip` runs AG round trips for
  inputs only and for linear words, jagged and full height, with a partial
  slot (kept) and a full one (dropped). `zeroed_linear_words_roundtrip_rs`
  runs the RS round trip. The plain registry rejects each proof.
- `zerocheck::multilinear`: `sparse_lookahead_matches_the_dense_pass`
  checks the live-span lookahead against the dense pass (see *The tail's
  lookahead over live spans*).
- `leanxmss`: `gates_drop_their_degree_one_words` pins the table above.
  `two_signatures_prove_and_verify` asserts its proof drops linear words in
  23 full slots.
