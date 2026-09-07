# 128-bit component-security audit

> **Historical-command note (2026-08-15):** `--test circuit_merkle`
> invocations below predate the tower's productionization —
> `tests/circuit_merkle.rs` now lives in `src/tower.rs`. The mvp*/envelope
> test names quoted below were retired with the productionization — the
> surviving end-to-end tower tests are `chain_spine_converges` and
> `chain_tower_e2e_with_lane` (`cargo test --release -p flock-prover --lib
> tower:: -- --ignored`). Commands are kept verbatim as an audit record.

> **Profile-consolidation note (2026-08-27, proof-IO v22):** the grind-free
> `Fast`/`Slim` audited below were deleted; the `Fast128`/`Slim128`
> schedules (audited under those names in the 08-13/08-14 passes: aggressive
> +2/level ladder, 16-bit query PoW at every level, larger deep-level batch
> grinding) now carry the `Fast`/`Slim` names. Where this document says
> "Fast uses `lambda_query = 0`" or tabulates Fast `Q` at 279/136/91/…, it
> describes the deleted profile; today's `Fast` grinds 16 bits per level
> like Slim (query term ≥ 112 bits, work-normalized ≥ 128) and its ladder
> rates are 1,3,5,7,9,11 rather than 1..6. `Fast100`/`Slim100` are
> unchanged. The component floors (`validate()`) are profile-independent
> and still hold; the per-profile derivation is in
> `LigeritoSecurityConfig::derive_profile_ladder`.

Status: implementation audit and independent re-review of the current
`min/recursion-128bit` working tree. The non-Ligerito audit was completed on
2026-08-11; Ligerito two-point OOD binding, the Flock paper's Appendix C.3
algebraic grinding,
and the 128-bit list-decoding query schedule were reviewed on 2026-08-12. The
Fable 5 external audit findings and Ron's 100/128 recursion-variant report
were resolved and revalidated on 2026-08-13. Ron's split-commitment F256 MCA
design and the full component-security pass were implemented and reviewed on
2026-08-13. A final independent pass that day found and fixed strict-profile
composition, verifier-policy binding, proof-decoder robustness, one malformed
ring-switch panic, and conservative L0 OOD degree accounting; the complete
native and recursive test matrix was rerun afterward.

This is the authoritative reviewer-facing summary of the 128-bit milestone.
It records what was implemented, why each bit count is sufficient, how the
prover/native verifier/recursive R1CS agree, and how to reproduce the test
evidence. It is intentionally self-contained so a reviewer does not need
branch-local planning or implementation-history documents.

The review target is **per-part 128-bit computational security** for the
algebraic challenge sites and the completed Ligerito list-decoding, OOD,
query, and mutually correlated agreement (MCA) components. As requested, the
following are not review blockers and are not used to qualify the conclusion:

- an eventual global soundness ledger, union bound, or centralized parameter
  handler; and
- inactive legacy APIs.

## Executive conclusion

**Go/no-go result: the strict Fast/Slim component-security pass is complete.**
The final pass found that Fast/Slim selected strict Ligerito parameters but
still disabled the non-Ligerito grinding families; this profile-composition
bug is fixed. No missing or under-protected component remains in the active
production or recursion paths. The Flock paper's Appendix C.3 claim batching
and queried consistency retain their strict base-field grinding; quadratic
fold/sumcheck and MCA arithmetic now run over F256; two-point OOD binding and
the Johnson query term also clear 128 bits independently.

This conclusion is profile-specific. `Fast` and `Slim` are the strict
128-bit Johnson profiles. `Fast100` and `Slim100` intentionally preserve the
old 100-bit query cost point and disable non-Ligerito algebraic grinding,
while `Secure` enables those grinding families but remains the historical
120-bit unique-decoding profile. Those compatibility profiles are useful
regression targets but must not be advertised as 128-bit configurations.

All active non-Ligerito algebraic challenge families found in the current
production prover/verifier paths have grinding under strict `Fast`/`Slim`
(and historical `Secure`):

- Boolean and element zerocheck/lincheck;
- ring switching, opening batching, merged sumcheck, multipoint and anchor;
- the Product-GKR wiring/permutation argument;
- dense, element, sigma and jagged accumulator folds; and
- the public chain and Merkle-path wrapper arguments.

The prover and native verifier use identical schedules, reject malformed nonce
shapes before challenge replay, and bind each nonce immediately before the
protected randomness. The active recursive Flock-proof circuit checks every
recorded child-proof and accumulator-fold `Pow` with native BLAKE3 arithmetic,
64-bit nonce constraints, and the requested leading-zero predicate. The check
word is circuit IO bound to its fixed value; it is not a private witness cell.

The current permutation/copy-constraint argument is the active batched
Product-GKR path. It is included in this conclusion. The older standalone
`permutation.rs` API is inactive and outside the requested review boundary.

## How to review this milestone

A reviewer can follow the work in this order:

1. Check the work-factor rule and exact native PoW relation below.
2. Check the degree and bit-count table in "Implemented schedules."
3. Follow the code map in "Production coverage matrix."
4. Compare the transcript order in "Prover / verifier agreement."
5. Inspect the recursive relation and generic tape walker.
6. Run the commands in "Test evidence."

No trust in the prover's claimed nonce validity or in an honest witness
generator is part of the recursive soundness argument: the R1CS relation and
its circuit wiring recompute and constrain every recorded nonzero PoW.

## External-audit resolution

The Fable 5 audit found one high-severity recursive soundness gap and three
secondary API/shape issues. All are resolved:

- `PowMaskTable` previously used `C = I` to define a private check cell as
  `pred_j * mask_j`. A malicious witness could write the product there. The
  whole check word is now an input in the table's IO schema, and every circuit
  invocation wires it to `0^127 || 1`; the last bit is the table's pinned-one
  column. Thus the first 127 equations are
  `pred_j * mask_j = 0`, including every supported `lambda <= 64` prefix bit.
- The same latent issue in `BitSpreadTable::zero_mask` is fixed by making its
  check word a zero-wired circuit input.
- Dense and succinct Ligerito verification now require exact query-grinding
  nonce-vector length, so a trailing unused nonce is rejected.
- Standalone `element_r1cs::{prove,verify}_with_grinding` now applies Secure
  PCS-opening grinding as well as PIOP grinding.
- Recursion's matrix-fold policy now comes from
  `PcsParams::matrix_fold_grinding()`, removing the duplicated policy switch.

The regression tests deliberately construct the former `C = I` repair and
show that it satisfies the raw table R1CS, then check that the repaired word
differs from the circuit input required by production wiring. A focused real
recursive proof accepts a valid nonce and rejects an invalid one.

The final independent pass resolved five additional robustness findings:

- strict `Fast` and `Slim` now enable every non-Ligerito grinding policy,
  including the public chain and Merkle-path wrappers; `Fast100` and
  `Slim100` retain the compatibility transcript without those nonces;
- mixed/union verification re-derives commitment parameters from the
  caller-selected profile and compares every security-relevant field, so a
  proof cannot select a weaker profile or Merkle hash through its commitment;
- all proof readers use a 64 MiB size ceiling, bounded bincode decoding, and
  trailing-byte rejection;
- a malformed ring-switch vector now returns a typed verification error
  instead of panicking; and
- L0's implicit OOD opening is conservatively charged degree `m = mu + 7`,
  rather than degree `mu`. Canonically regenerated diagnostics changed by
  less than one bit and the weakest OOD component remains 233.7 bits.

## Soundness rule

Let `q = 2^128`. For a nonzero degree-`D` polynomial evaluated at fresh field
randomness, Schwartz--Zippel gives

```text
Pr[false acceptance at this site] <= D / q.
```

Under the roadmap's work-normalized random-oracle model, a `lambda`-bit grind
changes the computational term to approximately

```text
D / 2^(128 + lambda).
```

The code centralizes the strict local rule as

```text
bits_for_degree(D) = floor(log2 D) + 1,   D >= 1,
bits_for_degree(0) = 0.
```

Thus a linear event uses one bit, a quadratic event two, degree 3 also two,
and degree 7 three. This is computational amplification, not an improvement
to the information-theoretic bound against an unbounded adversary.

Equivalently, finding a nonce that satisfies both the PoW predicate and a
degree-`D` bad-challenge condition takes expected work approximately

```text
2^(128 + lambda) / D.
```

The extra `+1` at powers of two is intentional because the target is a strict
inequality. For example,

```text
D = 1, lambda = 1:  1 / 2^129       = 2^-129
D = 2, lambda = 2:  2 / 2^130       = 2^-129
D = 3, lambda = 2:  3 / 2^130       < 2^-128
D = 7, lambda = 3:  7 / 2^131       < 2^-128
D = 256, lambda = 9: 256 / 2^137    = 2^-129.
```

Implementation: `grinding_bits_for_degree` in
[`challenger.rs`](../crates/flock-transcript/src/challenger.rs).

## Exact PoW relation

For chained transcript state `(cv, pending)`, the prover finds a 64-bit nonce
`w` such that the domain-separated fused compression

```text
O = Compress(cv, pending || w_le || 0^64 || padding,
             pow_squeeze_counter(lambda, message_len), 64, CHAIN_SQUEEZE)
prefix_lambda(O_1) = 0^lambda.
```

Here `O_i` are 128-bit output words. The protected scalar challenge is `O_0`,
the predicate is the disjoint word `O_1`, and the next chaining value is the
high half `O_2 || O_3`. At zero bits the canonical nonce is zero.

Under the random-function/XOF assumption for disjoint words of the custom
BLAKE3 compression, searching for `w` is the work amplifier without biasing
the protected field challenge. The nonce and the entire preceding transcript
are inputs to the same operation.

The recursive relation in `circuit_merkle.rs` wires this compression as the
ordinary challenge-squeeze row, constrains the nonce to 64 bits, and
constrains the selected predicate bits to zero. A scalar `Pow` therefore adds
no BLAKE row beyond the squeeze already needed for its challenge. It adds one
four-word `PowMaskTable` row. The row's check word is fixed by circuit wiring,
which is load-bearing under the Boolean circuit convention `C = I`.

Native and recursive implementations agree on byte order: `w` is
little-endian, BLAKE3 output bytes use their native serialization, and
"leading bits" means most-significant-bit first within each serialized byte.
The differential test covers masks from 1 through 128 bits, including byte and
128-bit boundaries. See `grinding-hash-fusion-design.md` for the full
transition and code map.

## Implemented schedules

All bit counts below are sufficient for the strict local rule.

| Family / challenge | degree bound | strict-profile bits |
| --- | ---: | ---: |
| Boolean zerocheck initial equality point | `m` | `bits_for_degree(m)` |
| Boolean zerocheck skip point | `2^(K_SKIP+1)-1` | `K_SKIP+1` |
| Boolean zerocheck ordinary round | 2 | 2 |
| Boolean lincheck batching/pins | 1 | 1 |
| Boolean lincheck ordinary round | 2 | 2 |
| Boolean lincheck final skip | `2^k-1` | `k` |
| AG-skip zerocheck `r_1` (fused nonce) | 474 | 9 explicit (no sampling credit at `r_1`) |
| AG-basis lincheck final skip (fused nonce) | 158 | 3 explicit + 5 sampling credit |
| Element zerocheck initial equality point | `m_words` | `bits_for_degree(m_words)` |
| Element zerocheck/lincheck ordinary round | 2 | 2 |
| Element lincheck batching | 1 | 1 |
| Ring-switch point in `F128^7` | at most 7 | 3 |
| Whole mixed opening coefficient vector | total degree 1 | one `Pow(1)` |
| Dense merged-sumcheck round | 2 | 2 |
| Multipoint coefficient `gamma` | `K-1` | `bits_for_degree(K-1)` |
| Multipoint / Frobenius-anchor round | 2 | 2 |
| Product-GKR fingerprint `(alpha,beta)` | live entries `L-1` | `bits_for_degree(L-1)` |
| Product-GKR layer batching / close | 1 | 1 |
| Product-GKR product-sumcheck round | 2 | 2 |
| Fold coefficient vector `lambda` or `mu` | total degree 1 | one `Pow(1)` |
| Fold column or row sumcheck round | 2 | 2 |
| Chain packed-position vector, dimension `d` | at most `d` | `bits_for_degree(d)` |
| Chain initial `(tau,alpha)` | at most `max(n,1)` | `bits_for_degree(max(n,1))` |
| Chain shift round | 2 | 2 |
| Merkle packed-position vector, dimension `d` | at most `d` | `bits_for_degree(d)` |
| Merkle initial `(tau,alpha)` | at most `max(n,path_log+1)` | corresponding rule |
| Merkle shift round | effective verifier degree 3 | 2 |
| Ligerito scalar claim batching | list union `L_max` | `floor(log2 L_max)+1` |
| Ligerito quadratic fold/sumcheck round | `2 L_max / 2^256` | 0; the weakest shipped raw bound is 246.4 bits |
| Ligerito queried-consistency batching | `L_max ceil(log2 Q)` | `floor(log2(L_max ceil(log2 Q)))+1` |

For Product-GKR the fingerprint degree is `L-1`, not `L`: the top homogeneous
term cancels because the live identity tags are permuted. For the Merkle shift,
the verifier accepts cubic interpolation even where the honest message may be
quadratic, so the malicious-proof degree is conservatively 3.

### Ligerito Part 3: list-decoding query schedule

At a Johnson-regime level with code rate `rho = 2^(-r)` and fixed slack
`eta = 0.02`, the proximity radius and consistency-query error are

```text
gamma = 1 - sqrt(rho) - eta,
epsilon_query <= (1 - gamma)^Q.
```

Writing

```text
b_per_query = log2(1 / (1 - gamma)),
lambda_query = the existing query-phase PoW bits,
```

the work-normalized security contribution is

```text
b_query = Q * b_per_query + lambda_query.
```

Part 3 chooses the smallest integer query count satisfying the **strict**
local target

```text
Q_min = floor((128 - lambda_query) / b_per_query) + 1,
Q_min * b_per_query + lambda_query > 128.
```

Fast uses `lambda_query = 0`, so its raw query error is strictly below
`2^-128`. Slim retains its existing 16-bit query-phase PoW and therefore
requires the raw query term to exceed 112 bits; the combined work-normalized
term is strictly above 128 bits. The PoW is verified by both the native
verifier and the recursive R1CS relation. Secure is a unique-decoding profile,
so its older 120-bit policy is not changed by this list-decoding milestone.

The canonical Johnson counts by inverse-rate exponent are:

| `r` in `rho = 2^-r` | Fast `Q` | Slim `Q` | Slim delivered bits |
| ---: | ---: | ---: | ---: |
| 1 | 279 | -- | -- |
| 2 | 136 | 119 | 128.267 |
| 3 | 91 | 79 | 128.228 |
| 4 | 68 | 60 | 129.338 |
| 5 | 55 | 48 | 128.578 |
| 6 | 46 | 41 | 130.221 |

Only the rates used by a profile's recursion ladder appear in its TOML. For
the representative `m27_fast` ladder this changes the per-level schedule from
`[218, 106, 71, 53]` (448 total) to `[279, 136, 91, 68]` (574 total). The
generator and validator recompute the bound from exact floating-point
formulas; the rounded `expected_eps_query_bits` fields are diagnostics, not
trusted security inputs. A boundary test proves that every generated Johnson
level clears 128 bits and that removing one query makes it fail or meet, but
not strictly clear, the target. Because queried-consistency batching has
degree `ceil(log2 Q)`, eight generated levels cross a power-of-two boundary;
their consistency-batching PoW increases by one bit as a derived consequence.

### F256 MCA and split commitments

The remaining proximity/MCA challenges are sampled in

```text
F256 = F128[u] / (u^2 + u + x^-1),
```

where `x` is the GHASH polynomial-basis generator. The absolute trace of
`x^-1` is one, so the Artin--Schreier polynomial is irreducible. Multiplication
uses three F128 products:

```text
p0 = a0 b0
p1 = a1 b1
p2 = (a0 + a1)(b0 + b1)
(a0 + a1 u)(b0 + b1 u)
  = (p0 + x^-1 p1) + (p2 + p0) u.
```

Commitments and NTTs remain over F128. At a code switch, an extension table
`f(x) = f0(x) + u f1(x)` becomes the base table

```text
g(b, x) = f_b(x),    b in {0,1},
```

with `b` as the least-significant variable. A linear basis `B(x)` is
transported as `(B(x), u B(x))`; folding the coordinate bit at challenge `r`
contributes the factor

```text
phi(r) = 1 + r(1 + u).
```

Thus every recursive level has four folds: one coordinate fold and three
original-variable folds. It removes three original variables, exactly as the
old three-round ladder did, while committing only ordinary F128 words. The
final extension residual is likewise exposed as adjacent `(c0,c1)` words.
OOD points and answers, query batching `alpha`, and claim/glue batching `beta`
remain in F128 and retain their existing protections.

The soundness use of extension challenges is compatible with base-field
commitments. If a base word `v in F_q^n` is close enough to an
`RS(F_{q^2})` codeword `c`, then the Frobenius conjugate `c^q` agrees with `v`
on the same positions. Inside the Johnson radius that agreement exceeds the
code rate, so uniqueness on those positions gives `c = c^q`; hence `c` is a
base-field codeword. This subfield-descent argument applies at every split
commit boundary.

For the Flock paper's Appendix C bounds, replacing `|F| = 2^128` by
`2^256` gives

```text
epsilon_MCA      <= a / 2^256
epsilon_sumcheck <= 2 L_max / 2^256.
```

All 70 embedded TOMLs are exact canonical generator output and validate with
the following weakest rounded values. Validation uses the unrounded formulas
and strict inequalities.

| component | weakest delivered value |
| --- | ---: |
| F256 MCA/proximity | 205.8 bits |
| F256 list-unioned quadratic sumcheck | 246.4 bits |
| two-point F128 OOD binding | 233.7 bits |
| F128 claim batching after grinding | 128.4 bits |
| F128 queried-consistency batching after grinding | 128.0 rounded; exact value is strictly above 128 |
| Fast/Slim Johnson queries after optional grinding | 128.2 bits |

Every generated `fold_grinding_bits` entry is consequently zero. The config
validator independently enforces a strict 128-bit floor for the F256 MCA,
OOD, Appendix C.3 batching, and strict Johnson query components rather than
trusting the profile's rounded diagnostic fields.

For deeper Johnson levels, the two explicit OOD points each evaluate a
degree-`mu` packed polynomial. At L0, one point is explicit degree `mu` and
the ordinary ring-switched opening supplies the other point with degree at
most `m = mu + 7`. Therefore the pair-collision factors used by validation
are respectively

```text
(mu / 2^128)^2                    (deeper levels)
(mu / 2^128) * (m / 2^128)       (L0).
```

Both are then union-bounded over unordered pairs in the Johnson list. This is
the conservative accounting implemented by `paper_ood_bits`; all 56 Johnson
configs (`Fast`, `Fast100`, `Slim`, and `Slim100`) validate against it.

Implementation map:

- `field/gf2_256.rs`: field representation, nonresidue and Karatsuba;
- `challenger.rs`: canonical two-word observe and one double-width squeeze;
- `pcs/ligerito/extension.rs`: active split-commit prover and succinct
  verifier, including the residual basis transport;
- `pcs/ring_switch.rs` and `pcs/tensor_algebra.rs`: F256 evaluation of the
  succinct ring-switch basis;
- `pcs.rs`: active mixed-opening verifier dispatch; and
- `circuit_merkle.rs`: paired-limb transcript replay and the
  `LeafEvalGate256`, `SpineGate256`, `ResidualGate256`, `PrefixGate256`, and
  `MacGate256` relations. Each general F256 product constrains the three
  products and both output limbs shown above.

Proof-container version 19 records F256 sumcheck messages and rejects v18
bytes. Recursive commitments, OOD answers, opened rows, and the final split
residual remain serialized as F128 words.

### Degree justifications by error family

The table uses the following concrete bad-event polynomials.

**Random equality-point reduction.** Converting a false table identity into a
scalar claim produces

```text
E(r) = sum_x eq(r, x) * f(x).
```

For `d` sampled coordinates, `eq(r,x)` is multilinear and `E` has total degree
at most `d`. This gives the Boolean `m` and element `m_words` initial bounds.

**Ordinary sumcheck.** Once the prover has observed/sent the current round
message, false acceptance at the new challenge is the zero set of the
difference between the claimed and required univariate identities. The
Boolean, element, merged-opening, multipoint, anchor and fold round
polynomials are products of at most two multilinear factors, hence degree at
most two. The chain shift has the same bound. The current Merkle wrapper
verifier accepts a degree-three interpolation, so its malicious-proof bound is
three.

**Optimized Boolean skip checks.** With `ell = 2^K_SKIP`, the zerocheck
combined skip polynomial has degree below `2 * ell`, hence degree at most
`2^(K_SKIP+1)-1`. The lincheck closing interpolation has degree below `ell`,
hence at most `2^k-1` for its actual skip width `k`.

**Linear batching.** Boolean/element lincheck batching, Product-GKR layer
batching/close, opening coefficient vectors and matrix-fold coefficient
vectors reduce to a nonzero polynomial such as

```text
E(alpha) = E_0 + alpha * E_1
```

or `sum_i alpha_i E_i`; its total degree is one.

**Ring switching.** After the claimed `s_hat_v` is bound and its public claim
relation checked, a false bridge to the packed witness leaves a nonzero
multilinear discrepancy evaluated at `r'' in F128^7`. Its total degree is at
most seven.

**Product-GKR fingerprint.** With `L` live entries the initial error is

```text
prod_x (f_x + alpha * id_x       + beta)
+
prod_x (g_x + alpha * id_sigma(x) + beta).
```

In characteristic two the two total-degree-`L` homogeneous terms are equal
and cancel because `sigma` permutes the live tags. The remaining degree is at
most `L-1`.

**Multipoint batching.** If the claimed value discrepancies are `e_j`, the
bad batching event is

```text
T(gamma) = sum_(j=0)^(K-1) gamma^j * e_j = 0,
```

which has degree at most `K-1` when the discrepancy vector is nonzero.

### Multipoint boundary

The multipoint relation uses

```text
T(gamma) = sum_(j=0)^(K-1) gamma^j e_j,
K = 128 n_RS + n_groups.
```

The previous fixed one-bit schedule was insufficient. The implementation now
derives the schedule from `K-1`: `K=256` needs 8 bits and `K=257` needs 9.
The common mixed route has two RS claims and at least one scalar group, hence
uses at least 9 bits.

### Why one PoW can protect a vector squeeze

Several optimized sites sample a whole vector after one PoW. This is sound
when the bad event is a nonzero multivariate polynomial of bounded **total**
degree in that vector; the number of coordinates is not itself the degree.

- Opening batching checks `sum_i gamma_i E_i = 0`, total degree one.
- Matrix folding checks `sum_i lambda_i E_i = 0` and later
  `sum_i mu_i E'_i = 0`, each total degree one.
- The Boolean/element initial equality-point conversions have total degree at
  most their stated dimension, so their dynamic bit count does depend on that
  dimension.

This is why opening and fold coefficient vectors use one `Pow(1)`, while the
initial equality points use `bits_for_degree(dimension)`.

### Challenge-dependent denominator audit

The Convention-A verifiers reconstruct a missing endpoint through equations
of the form

```text
g(0) = (running + t * g(1)) / (1 + t).
```

The exceptional event `t = 1` is itself a degree-one bad set. It does not
create an unprotected site:

- Boolean zerocheck's seven protocol-fixed inner coordinates are not one;
  every sampled outer coordinate comes from the already-grinded initial
  vector.
- Every element-zerocheck `tau_i` comes from its grinded initial vector.
- Product-GKR's `t` coordinates are prior protected round or layer-close
  challenges.

For each initial vector, the union of its sampled exceptional hyperplanes has
degree at most the dimension used by `bits_for_degree`. Thus that exceptional
family, treated as its own algebraic part, also has a strict sub-`2^-128`
work-normalized term.

### AG-skip challenge family (union-AG route)

The AG-skip zerocheck (`zerocheck/ag_skip.rs`) and the AG-basis lincheck
skip replace two φ₈-basis challenges with points on the genus-95 cover.
Both are covered by the same strict local rule, with two AG-specific
ingredients: code-degree numerators and a provable sampling credit.

**Numerators.** Code dimensions fix the degrees through Riemann–Roch
(`deg = dim + g - 1`, `g = 95`): the base code `L(D)` has `dim 64`, so
`deg D = 158`; the product code `L(2D)` has `dim 222`, so `deg 2D = 316`.

- `r_1` protects the round-1 message pair: a false message survives only if
  a nonzero product-code word (at most 316 zeros — the ab check) or a
  nonzero base-code word (at most 158 — the c check) vanishes at `r_1`:
  numerator `474`, so `bits_for_degree(474) = 9`.
- The lincheck's fresh AG skip point protects `z_partial` exactly as the φ₈
  final skip does, but the interpolant is a base-code word: numerator `158`,
  so `bits_for_degree(158) = 8` (the φ₈ basis's `2^6 - 1 = 63 -> 6` row).

The challenge space is the set of valid cover points: Hasse–Weil bounds the
genus-95 cover's count within `2 * 95 * 2^64` of `2^128`, and the sampler's
exclusions (denominator poles — the x-degree-49 product denominator — and
points at infinity) remove a few hundred more, so the denominator is
`2^128 (1 - 2^-56)`; the slack is absorbed by the strict `+1` bit.

**Fused nonce and the sampling credit.** Both sites derive their point by
rejection sampling from a transcript seed, with a nonce in the proof
(`AgProof::r1_nonce`; the lincheck's skip slot in its nonce vector). Under a
strict schedule the nonce is FUSED: `H(seed || nonce)` must clear the
explicit PoW target AND decode to a valid point — both criteria on the same
hash, so every candidate an adversary evaluates re-enters the PoW and there
is no free choice among valid nonces. The PoW predicate is the transcript
convention: leading zero bits, MSB-first within each serialized byte, of the
hash's second 16-byte word — the same predicate word and bit order the
recursion circuit's `PowMaskTable` checks, so the in-circuit form is a
gadget reuse. The verifier is one attempt (constant-shape, no rejection
replay).

The rejection sampling itself contributes exactly `log2(32) = 5` bits, by
two facts, neither empirical:

1. *Uniformity*: the sampler weights every reachable cover point at exactly
   `1 / (2^128 * 4 * 8)` — slot flattening over the degree-4 base fiber,
   and the z-fiber is all-or-nothing (three Artin–Schreier levels, eight
   lifts selected by three uniform choice bits; any failure rejects the
   whole x). So a valid draw costs at least `~32` attempts.
2. *Hasse–Weil* pins the acceptance probability to `(1/32)(1 ± 2^-56)`.

Work-normalized, a candidate costs at least one hash and succeeds with
probability `p * 2^-b * numerator / N <= 2^-(b+5) * numerator / 2^128 * (1 + 2^-55)`,
so `b_explicit = bits_for_degree(numerator) - 5` where the sampling credit
applies. The credit applies at the LINCHECK skip only (**3 explicit
bits**): its decode stays canonical end-to-end. At `r_1` the recursion
circuit binds the decode with RELAXED canonicity (`emit_ag_point_binding`
accepts any of the <= 32 fiber points over the XOF-derived `x`), which
returns the sampler's 5 flattening bits to a circuit-side prover — so
`r_1` carries **all `bits_for_degree(474) = 9` bits explicitly** and the
total budget is unchanged. The constants are tied together by
`ag_skip::credit_constants_are_pinned` (code degrees, the `4 * 8 = 32`
sampler shape, and both explicit-bit splits) and the statistical pin
`genus95_curve_code::tests::acceptance_rate_is_one_in_32`; a sampler
reshape breaks the guard tests before it can silently void the credit.

The AG protocol's seven friendly inner coordinates are protocol-fixed
constants (the γ-geometric weights), not challenges — the same treatment as
the boolean zerocheck's seven fixed inner coordinates in the
challenge-dependent denominator audit above. The ungrinded direct route
(`prove_fast_ag`) keeps the plain single-attempt nonce and makes no 128-bit
claim (its nonce freedom is at most `log2(20000) = 14.3` bits).

Remaining AG-specific obligations before a strict profile may select this
family: an external check of the curve data underlying both uses of
Hasse–Weil (genus, absolute irreducibility — the `AG_codes`
`F2_human_audit/` artifacts, which the code-degree soundness already
assumes), and the recursive-circuit treatment of the fused check (one hash
plus one point evaluation; constant-shape by construction).

## Production coverage matrix

This table maps each implemented family to its policy, active native entry
points, proof witness, and recursive handling. Strict `Fast`/`Slim` and
historical `Secure` select the PIOP, Product-GKR and PCS policies through
`PcsParams`; the recursion tower selects the matching fold policy through
`tower_fold_grinding()`.

| Family | Policy / proof data | Active prover and verifier | Recursive R1CS |
| --- | --- | --- | --- |
| Boolean zerocheck | `ZerocheckGrinding`; `ZerocheckProof.grinding_nonces` | `zerocheck.rs`; normal and union calls in `prover.rs` / `verifier.rs` | generic child-tape `Pow` checks |
| Boolean lincheck | `LincheckGrinding`; `LincheckProof.grinding_nonces` | `lincheck.rs`, `lincheck/union.rs` | generic child-tape `Pow` checks |
| Element zerocheck/lincheck | `element_r1cs::Grinding`; both subproof nonce vectors | `element_r1cs/{zerocheck,lincheck,union}.rs` | generic child-tape `Pow` checks plus element verifier arithmetic |
| Product-GKR permutation | `BatchedGrinding`; `ProductGkrBatchedProof.grinding_nonces` | `circuit::prove_wiring_with_grinding`; matching ordinary/deferred verification | generic child-tape `Pow` checks plus GKR arithmetic |
| Ring switch / opening batching / merged rounds | `OpeningGrinding`; ring-switch nonce and exact nonce vectors | `pcs.rs`, `pcs/ring_switch.rs` | generic child-tape `Pow` checks plus opening arithmetic |
| Multipoint / anchor | `MultipointGrinding`; gamma, round and anchor nonces | active twisted multipoint functions in `pcs/jagged.rs` | generic child-tape `Pow` checks plus multipoint/anchor arithmetic |
| Dense, element and sigma folds | `FoldGrinding`; `FoldProof.grinding_nonces` | `matrix_fold.rs`, `aggregate.rs` | generic fold-tape `Pow` checks plus fold arithmetic |
| Jagged folds | same `FoldGrinding` and proof type | jagged fold entry points in `matrix_fold.rs`, `aggregate.rs` | same generic fold-tape handling |
| Ligerito Appendix C.3 | per-level claim/consistency schedules; fold nonces are empty under F256 | `pcs/ligerito/extension.rs`, active succinct verifier | generic child-tape `Pow` checks plus paired-limb Ligerito arithmetic |
| Ligerito consistency queries | per-level `queries` and query-phase `grinding_bits` | `pcs/ligerito.rs`, dense and succinct verifiers | query sampling, openings, and query-phase `Pow` replayed from the child tape |
| Ligerito MCA/proximity | F256 fold challenges, split F128 commitments | `pcs/ligerito/extension.rs` | F256 leaf, spine, residual, prefix, and MAC gates |

Key reviewer entry points:

- policy selection: [`pcs/commit.rs`](../crates/flock-core/src/pcs/commit.rs);
- native PoW and degree helper:
  [`challenger.rs`](../crates/flock-transcript/src/challenger.rs);
- Boolean PIOPs: [`zerocheck.rs`](../crates/flock-core/src/zerocheck.rs),
  [`lincheck.rs`](../crates/flock-core/src/lincheck.rs), and
  [`lincheck/union.rs`](../crates/flock-core/src/lincheck/union.rs);
- element PIOP: [`element_r1cs.rs`](../crates/flock-core/src/element_r1cs.rs)
  and [`element_r1cs/`](../crates/flock-core/src/element_r1cs);
- permutation/copy constraints:
  [`product_gkr.rs`](../crates/flock-core/src/product_gkr.rs) and
  [`circuit.rs`](../crates/flock-core/src/circuit.rs);
- PCS transport: [`pcs.rs`](../crates/flock-core/src/pcs.rs),
  [`pcs/ring_switch.rs`](../crates/flock-core/src/pcs/ring_switch.rs), and
  [`pcs/jagged.rs`](../crates/flock-core/src/pcs/jagged.rs);
- accumulation: [`matrix_fold.rs`](../crates/flock-core/src/matrix_fold.rs)
  and [`aggregate.rs`](../crates/flock-core/src/aggregate.rs);
- F256 Ligerito: [`extension.rs`](../crates/flock-core/src/pcs/ligerito/extension.rs),
  [`gf2_256.rs`](../crates/flock-field/src/gf2_256.rs), and
  [`tensor_algebra.rs`](../crates/flock-core/src/pcs/tensor_algebra.rs);
- production plumbing: [`prover.rs`](../crates/flock-prover/src/prover.rs)
  and [`verifier.rs`](../crates/flock-core/src/verifier.rs); and
- recursive recording and R1CS replay:
  [`transcript_record.rs`](../crates/flock-transcript/src/transcript_record.rs) and
  [`tower/mod.rs`](../crates/flock-prover/src/tower/mod.rs).

The native chain and Merkle-path wrappers also carry grinding for their own
packed-position and shift arguments. Their native implementations are recorded
in the schedule table for completeness. Recursive verification of those
wrapper arguments is explicitly outside this milestone's review scope.

### Active-challenge inventory result

The re-review searched every active `sample_f128` / `sample_f128_vec` in the
production verifier families above and followed its caller into the normal,
mixed/union, deferred, and recursive routes. Every algebraic squeeze is either
immediately preceded by its policy's PoW or is a previously protected
challenge reused by a later check.

Raw samples used only to create transcript fork seeds or bind transcript state
do not independently test a polynomial identity; the algebraic challenges
inside the resulting child transcript remain individually protected.
Ligerito OOD, Appendix C.3 algebraic randomness, the list-decoding query term,
and F256 MCA/proximity arithmetic have also been audited.

## Transcript and proof-shape improvements

### Opening coefficients

The old merged opening used a separate `Pow(1)` and scalar squeeze for every
claim. A representative child had 199 such claims. The new order is

```text
run all ring switches;
observe all packed-direct values;
Pow(1);
sample_f128_vec(n_RS + n_PD).
```

The discrepancy is total degree one in the whole coefficient vector, so one
PoW is sufficient. Ring-switch outputs are scaled after the shared vector is
sampled. This removes 198 PoW witnesses and 396 serialized finalizations per
representative child (198 PoWs plus 198 scalar-to-vector squeeze savings).

### Matrix-fold coefficients

Dense, element, sigma and jagged folds similarly use one vector squeeze for
all `lambda` coefficients and one for all `mu` coefficients. The recursive
reader maps every challenge ordinal to `(finalization, output-word offset)`;
it no longer assumes one challenge per finalization.

### Proof format

The incompatible additions and transcript changes placed the aggregate
proof format at v20 as of this audit (head is at v21 — 2026-08-14, the
merged-union R1cs payload). Product-GKR, matrix fold, chain and Merkle shift proofs carry
transcript-ordered nonce vectors; chain/Merkle wrappers carry their
packed-position nonce; opening batching has one nonce and one vector squeeze.
Version 16 added two-point Ligerito OOD data; version 17 adds the Flock
paper's Appendix C.3 claim- and consistency-batching nonce vectors. Version
18 selects the larger
strict-128 Johnson query schedules; the Rust proof structure is unchanged,
but v17 caps, authentication paths, and transcript challenges cannot be
replayed under the new public schedule. Version 19 moves Ligerito fold
challenges, sumcheck messages, and running claims to F256 and changes every
recursive code switch to the split-coordinate representation.
Version 20 enables all non-Ligerito grinding families for strict Fast/Slim,
changing their transcript nonce shape without adding proof-structure fields.
Deterministic proof-byte fixtures were re-pinned only after two identical
generation runs.

## Prover / verifier / recursive-circuit agreement

For each enabled family the native verifier:

1. derives the public policy and commitment parameters from the
   caller-selected PCS profile;
2. checks the exact expected nonce count (or canonical optional scalar);
3. observes the same proof messages as the prover;
4. verifies the nonce before sampling protected randomness; and
5. replays the same arithmetic relation.

The proof-carried commitment profile is data, not verifier policy. Mixed and
union verification compare the commitment against the complete expected
parameter tuple (`m`, rate, batch size, profile, lane shape, and Merkle hash)
before accepting it. The CLI likewise requires or defaults an expected
profile and rejects disagreement. The recursive relation is constructed for
that fixed expected transcript/parameter shape; native recording rejects a
profile mismatch before circuit construction.

The load-bearing transcript order is always

```text
observe the prover message(s) that fix the bad-event polynomial;
verify/grind Pow(lambda);
sample the protected scalar or vector;
use that sample in the verifier equation.
```

The family-specific nonce orders are:

| Family | Nonce order |
| --- | --- |
| Boolean zerocheck | initial vector; univariate skip; every ordinary round |
| Boolean lincheck | `alpha`; pinned-table `beta`s in slot order; every ordinary round; final skip |
| Element PIOP | initial `tau`; every zerocheck round; lincheck `alpha`; every lincheck round |
| Product-GKR | fingerprint; for each layer: `lambda`, its rounds, close |
| Outer PCS transport | each ring switch; shared opening coefficient vector; merged rounds; multipoint gamma; multipoint rounds; anchor rounds |
| Matrix fold | column coefficient vector; column rounds; row coefficient vector; row rounds |
| Ligerito | each OOD `beta`; per-level query grind; per-level consistency `alpha`; per-level glue `beta` (F256 fold challenges need no PoW) |

Proofs with vector nonce fields are checked for their **exact** expected
length before replay. Optional scalar nonce fields use a canonical zero when
their site is disabled, so they cannot silently become an extra transcript
grinding knob.

For active recursive Flock proofs, `RecordingChallenger` records every PoW.
The recursive circuit uses the generic PoW relation for all child-tape sites,
including Product-GKR and PCS transport. Accumulator fold tapes are also
replayed in-circuit, including both coefficient-vector PoWs and every round
PoW. Hand parsers now tolerate PoW payload insertion and vector squeezes by
deriving payload/challenge locations from the op tape. The Ligerito opening
parser anchors at the protocol domain label rather than guessing from cap byte
lengths. An F256 transcript message is consumed as two ordered F128 words and
an F256 challenge as one `SqueezeSlice(2)`; both limbs feed constrained
Karatsuba arithmetic. The residual inner product is copy-constrained to both
limbs of the sumcheck's final running claim.

The strict Fast tower test consumes a recursive node proof as a child, so this is
not only a one-level parse check.

The recursive PoW implementation has three complementary tests:

- a native/circuit differential test checks the exact BLAKE3 block,
  serialization, and prefix masks; and
- a focused R1CS proof accepts a valid nonce and rejects a neighboring invalid
  nonce; and
- adversarial table tests exhibit the formerly possible private-cell repair
  and pin the load-bearing check words as circuit inputs.

The profile-matrix node tests exercise the generic relation at real Boolean, element,
Product-GKR, PCS-transport and fold sites. The strict Fast tower additionally
proves composability by consuming a recursive node proof as a child.

## Historical pre-fusion BLAKE and performance census

This section preserves the measurement that motivated fusion. It is not the
current implementation; the current isolated results are recorded near the
end of this document and in `grinding-hash-fusion-design.md`.

Commands:

```text
B3_CENSUS=1 TOWER_PROFILE={fast,secure} cargo test --release \
  -p flock-prover --test circuit_merkle \
  mvp11_two_to_one_recursion_node -- --ignored --nocapture
```

Historical per-child BLAKE census:

| component | Fast | Secure | delta |
| --- | ---: | ---: | ---: |
| Fiat--Shamir transcript chain | 1,707 | 2,438 | +731 |
| `H(child_publics)` | 216 | 432 | +216 |
| Ligerito leaves, paths and caps | 9,266 | 15,668 | +6,402 |
| standalone nonzero-PoW checks | 15 | 445 | +430 |
| **total per child** | **11,204** | **18,983** | **+7,779** |

The two-child fold region contributes 2,229 rows in Fast and 3,249 in Secure.
That pre-fusion Secure node had 1,400 standalone PoW BLAKE rows in total:
`2 * 445` child checks plus 510 fold checks.

| node metric | Fast | Secure | change |
| --- | ---: | ---: | ---: |
| online proving (three-run median) | 232 ms | 350 ms | +50.9% |
| native verification | 10 ms | 13 ms | +3 ms |
| outer proof | 291.5 KiB | 589.2 KiB | +102.1% |
| BLAKE rows | 24,637 | 41,215 | +67.3% |
| BLAKE slot `nu` | 15 | 16 | one capacity bit |
| circuit cell `mu` | 23 | 24 | one capacity bit |

These are full profile comparisons, not an isolated grinding benchmark.
Secure changes the child Ligerito geometry from 448 to 748 total queries; the
12,804 extra Ligerito opening rows across two children remain the dominant
increase. The new algebraic families plus fold grinding are partly offset by
opening-vector consolidation: relative to the earlier Secure snapshot of
40,035 rows, the completed node is 41,215 rows, a net increase of 1,180.

One Secure child records 449 PoW operations: 445 nonzero and four canonical
zero-bit sites. The distribution is

```text
{0: 4, 1: 52, 2: 380, 3: 4, 4: 2, 5: 3, 6: 1, 7: 1, 9: 1, 18: 1}.
```

Product-GKR accounts for 276 of the nonzero child sites. This high count is
expected from its per-layer quadratic rounds; each costs one recursive BLAKE
row, while the one 18-bit fingerprint grind dominates native nonce-search
work for this measured child.

## Test evidence

The final review reran the following commands from the repository root:

```sh
cargo test --locked --workspace --release

cargo test --locked --release -p flock-prover --test verifier_roundtrip \
  strict_fast_profile_grinds_boolean_piops -- --ignored --nocapture
cargo test --locked --release -p flock-prover --test union_element \
  strict_fast_profile_grinds_element_piops -- --ignored --nocapture

cargo test --locked --release -p flock-prover \
  proof_io::tests -- --ignored --nocapture
cargo test --locked --release -p flock-prover \
  proof_bytes_pinned -- --ignored --nocapture

TOWER_PROFILE=fast cargo test --locked --release -p flock-prover \
  --test circuit_merkle mvp11_two_to_one_recursion_node \
  -- --ignored --exact --nocapture
TOWER_PROFILE=fast cargo test --locked --release -p flock-prover \
  --test circuit_merkle mvp12_recursion_tower \
  -- --ignored --exact --nocapture
TOWER_PROFILE=slim cargo test --locked --release -p flock-prover \
  --test circuit_merkle envelope_registry_diff \
  -- --ignored --exact --nocapture
TOWER_PROFILE=slim cargo test --locked --release -p flock-prover \
  --test circuit_merkle chain_tower_e2e_with_lane \
  -- --ignored --exact --nocapture
TOWER_ENV_M=29 TOWER_PROFILE=slim cargo test --locked --release \
  -p flock-prover --test circuit_merkle chain_spine_converges \
  -- --ignored --exact --nocapture
```

Results:

- `flock-core --lib`: 501 tests passed and 22 were ignored; every active
  integration suite also passed.
- `flock-prover`: 81 active library tests passed and 23 were ignored; every
  active integration suite also passed.
- Strict Fast Boolean and element/mixed production roundtrips passed.
- All three current v20 proof-bundle serialization/verification roundtrips
  passed (the earlier non-Ligerito checkpoint used v15).
- The production PCS roundtrip uses the embedded `m22_fast` F256 config and
  passes through `open_batch_mixed_ligerito` and the active mixed verifier.
- The Slim fixed-shape leaf/node and four-chain recursion tower pass. The
  tower consumes two first-level recursive proofs, proves the internal node,
  verifies it, and rejects its statement/tape tampering cases.
- Fast100 and Secure `mvp11`/`mvp12`, plus Slim100 and Secure chain towers,
  pass under their corresponding F256 transcript shapes.
- Fast100 `mvp11`/`mvp12`, Slim100 first-level/chain-tower/spine, strict-Slim
  `m=29` spine, and the Secure chain tower all passed on 2026-08-13.
- Ron's three ignored merged-transport byte-pin tests were run explicitly.
  The intentional v20 strict-profile transcript change moved all thirteen
  fixture digests; the replacements were identical across two deterministic
  generation runs and all three tests pass with normal pin checking.

Focused evidence in the full suites additionally covers:

- invalid and malformed nonces for Boolean, element, Product-GKR, ring switch,
  dense fold, jagged fold and multipoint/anchor proofs;
- the `K=256/257` multipoint schedule boundary;
- standalone sigma and jagged recursive fold-tape replay after vector
  squeezing; and
- native/circuit PoW agreement plus valid/invalid recursive nonce behavior.

The 2026-08-12 Ligerito Part 2 review additionally ran:

```sh
cargo test -p flock-core --lib
cargo test -p flock-prover
cargo test -p flock-prover proof_io::tests -- --ignored --nocapture
cargo test -p flock-prover --test circuit_merkle \
  mvp10_circuit_inner_tape -- --ignored --exact --nocapture
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

This was the pre-Part-3 checkpoint. All passed. The core run then had 483
active tests and 22 ignored. The focused
Ligerito mutation test checks both dense and succinct verifiers, invalid
claim/consistency nonces, and missing/extra nonce vectors. The config suite
rejects each under-sized Flock-paper Appendix C.3 schedule and checks all 42
embedded TOMLs against canonical generator output. The recursive tests prove that the
new `Pow` operations and protected arithmetic are replayed inside R1CS.

The 2026-08-12 Ligerito Part 3 review additionally ran:

```sh
cargo test -p flock-core --lib
cargo test -p flock-prover
cargo test -p flock-prover proof_io::tests -- --ignored --nocapture
cargo test -p flock-prover --test circuit_merkle \
  mvp10_circuit_inner_tape -- --ignored --exact --nocapture
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

This was the Part-3 checkpoint. All passed. The core run then had 485 active
tests and 22 ignored. The full prover
suite and all three v18 proof-container round trips passed. Focused config
tests check all 28 Fast/Slim Johnson configurations against canonical
derivation, check the strict one-query boundary, and reject a coherently
tampered under-target schedule. A real `m22_fast` native round trip and both
debug and release recursive verifier paths passed.

### Part 3 isolated performance

The release comparison holds the fused grinding implementation and two-point
OOD design fixed, changing only the Johnson query counts. Command:

```sh
cargo test --release -p flock-prover --test circuit_merkle \
  mvp10_leaf_outer_inner_tape -- --ignored --exact --nocapture
```

| representative `m27_fast` metric | Part 2 | Part 3 | change |
| --- | ---: | ---: | ---: |
| total consistency queries | 448 | 574 | +28.1% |
| child BLAKE rows | 8,207 | 10,656 | +29.8% |
| child proof | 323.5 KiB | 434.4 KiB | +34.3% |
| child prove median | 81 ms | 86 ms | +6.2% |
| child native verify | 6 ms | 7 ms | +1 ms |
| outer BLAKE rows | 11,183 | 14,840 | +32.7% |
| outer proof | 253.4 KiB | 313.8 KiB | +23.8% |
| outer prove | 135 ms | 147 ms | +8.9% |
| outer native verify | 9 ms | 10 ms | +1 ms |

The child proving ranges overlapped (`76--92 ms` before and `85--101 ms`
after), so the timing delta is indicative rather than a stable microbenchmark.
The outer circuit stayed at `nu = 14`, `mu = 22`; Part 3 did not cross a
capacity boundary.

## Final component-security conclusion

There is no remaining in-scope 128-bit component blocker in the active
`Fast`/`Slim` production and recursive-verification paths. Ligerito has
two-point OOD binding, strict Johnson query schedules, strict base-field
grinding for its Appendix C.3 batching challenges, and F256
MCA/quadratic-sumcheck arithmetic. The prover, active native verifier, and
recursive R1CS consume the same v20 transcript and reject malformed proof
shapes and the tested mutations.

This is deliberately not a claim about a global union-bound ledger, inactive
legacy APIs, or the explicitly named 100/120-bit compatibility profiles.

### Isolated F256 performance

The controlled comparison checks out the exact pre-F256 commit `8826508` in a
separate worktree, keeps `TOWER_PROFILE=slim` and all grinding/OOD/query work
fixed, and runs the same release tests on the same 32-thread host.

| Slim recursive metric | pre-F256 | split F256 | change |
| --- | ---: | ---: | ---: |
| leaf live BLAKE rows | 4,330 | 4,620 | +6.7% |
| leaf outer proof | 231.0 KiB | 281.7 KiB | +21.9% |
| leaf prove, two observed medians | 305/350 ms | 946/910 ms | about 2.8x |
| node live BLAKE rows | 20,424 | 24,483 | +19.9% |
| node total BLAKE rows | 23,580 | 27,651 | +17.3% |
| node constrained gate slots | 303 | 710 | +134.3% |
| node online time (`envelope_registry_diff`) | 282 ms | 592 ms | +110% |
| chain-tower internal online time | 336 ms | 674 ms | +101% |
| chain-tower internal verify | 12 ms | 14 ms | +16.7% |
| recursive outer proof | 231.0 KiB | 281.7 KiB | +21.9% |
| recursive public words | 5,300 | 5,684 | +7.2% |

The hash growth is not a new PoW cost: fold grinding disappears and leaf PoW
rows fall from 26 to 14. It comes mainly from hashing larger F256 transcript
and public segments. The dominant proving-time increase is extension
arithmetic: every general F256 multiplication becomes three constrained F128
products, the gate registry grows from 19 to 29 types, and the circuit cell
capacity moves from `mu=24` to `mu=25`.

Native serialized proofs grow less: the v20 R1CS bundle is 389.7 KiB versus
363.2 KiB (+7.3%), and the mixed bundle is 357.2 KiB versus 330.2 KiB (+8.2%).
Timings are host-local and the leaf samples are visibly noisy; sizes and row
counts are deterministic.

A fresh strict-Fast v20 smoke benchmark on the same 32-thread host gives
11,076 BLAKE rows and a 357.1 KiB proof per leaf. The two-to-one node has
33,295 BLAKE rows, a 453.0 KiB proof, 14 ms native verification, and a 621 ms
observed online time. The level-2 tower node has 39,605 BLAKE rows, a 475.0
KiB proof, and a 744 ms observed online time. Relative to the immediately
preceding v19 strict-Fast run, enabling the omitted non-Ligerito grinding adds
12 BLAKE rows at level 1 and 16 at level 2, while adding about 4.1 KiB to each
outer proof. Timings remain within the earlier run-to-run range.

Strict Slim v20 gives 5,755 leaf BLAKE rows, 27,651 node BLAKE rows, and a
285.8 KiB outer proof. Those BLAKE row totals are unchanged from v19 because
the fused PoW/squeeze relation reuses transcript finalizations; the recursive
PoW census nevertheless rises from 14/48 to 109/2,074 at the representative
leaf/node, demonstrating that the new checks are present. The observed node
online time was 573 ms versus 554 ms before, which is not a stable regression
outside the noise of these host-local samples. The strict Slim chain tower
and the `m=29` converging spine both passed with the full PoW census.

The exact benchmark commands are:

```sh
TOWER_PROFILE=slim cargo test --release -p flock-prover \
  --test circuit_merkle envelope_registry_diff -- --ignored --exact --nocapture
TOWER_PROFILE=slim cargo test --release -p flock-prover \
  --test circuit_merkle chain_tower_e2e_with_lane -- --ignored --exact --nocapture
cargo test --release -p flock-prover proof_io::tests -- --ignored --nocapture
```

### Earlier grinding-only measurements

The isolated benchmark was completed after rebasing onto the fresh
`recursion_circuit` branch. Holding the original 448-query Johnson/OOD
Ligerito geometry fixed, the 2-to-1 recursion node changed from 24,637 to
27,559 BLAKE rows (+11.9%), from 291.5 to 302.0 KiB (+3.6%), and from a
three-run median of 232 to 257 ms online proving (+10.8%). The corresponding
level-2 tower changes were 27,600 to 32,400 rows (+17.4%) and 299.0 to 316.2
KiB (+5.8%). Both isolated recursive tests passed. The temporary config
substitution was removed after measurement. The component census and
Secure/UDR comparison are recorded in the historical-performance section
above.

The grinding verifier was subsequently fused with its protected
Fiat--Shamir squeeze. Repeating the same isolated experiment reduced the
Secure-grinding node from 27,559 to 24,455 BLAKE rows and from 302.0 to 295.2
KiB. Against an optimized no-new-grinding control (24,451 rows, 291.5 KiB),
the remaining grinding-only overhead is four BLAKE rows (+0.016%) and 3.7 KiB
(+1.3%). A final same-shape rerun measured three-run online medians of 217
versus 208 ms (+4.3%) and prove-component medians of 203 versus 195 ms
(+4.1%). The four-leaf recursion tower passed as well. The exact fused
transition and its security argument are documented in
`grinding-hash-fusion-design.md`.

## Code-quality notes

Good foundations now include a central degree-to-bits helper, exact nonce
shape checks, canonical disabled fields, vectorized linear batching, one
generic recursive PoW relation, load-bearing IO-schema tests, and
native/circuit differential tests. The F256 implementation is isolated in a
small field module and a split-Ligerito module, with algebraic equivalence
tests for coordinate splitting, residual bases, and ring-switch tensors.
Verifier-facing proof IO is fail-closed: exact wire version and flavor,
bounded input and allocation size, no trailing bytes, complete expected PCS
parameter matching, and typed errors for malformed opening vectors.

The largest maintenance risk remains the hand-written transcript parser in
`circuit_merkle.rs`; deriving challenge and payload maps from the op tape and
anchoring at protocol labels have removed several fixed-offset bugs, but a
generated verifier transcript IR would be safer long term. The older private
base-field Ligerito implementation also remains in `ligerito.rs` as inactive
reference code and produces dead-code warnings after production dispatch
moved to `ligerito/extension.rs`. It is outside the active security boundary,
but deleting or feature-gating it would make the implementation easier to
maintain.

The recursion test's `tower_fold_grinding()` now delegates to
`PcsParams::matrix_fold_grinding()`, so the production policy and recursive
fold-tape policy have one source of truth.

## Reviewer checklist

A reviewer should be able to answer yes to all of the following after
following the code map and tests:

- Is every false polynomial fixed in the transcript before its PoW and
  challenge?
- Does its stated total-degree bound imply the configured bit count?
- Do prover and verifier derive dynamic counts from the same public shape?
- Does the verifier reject missing, extra, invalid, and noncanonical nonces?
- Does the recursive tape expose the exact pre-PoW transcript digest and nonce
  word to `emit_pow_checks`?
- Are the subsequent arithmetic challenge wires outputs of the post-nonce
  transcript state?
- Are vector-squeeze word offsets derived rather than assumed?
- Do the strict-Fast production node and strict-Slim tower tests pass, along
  with the Secure compatibility regression?
- Does every Johnson query level satisfy
  `Q * log2(1/(1-gamma)) + lambda_query > 128` exactly?
- Are every F256 challenge and message represented by the same ordered limb
  pair in the prover, native verifier, and recursive circuit?
- Do the exact MCA, OOD, and Flock-paper Appendix C.3 validators each require a strict
  delivered value above 128 bits?

The final 2026-08-13 review answered yes to every in-scope item.
