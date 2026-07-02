"""
sumcheck.py - a small, generic sumcheck protocol (Lund-Fortnow-Karloff-Nisan).

Goal: prove   claim = sum_{b in {0,1}^m}  P( t_1[b], t_2[b], ..., t_k[b] )
where each t_i is a multilinear table and P is a fixed low-degree combiner
(a product / difference of its inputs). Both zerocheck and lincheck in Flock are
instances of this with different tables and combiners.

Each round binds one boolean variable, sending a univariate polynomial of degree
`degree` (= the total degree of P in a single variable). We transmit that
polynomial as its evaluations at x = 0, 1, ..., degree and interpolate.

Design: the prover and verifier share the SAME `combine` closure. `combine` maps
a list of (partially bound) tables to an elementwise GF array. The prover uses it
to build each round polynomial; the verifier uses it only once, at the very end,
to check the final point - given oracle evaluations supplied by the caller.
"""

import numpy as np
from field import GF, ONE, ZERO
from mle import bind_first, num_vars


def _lagrange_interpolate_eval(xs, ys, at):
    """Evaluate, at point `at`, the polynomial through points (xs[i], ys[i]) over F.
    (Same Lagrange idea as the RareSkills homework, specialized to small degree.)"""
    # NOTE: use fresh GF(0)/GF(1) and non-in-place ops. `x = ONE; x *= y` would
    # mutate the shared global ONE in place (galois scalars are 0-d arrays).
    total = GF(0)
    k = len(xs)
    for j in range(k):
        num = GF(1)
        den = GF(1)
        for i in range(k):
            if i == j:
                continue
            num = num * (at - xs[i])
            den = den * (xs[j] - xs[i])
        total = total + ys[j] * num / den
    return total


def prove(tables, degree, combine, transcript):
    """
    Run the sumcheck prover.

    Returns (round_evals, challenges, final_tables) where
      round_evals[t] = [g_t(0), ..., g_t(degree)]  (the message for round t)
      challenges[t]  = the verifier challenge folded in after round t
      final_tables   = each input table fully bound to `challenges` (length 1)
    """
    m = num_vars(tables[0])
    xs = GF(list(range(degree + 1)))
    cur = [GF(t) for t in tables]
    round_evals = []
    challenges = []

    for _ in range(m):
        # Build g(x) = sum over the remaining cube of P(tables bound with var0 := x),
        # sampled at x = 0..degree.
        evals = []
        for x in xs:
            bound = [bind_first(t, x) for t in cur]
            evals.append(GF(np.sum(combine(bound))))
        evals = GF(evals)
        round_evals.append(evals)

        transcript.absorb_fe_list(evals)
        ch = transcript.challenge()
        challenges.append(ch)

        cur = [bind_first(t, ch) for t in cur]

    return round_evals, GF(challenges), [t[0] for t in cur]


def verify(round_evals, degree, claim, transcript):
    """
    Verify the sumcheck rounds against `claim`. Does NOT check the final point
    (the caller does that using externally obtained oracle evaluations, e.g. from
    the PCS). Returns (challenges, final_expected) where final_expected is the
    value P(...) must equal at the random point r = challenges.
    """
    xs = GF(list(range(degree + 1)))
    challenges = []
    cur_claim = claim

    for evals in round_evals:
        evals = GF(evals)
        # fundamental sumcheck check: g(0) + g(1) == running claim
        if GF(evals[0]) + GF(evals[1]) != cur_claim:
            raise ValueError("sumcheck: round consistency g(0)+g(1) != claim failed")
        transcript.absorb_fe_list(evals)
        ch = transcript.challenge()
        challenges.append(ch)
        cur_claim = _lagrange_interpolate_eval(xs, evals, ch)

    return GF(challenges), cur_claim
