#!/usr/bin/env python3
"""Per-round fold-grinding table for the affine F_{2^128} Ligerito fold
(docs/affine-fold-f128.tex).

For every level of a Ligerito config and every fold round j = 1..k of that
level, evaluate the general form of Lemma 3.1 of Chiesa-Yogev-Weissenberg-
Zheng, "Reed-Solomon Proximity Gaps with Small Error via Affine Mixing":

    #bad (alpha, beta) <= (4 d^2 M^2 + 1) D^3 v^2 + ell * max{q, C(n,2)}

for any positive integers m, D, B with

    (3.4)  sum_{j<=M} (mA - D - dj)(D - j + 1) > n sum_{s<min(m,M+1)} (m - s)(D - s + 1),
           1 <= B <= A <= n,  (B - 1)^2 > n (d + 1),
           M = min{D, floor((mA - D - 1) / d)},  v = n / (A - B + 1),
           ell = L_{d+1}(B - 1)  (integer Johnson bound, MacWilliams-Sloane).

Round j of a level folds words of RS[D, d + j - 1] into RS[D, d + j] with
d = 2^log_msg_cols - 1; the agreement threshold is A = ceil((sqrt(rho) +
eta) n) with rho = 2^-log_inv_rate, as in the config's Johnson analysis.
By the subfield-challenge lemma of the design note the bound does not depend
on the interleaving factor, so one evaluation per (level, round) suffices.

Per round:  eps_round = eps_aff + 2/q   (grid claim transport, one candidate
after OOD), and the fold grinding is ceil(target + log2 eps_round) bits.
The script also reports the query-term cost of agreement loss s = 1 per
round: t * log2(1 + k / (a n)) bits with a = sqrt(rho) + eta.

The (m, D, B) search is exhaustive over m and over a grid of B; D is the
smallest value satisfying (3.4) for that m. It is a search for a valid
certificate, so any reported bound is a proven bound, but a finer search
could find a slightly smaller one.
"""

import argparse
import math
import re
import sys
from math import comb, isqrt


def parse_levels(path):
    """Read the [[levels]] tables of a Ligerito TOML (flat keys only)."""
    levels, cur = [], None
    for raw in open(path):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "[[levels]]":
            cur = {}
            levels.append(cur)
        elif line.startswith("["):
            cur = None
        elif cur is not None:
            m = re.fullmatch(r"(\w+)\s*=\s*(.+)", line)
            if m:
                key, val = m.groups()
                try:
                    cur[key] = float(val) if "." in val else int(val)
                except ValueError:
                    cur[key] = val.strip('"')
    return levels


def johnson_list(n, T, d0):
    """Integer Johnson bound L_{d0}(T): the largest L <= floor(J) satisfying
    n k (k - 1) + 2 k r <= L (L - 1) d0 with L T = n k + r, 0 <= r < n."""
    if T * T <= n * d0:
        return None
    J = (n * (T - d0)) // (T * T - n * d0)
    for L in range(J, 0, -1):
        k, r = divmod(L * T, n)
        if n * k * (k - 1) + 2 * k * r <= L * (L - 1) * d0:
            return L
    return 1


def lemma31_feasible(n, d, A, m, D):
    M = min(D, (m * A - D - 1) // d)
    if M < 1:
        return None
    unknowns = sum((m * A - D - d * j) * (D - j + 1) for j in range(M + 1))
    equations = n * sum((m - s) * (D - s + 1) for s in range(min(m - 1, M) + 1))
    return M if unknowns > equations else None


def best_bound(n, d, A, log_q, max_m):
    """Smallest Lemma 3.1 bad-pair count over (m, D, B); returns
    (log2 eps, ell, m, D, B) or None."""
    q = 1 << log_q
    b_min = isqrt(n * (d + 1)) + 2  # (B-1)^2 > n(d+1)
    if b_min > A:
        return None
    b_step = max(1, (A - b_min) // 200)
    best = None
    for m in range(1, max_m + 1):
        # Feasibility of (3.4) is not monotone in D in general: scan upward
        # and keep the first feasible D (smallest D^3 factor).
        D, M = None, None
        for cand in range(m, 20000):
            M = lemma31_feasible(n, d, A, m, cand)
            if M is not None:
                D = cand
                break
            if (m * A - cand - 1) // d < 1:
                break
        if D is None:
            continue
        for B in range(A, b_min - 1, -b_step):
            ell = johnson_list(n, B - 1, d + 1)
            if ell is None:
                break
            v2_num, v2_den = n * n, (A - B + 1) ** 2
            bad = ((4 * d * d * M * M + 1) * D ** 3 * v2_num) / v2_den \
                + ell * max(q, comb(n, 2))
            log_eps = math.log2(bad) - 2 * log_q
            if best is None or log_eps < best[0]:
                best = (log_eps, ell, m, D, B)
    return best


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Ligerito TOML, e.g. "
                   "crates/flock-core/configs/ligerito/m32_fast.toml")
    p.add_argument("--field-bits", type=int, default=128, help="log2 |F| (default 128)")
    p.add_argument("--target", type=float, default=128.0,
                   help="per-round security target in bits (default 128)")
    p.add_argument("--max-m", type=int, default=40,
                   help="largest interpolation multiplicity to try (default 40)")
    args = p.parse_args()

    levels = parse_levels(args.config)
    if not levels:
        sys.exit(f"no [[levels]] in {args.config}")
    q_bits = args.field_bits
    transport = 2.0 ** (1 - q_bits)

    print(f"{'lvl':>3} {'rnd':>3} {'rho':>8} {'log n':>5} {'d_j':>8} "
          f"{'ell':>4} {'m':>3} {'D':>5} {'log2 eps_aff':>12} "
          f"{'log2 eps_rnd':>12} {'grind':>5}")
    worst = 0
    for li, lv in enumerate(levels):
        c, lr = lv["log_msg_cols"], lv["log_inv_rate"]
        k, eta, t = lv["log_num_interleaved"], lv["eta"], lv["queries"]
        n, rho = 1 << (c + lr), 2.0 ** -lr
        A = math.ceil((math.sqrt(rho) + eta) * n)
        for j in range(1, k + 1):
            d = (1 << c) - 1 + j
            r = best_bound(n, d, A, q_bits, args.max_m)
            if r is None:
                print(f"{li:>3} {j:>3} {rho:>8.2e} {c + lr:>5} {d:>8}   no valid (m, D, B)")
                worst = None if worst is None else max(worst, math.inf)
                continue
            log_eps, ell, m, D, _ = r
            log_rnd = math.log2(2.0 ** log_eps + transport)
            grind = max(0, math.ceil(args.target + log_rnd - 1e-9))
            if worst is not None:
                worst = max(worst, grind)
            print(f"{li:>3} {j:>3} {rho:>8.2e} {c + lr:>5} {d:>8} {ell:>4} "
                  f"{m:>3} {D:>5} {log_eps:>12.2f} {log_rnd:>12.2f} {grind:>5}")
        a = math.sqrt(rho) + eta
        loss = t * math.log2(1 + k / (a * n))
        print(f"    level {li}: agreement-loss query cost {loss:.4f} bits "
              f"(t = {t}, k = {k})")
    print(f"max fold grinding: {worst if worst is not None else 'n/a'} bits")


if __name__ == "__main__":
    main()
