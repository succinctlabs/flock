"""
flock_perf.py - wall-clock comparison: BATCH proving vs K-times separate proving.

The paper's central pitch (section 1) is that proving a *batch* of K identical
circuits is far cheaper than treating them independently, and the verifier's per-
batch work stays near the cost of a single instance (block-diagonal collapse, 4.1).

This script measures, for growing K:

  BATCH     : one Flock proof of the whole K-instance R1CS (+ the hash-chain glue)
  SEPARATE  : K independent Flock proofs, one per single-instance R1CS (K=1 each)

and reports prove time, verify time, and the number of proofs the verifier must
check. (SEPARATE also *cannot* enforce the cross-instance chain - that binding is
a batch-only feature - which we note as a correctness gap, not just a speed one.)

Run:  python3 flock_perf.py     (needs `galois` and `numpy`; see README.md)
"""

import time
import numpy as np
import r1cs
import flock


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def bench(K, seed=0):
    ys = [int(x) for x in np.random.default_rng(seed).integers(0, 2, size=K)]
    z, _ = r1cs.gen_witness(K, x0=1, ys=ys)

    # ---- BATCH: a single proof over all K instances ----
    proof, t_prove_batch = _time(lambda: flock.prove(z, K))
    ok, t_verify_batch = _time(lambda: flock.verify(proof))
    assert ok

    # ---- SEPARATE: K proofs, one instance each ----
    t_prove_sep = 0.0
    t_verify_sep = 0.0
    for i in range(K):
        zi, _ = r1cs.gen_witness(1, x0=int(z[i * r1cs.BASE + 1]),
                                 ys=[int(z[i * r1cs.BASE + 2])])
        pi, dt = _time(lambda: flock.prove(zi, 1))
        t_prove_sep += dt
        _, dt = _time(lambda: flock.verify(pi))
        t_verify_sep += dt

    return {
        "K": K,
        "prove_batch": t_prove_batch, "prove_sep": t_prove_sep,
        "verify_batch": t_verify_batch, "verify_sep": t_verify_sep,
        "proofs_batch": 1, "proofs_sep": K,
    }


def main():
    print(f"{'K':>4} | {'prove batch':>11} {'prove sep':>10} {'speedup':>7} "
          f"| {'verify batch':>12} {'verify sep':>10} {'speedup':>7} "
          f"| {'#proofs b/s':>11}")
    print("-" * 92)
    for k in range(1, 7):                 # K = 2 .. 64
        K = 1 << k
        r = bench(K)
        psp = r["prove_sep"] / r["prove_batch"]
        vsp = r["verify_sep"] / r["verify_batch"]
        print(f"{K:>4} | {r['prove_batch']*1e3:>9.1f}ms {r['prove_sep']*1e3:>8.1f}ms "
              f"{psp:>6.2f}x | {r['verify_batch']*1e3:>10.1f}ms {r['verify_sep']*1e3:>8.1f}ms "
              f"{vsp:>6.2f}x | {r['proofs_batch']:>4}/{r['proofs_sep']:<5}")
    print("\nnote: SEPARATE cannot prove the hash-chain (out_i = x_{i+1}) - that")
    print("cross-instance binding exists only in the BATCH witness (glue G).")


if __name__ == "__main__":
    main()
