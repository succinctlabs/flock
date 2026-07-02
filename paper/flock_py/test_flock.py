"""
test_flock.py - self-contained test suite for the pedagogical Flock.

Run:  python3 test_flock.py     (needs `galois` and `numpy`; see README.md)
Exits non-zero on the first failure; prints a summary otherwise.

Two kinds of tests, deliberately paired:
  - COMPLETENESS: honest prover + honest verifier accept (MLE identities,
    sumcheck round-trip, PCS round-trip, end-to-end for several batch sizes).
  - SOUNDNESS: each tampering is caught by the specific sub-protocol whose job
    it is to catch it (zerocheck for a bad gate, glue for a broken chain, the
    PCS for forged evaluations/columns, sumcheck for altered round messages).
"""

import numpy as np
from field import GF
from mle import eval_mle, eq_vector
from transcript import Transcript
import sumcheck
import pcs
import r1cs
import flock

PASS = []


def ok(name):
    PASS.append(name)
    print(f"  ok  {name}")


def fail(name, e):
    print(f" FAIL {name}: {e}")
    raise SystemExit(1)


# ------------------------------- MLE -------------------------------------------

def test_mle_agrees_on_cube():
    """Defining property of the multilinear extension: f_hat interpolates f,
    i.e. evaluating at every boolean corner returns the original table entry."""
    m = 4
    t = GF.Random(1 << m)
    for i in range(1 << m):
        pt = GF([(i >> v) & 1 for v in range(m)])
        assert eval_mle(t, pt) == t[i]
    ok("MLE agrees with table on all 2^m corners")


# ----------------------------- sumcheck ----------------------------------------

def test_sumcheck_roundtrip_and_soundness():
    """Honest sumcheck verifies: challenges match, the verifier's final expected
    value equals P at the bound point, and the bound tables really are the MLEs
    at the challenge point. Then: flipping one round message must be rejected."""
    m = 4
    a, b, c = GF.Random(1 << m), GF.Random(1 << m), GF.Random(1 << m)
    r = GF.Random(m)
    eqw = eq_vector(r)

    def combine(ts):
        e, a, b, c = ts
        return e * (a * b - c)

    claim = GF(np.sum(combine([eqw, a, b, c])))
    rounds, ch, final = sumcheck.prove([eqw, a, b, c], 3, combine, Transcript())
    chv, expected = sumcheck.verify(rounds, 3, claim, Transcript())
    assert np.array_equal(np.array(ch), np.array(chv))
    assert combine([GF([x]) for x in final])[0] == expected
    assert final[1] == eval_mle(a, ch)
    ok("sumcheck prove/verify + final-point consistency")

    bad = [x.copy() for x in rounds]
    bad[0] = bad[0].copy(); bad[0][0] = bad[0][0] + GF(1)
    try:
        sumcheck.verify(bad, 3, claim, Transcript())
        fail("sumcheck soundness", "tamper not caught")
    except ValueError:
        ok("tampered sumcheck rejected")


# -------------------------------- PCS ------------------------------------------

def test_pcs_roundtrip_and_soundness():
    """PCS round-trip on several sizes (odd/even m exercise both row/col splits),
    then two forgeries: a wrong claimed value v, and a tampered committed column
    (which must fail the Merkle check or a code-consistency check)."""
    for m in [2, 3, 5, 6]:
        t = GF.Random(1 << m)
        prover, cm = pcs.commit(t)
        r = GF.Random(m)
        proof = pcs.open(prover, r, Transcript())
        v = pcs.verify(cm, r, proof, Transcript())
        assert v == eval_mle(t, r)
    ok("PCS commit/open/verify matches MLE (m=2,3,5,6)")

    t = GF.Random(1 << 4)
    prover, cm = pcs.commit(t)
    r = GF.Random(4)
    proof = pcs.open(prover, r, Transcript())
    proof["v"] = proof["v"] + GF(1)
    try:
        pcs.verify(cm, r, proof, Transcript())
        fail("PCS soundness (value)", "tampered value accepted")
    except ValueError:
        ok("PCS rejects tampered evaluation value")

    proof = pcs.open(prover, r, Transcript())
    q, col, path = proof["openings"][0]
    col2 = col.copy(); col2[0] = col2[0] + GF(1)
    proof["openings"][0] = (q, col2, path)
    try:
        pcs.verify(cm, r, proof, Transcript())
        fail("PCS soundness (column)", "tampered column accepted")
    except ValueError:
        ok("PCS rejects tampered committed column")


# ------------------------------ end-to-end -------------------------------------

def test_end_to_end():
    """Full pipeline (commit -> zerocheck -> lincheck -> glue -> open) accepts an
    honest hash-chain witness for batch sizes K = 2, 4, 8, 16."""
    for k in [1, 2, 3, 4]:
        K = 1 << k
        ys = [int(x) for x in np.random.default_rng(k).integers(0, 2, size=K)]
        z, _ = r1cs.gen_witness(K, x0=1, ys=ys)
        assert r1cs.check_r1cs(z, K) and r1cs.check_glue(z, K)
        assert flock.verify(flock.prove(z, K))
    ok("end-to-end prove/verify for K=2,4,8,16")


def test_soundness_bad_gate():
    """Flip one gate's output bit so x*y != out for instance 0. This violates the
    Hadamard constraint (Az) o (Bz) = Cz, so the ZEROCHECK must reject."""
    K = 4
    z, _ = r1cs.gen_witness(K, 1, [1, 1, 0, 1])
    z[0 * r1cs.BASE + 3] ^= 1                 # corrupt out_0
    try:
        flock.verify(flock.prove(z, K))
        fail("bad-gate soundness", "accepted invalid witness")
    except ValueError:
        ok("corrupt AND-gate rejected (zerocheck)")


def test_soundness_broken_chain():
    """The subtler attack: every gate is individually valid (so the zerocheck
    passes) but instance 1's input is not instance 0's output. Only the GLUE
    check Gz = 0 - the batch-only cross-instance binding - can catch this."""
    K = 4
    z, _ = r1cs.gen_witness(K, 1, [1, 1, 0, 1])
    base = 1 * r1cs.BASE                        # instance 1: valid gate, wrong input
    z[base + 1], z[base + 2], z[base + 3] = 0, 1, 0
    assert (z[base + 1] & z[base + 2]) == z[base + 3]   # gate still valid
    try:
        flock.verify(flock.prove(z, K))
        fail("broken-chain soundness", "accepted broken chain")
    except ValueError:
        ok("broken hash-chain rejected (glue)")


if __name__ == "__main__":
    print("Flock (pedagogical) test suite\n")
    test_mle_agrees_on_cube()
    test_sumcheck_roundtrip_and_soundness()
    test_pcs_roundtrip_and_soundness()
    test_end_to_end()
    test_soundness_bad_gate()
    test_soundness_broken_chain()
    print(f"\nAll {len(PASS)} checks passed.")
