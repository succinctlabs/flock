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

The suite doubles as a WORKED EXAMPLE of the protocol math: each section
pretty-prints the actual values flowing through the operation it tests -
inputs, intermediate steps, and the checks - so you can follow the algebra on
screen next to the paper. Nothing printed is mocked: every number comes out of
the same code paths the assertions exercise, and the demos deliberately use
operands small enough (single hex digits) that you can redo the arithmetic by
hand. Genuinely random 128-bit elements are abbreviated as 0xHEAD..TAIL purely
for layout; the code always computes on the full value.
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
    print(f"\n  ok  {name}")


def fail(name, e):
    print(f" FAIL {name}: {e}")
    raise SystemExit(1)


# ------------------------- pretty-printing helpers -----------------------------
# Everything in this block is formatting only - none of it participates in the
# checks. The one exception is replay_sumcheck(), which deliberately re-runs the
# verifier's math (with asserts) so that what it prints CANNOT drift from what
# sumcheck.verify actually accepts.

WIDTH = 78


def banner(title):
    """Section header - one per protocol building block."""
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def step(label, text):
    """A numbered step (or phase, or attack) inside a section."""
    print(f"\n  [{label}] {text}")


def note(text, indent=6):
    """One indented line of worked math."""
    print(" " * indent + text)


def fe(x):
    """Format a field element for display. Small values (the hand-checkable
    demos) print in full; big ones (128-bit challenges, sumcheck messages)
    abbreviate to 0xHEAD..TAIL so lines stay readable. Display only - the
    underlying 128-bit value is always what the code computes with."""
    v = int(x)
    if v < (1 << 32):
        return f"0x{v:x}"
    h = f"{v:032x}"
    return f"0x{h[:4]}..{h[-4:]}"


def fe_poly(x):
    """Format a small field element as a polynomial in x over F2. Bit i of the
    integer representation is the coefficient of x^i - this makes the carry-less
    arithmetic in the field demo legible as plain polynomial algebra."""
    v = int(x)
    if v == 0:
        return "0"
    terms = []
    for i in range(v.bit_length() - 1, -1, -1):
        if (v >> i) & 1:
            terms.append("1" if i == 0 else ("x" if i == 1 else f"x^{i}"))
    return " + ".join(terms)


def vec(xs, limit=8):
    """Format a vector of field elements, truncating very long ones."""
    xs = list(xs)
    body = ", ".join(fe(x) for x in xs[:limit])
    if len(xs) > limit:
        body += f", ... ({len(xs)} elements)"
    return f"[{body}]"


def bits(v, group=r1cs.BASE):
    """Format a 0/1 vector as bit blocks, one block per circuit instance -
    the natural way to read a batched witness z = [1,x,y,out | 1,x,y,out | ...]."""
    s = "".join(str(int(x)) for x in np.array(v))
    return " ".join(s[i:i + group] for i in range(0, len(s), group))


def replay_sumcheck(round_evals, degree, claim, tr, indent=6):
    """Print-and-check replay of sumcheck.verify(): identical transcript absorbs
    in the identical order, hence identical challenges, and the same two facts
    re-checked (with asserts) every round:
      (1) g_t(0) + g_t(1) must equal the running claim - the sum over the
          remaining cube, split by the value of the variable being bound;
      (2) the next running claim becomes g_t(r_t), Lagrange-interpolated from
          the degree+1 samples the prover sent.
    Returns (challenges, final_expected) exactly like sumcheck.verify, so the
    caller can assert the replay agrees with the real verifier."""
    xs = GF(list(range(degree + 1)))
    challenges = []
    cur = claim
    for t, evals in enumerate(round_evals):
        evals = GF(evals)
        g01 = GF(evals[0]) + GF(evals[1])
        assert g01 == cur, "replay diverged from sumcheck.verify"
        note(f"round {t}: prover sends g{t}, sampled at x = 0..{degree}: {vec(evals)}", indent)
        note(f"         g{t}(0) + g{t}(1) = {fe(g01)}  == running claim  ok", indent)
        tr.absorb_fe_list(evals)
        ch = tr.challenge()
        challenges.append(ch)
        cur = sumcheck._lagrange_interpolate_eval(xs, evals, ch)
        note(f"         challenge r{t} = {fe(ch)}  ->  new claim g{t}(r{t}) = {fe(cur)}", indent)
    return GF(challenges), cur


# ------------------------------- field demo -------------------------------------

def demo_field():
    """Not a counted test - a worked tour of GF(2^128) arithmetic (field.py),
    with operands small enough to redo by hand. The asserts pin every printed
    identity to what galois actually computes."""
    banner("FIELD  GF(2^128), irreducible p(x) = x^128 + x^7 + x^2 + x + 1 (field.py)")
    print("  An element is a 128-bit integer whose bit i is the coefficient of x^i.")
    print("  Witness DATA lives in the {0,1} subfield; verifier CHALLENGES range over")
    print("  the whole field - that gap is where Schwartz-Zippel soundness comes from.")

    step(1, "addition is coefficientwise XOR (characteristic 2, so 1 + 1 = 0)")
    a, b = GF(0b1010), GF(0b0011)              # x^3 + x   and   x + 1
    s = a + b
    note(f"a     = {int(a):#06b} = {fe_poly(a)}")
    note(f"b     = {int(b):#06b} = {fe_poly(b)}")
    note(f"a + b = {int(s):#06b} = {fe_poly(s)}    (the two x terms cancelled)")
    assert int(s) == int(a) ^ int(b)
    note(f"a + a = {fe(a + a)}    (every element is its own negative, so + and - coincide)")
    assert a + a == GF(0)

    step(2, "multiplication is carry-less polynomial product, then reduce mod p(x)")
    small = GF(3) * GF(3)
    note("no reduction needed while degrees stay below 128:")
    note(f"    (x+1)*(x+1) = x^2 + 2x + 1 = x^2 + 1 (the 2x vanishes):  GF(3)*GF(3) = GF({int(small)})")
    assert int(small) == 0b101
    hi = GF(1 << 127) * GF(2)
    note("crossing degree 128 triggers the reduction x^128 = x^7 + x^2 + x + 1 (mod p):")
    note(f"    x^127 * x = x^128  ->  GF(2^127)*GF(2) = {int(hi):#b} = {fe_poly(hi)}")
    assert int(hi) == 0b10000111

    step(3, "inverses exist (F is a FIELD), so division is well-defined")
    c = GF(0x0123456789abcdef0123456789abcdef)     # an arbitrary big element
    ci = GF(1) / c
    note(f"c        = {fe(c)}")
    note(f"c^-1     = {fe(ci)}")
    note(f"c * c^-1 = {fe(c * ci)}")
    assert c * ci == GF(1)


# ------------------------------- MLE -------------------------------------------

def test_mle_agrees_on_cube():
    """Defining property of the multilinear extension: f_hat interpolates f,
    i.e. evaluating at every boolean corner returns the original table entry."""
    banner("MLE  f_hat(x) = sum_b f(b) * eq(b, x) - table -> polynomial (mle.py)")

    # Worked example small enough to follow by hand: m = 2 variables (a 4-entry
    # table) evaluated at the deliberately tiny point r = (2, 3), so every eq
    # weight is a one-hex-digit carry-less product. Remember: in this field
    # subtraction IS addition (XOR), so 1 - 2 = 1 + 2 = 3.
    t = GF([3, 1, 4, 1])
    r = GF([2, 3])
    print("  worked example: m = 2, table f = [3, 1, 4, 1], point r = (2, 3)")
    print("  (little-endian corners: f(b0,b1) lives at index b0 + 2*b1)")

    step(1, "the eq weights: eq(b, r) = prod_v (r_v if b_v = 1 else 1 - r_v)")
    eqw = eq_vector(r)
    for i in range(4):
        b0, b1 = i & 1, (i >> 1) & 1
        f0 = r[0] if b0 else GF(1) - r[0]
        f1 = r[1] if b1 else GF(1) - r[1]
        s0 = " r0 " if b0 else "1-r0"
        s1 = " r1 " if b1 else "1-r1"
        note(f"b=({b0},{b1}):  ({s0})*({s1}) = {fe(f0)} * {fe(f1)} = {fe(f0 * f1)}")
        assert f0 * f1 == eqw[i]
    total = GF(np.sum(eqw))
    note(f"sum of the four weights = {fe(total)}    (partition of unity: always exactly 1)")
    assert total == GF(1)

    step(2, "evaluate: f_hat(r) = <f, eq> - one weighted sum over the table")
    v = eval_mle(t, r)
    note("f_hat(2,3) = " + " + ".join(f"{int(t[i])}*{fe(eqw[i])}" for i in range(4)))
    note("           = " + " + ".join(fe(GF(t[i]) * eqw[i]) for i in range(4))
         + f" = {fe(v)}    (+ is XOR)")
    assert GF(np.sum(GF(t) * eqw)) == v

    step(3, "interpolation: at boolean corners the MLE reproduces the table")
    corners = []
    for i in range(4):
        pt = GF([(i >> vbit) & 1 for vbit in range(2)])
        cv = eval_mle(t, pt)
        assert cv == t[i]
        corners.append(f"f_hat({int(pt[0])},{int(pt[1])}) = {int(cv)}")
    note("   ".join(corners))
    note("which is exactly the table f = [3, 1, 4, 1] again  ok")

    # The counted test: the same property on a random m = 4 table, all 16 corners.
    m = 4
    t = GF.Random(1 << m)
    for i in range(1 << m):
        pt = GF([(i >> v) & 1 for v in range(m)])
        assert eval_mle(t, pt) == t[i]
    note("(re-checked quietly on a random m = 4 table: all 16 corners agree)", 2)
    ok("MLE agrees with table on all 2^m corners")


# ----------------------------- sumcheck ----------------------------------------

def test_sumcheck_roundtrip_and_soundness():
    """Honest sumcheck verifies: challenges match, the verifier's final expected
    value equals P at the bound point, and the bound tables really are the MLEs
    at the challenge point. Then: flipping one round message must be rejected."""
    banner("SUMCHECK  reduce a sum over {0,1}^m to one random point (sumcheck.py)")
    m = 4
    a, b, c = GF.Random(1 << m), GF.Random(1 << m), GF.Random(1 << m)
    r = GF.Random(m)
    eqw = eq_vector(r)

    def combine(ts):
        e, a, b, c = ts
        return e * (a * b - c)

    claim = GF(np.sum(combine([eqw, a, b, c])))
    print(f"  statement: claim = sum over b in {{0,1}}^{m} of  eq(w,b)*(a(b)*b(b) - c(b))")
    print(f"  with random tables a, b, c (so the values below are full 128-bit elements).")
    print(f"  Degree 3 per variable -> each round message is 4 samples of a univariate g.")
    note(f"claimed sum = {fe(claim)}", 2)

    rounds, ch, final = sumcheck.prove([eqw, a, b, c], 3, combine, Transcript())
    chv, expected = sumcheck.verify(rounds, 3, claim, Transcript())
    assert np.array_equal(np.array(ch), np.array(chv))
    assert combine([GF([x]) for x in final])[0] == expected
    assert final[1] == eval_mle(a, ch)

    step(1, f"the {m} rounds, replayed with the verifier's own two checks per round")
    rch, rfinal = replay_sumcheck(rounds, 3, claim, Transcript())
    # the replay must agree with the real verifier on everything it derived
    assert np.array_equal(np.array(rch), np.array(chv))
    assert rfinal == expected

    step(2, "final point: after m rounds the sum has shrunk to ONE point r = (r0..r3)")
    e_, a_, b_, c_ = final
    note(f"prover's fully-bound tables:  eq = {fe(e_)}, a = {fe(a_)}, b = {fe(b_)}, c = {fe(c_)}")
    note(f"P at the point = eq*(a*b - c) = {fe(e_ * (a_ * b_ - c_))}  == final claim {fe(expected)}  ok")
    note(f"cross-check: binding table a to the challenges equals the MLE a(r): {fe(eval_mle(a, ch))}  ok")
    ok("sumcheck prove/verify + final-point consistency")

    bad = [x.copy() for x in rounds]
    bad[0] = bad[0].copy(); bad[0][0] = bad[0][0] + GF(1)
    step(3, "soundness: add 1 to g0(0) and hand the verifier the altered messages")
    note(f"g0(0): {fe(rounds[0][0])}  ->  {fe(bad[0][0])}   (now g0(0)+g0(1) != claim)")
    try:
        sumcheck.verify(bad, 3, claim, Transcript())
        fail("sumcheck soundness", "tamper not caught")
    except ValueError as e:
        note(f"verifier raises: {e}")
        ok("tampered sumcheck rejected")


# -------------------------------- PCS ------------------------------------------

def test_pcs_roundtrip_and_soundness():
    """PCS round-trip on several sizes (odd/even m exercise both row/col splits),
    then two forgeries: a wrong claimed value v, and a tampered committed column
    (which must fail the Merkle check or a code-consistency check)."""
    banner("PCS  hash-based commitment: Merkle + Reed-Solomon, Ligero-style (pcs.py)")

    # Worked example: commit to the SAME 4-entry table used in the MLE section and
    # prove its evaluation at the same point r = (2,3) - so the value proven here
    # is the one computed by hand there. m = 2 -> a 2x2 matrix; rate-1/2 code ->
    # each 2-entry row becomes a 4-entry codeword.
    t = GF([3, 1, 4, 1])
    r = GF([2, 3])
    print("  worked example: commit f = [3, 1, 4, 1], then prove the MLE section's")
    print("  claim f_hat(2,3) against that commitment.")
    prover, cm = pcs.commit(t)

    step(1, "shape the table into a matrix and Reed-Solomon-encode each row")
    note(f"T = f reshaped to {prover.n_rows}x{prover.n_cols} (row i = t[{prover.n_cols}i .. {prover.n_cols}i+{prover.n_cols - 1}]):")
    for row in prover.T:
        note(f"    {vec(row)}")
    note(f"each row's entries are coefficients of a polynomial, evaluated at x = 0..{prover.n_enc - 1}:")
    for i, row in enumerate(prover.Enc):
        note(f"    Enc row {i}: {vec(row)}")
    note("(check row 0 by hand: p(x) = 3 + 1*x gives p(0)=3, p(1)=3+1=2, p(2)=3+2=1,")
    note(" p(3)=3+3=0 - additions are XOR. Redundancy is what makes spot-checks bite.)")

    step(2, "Merkle-commit the encoded COLUMNS; the root alone is the commitment")
    note(f"leaf q = SHA-256('leaf' || Enc[:,q]), {cm.n_enc} leaves  ->  root = {cm.root.hex()[:16]}..")

    step(3, "open at r = (2,3): the tensor split f_hat(r) = eq_rows^T . T . eq_cols")
    proof = pcs.open(prover, r, Transcript())
    eq_cols = eq_vector(r[:prover.m2])
    eq_rows = eq_vector(r[prover.m2:])
    note(f"r splits: col var -> eq_cols = {vec(eq_cols)},  row var -> eq_rows = {vec(eq_rows)}")
    note(f"prover sends the folded row  w = eq_rows^T . T = {vec(proof['w'])}")
    note(f"claimed value v = <w, eq_cols> = {fe(proof['v'])}   (the MLE section's f_hat(2,3))")
    assert proof["v"] == eval_mle(t, r)

    step(4, "verify: spot-check random columns against the root and the code")
    qs = proof["queries"]
    note(f"{len(qs)} Fiat-Shamir column queries -> columns {sorted(set(qs))}")
    note("(this demo code has only 4 columns, so the 24 queries repeat; real")
    note(" parameters make each query an independent binomial trial)")
    q, col, path = proof["openings"][0]
    note(f"take query q = {q}: opened column Enc[:,{q}] = {vec(col)}")
    assert pcs.Merkle.verify(cm.root, q, pcs._leaf_hash(col), path)
    note(f"Merkle: {len(path)} sibling hashes recompute the committed root  ok")
    enc_w = pcs._encode_vec(GF(proof["w"]), cm.n_enc)
    lhs = GF(np.sum(eq_rows * GF(col)))
    note(f"eval-consistency (the code is LINEAR, so encoding commutes with eq_rows):")
    note(f"    <eq_rows, Enc[:,{q}]> = {fe(lhs)}  ==  Enc(w)[{q}] = {fe(enc_w[q])}  ok")
    assert lhs == enc_w[q]
    enc_rg = pcs._encode_vec(GF(proof["r_gamma"]), cm.n_enc)
    plhs = GF(np.sum(GF(proof["gamma"]) * GF(col)))
    note(f"proximity (random combo gamma of the rows, same linearity trick):")
    note(f"    <gamma, Enc[:,{q}]> = {fe(plhs)}  ==  Enc(gamma^T T)[{q}] = {fe(enc_rg[q])}  ok")
    assert plhs == enc_rg[q]
    v = pcs.verify(cm, r, proof, Transcript())
    note(f"pcs.verify accepts and returns v = {fe(v)}")
    assert v == eval_mle(t, r)

    step(5, "the counted round-trip: m = 2, 3, 5, 6 (both row/col split parities)")
    for m in [2, 3, 5, 6]:
        t = GF.Random(1 << m)
        prover, cm = pcs.commit(t)
        r = GF.Random(m)
        proof = pcs.open(prover, r, Transcript())
        v = pcs.verify(cm, r, proof, Transcript())
        assert v == eval_mle(t, r)
    ok("PCS commit/open/verify matches MLE (m=2,3,5,6)")

    step(6, "soundness: two forgeries against a fresh m = 4 commitment")
    t = GF.Random(1 << 4)
    prover, cm = pcs.commit(t)
    r = GF.Random(4)
    proof = pcs.open(prover, r, Transcript())
    proof["v"] = proof["v"] + GF(1)
    note(f"forgery 1: claim v + 1 = {fe(proof['v'])} instead of the true evaluation")
    try:
        pcs.verify(cm, r, proof, Transcript())
        fail("PCS soundness (value)", "tampered value accepted")
    except ValueError as e:
        note(f"verifier raises: {e}")
        ok("PCS rejects tampered evaluation value")

    proof = pcs.open(prover, r, Transcript())
    q, col, path = proof["openings"][0]
    col2 = col.copy(); col2[0] = col2[0] + GF(1)
    proof["openings"][0] = (q, col2, path)
    note(f"forgery 2: flip one entry of opened column {q}; its leaf hash changes,")
    note("so the untouched authentication path can no longer reach the root")
    try:
        pcs.verify(cm, r, proof, Transcript())
        fail("PCS soundness (column)", "tampered column accepted")
    except ValueError as e:
        note(f"verifier raises: {e}")
        ok("PCS rejects tampered committed column")


# ------------------------------- R1CS demo --------------------------------------

def demo_r1cs():
    """Not a counted test - a worked example of the STATEMENT being proven
    (r1cs.py): the batched AND-gate R1CS and the hash-chain glue, checked
    directly over F2. These are the very equations the zerocheck, lincheck and
    glue sumchecks will re-assert algebraically in the end-to-end section."""
    banner("R1CS  batched AND-gates + hash-chain glue (r1cs.py)")
    print("  base gate: out = x AND y = x*y over F2. One instance's witness block is")
    print("  z0 = [1, x, y, out] (slots 0..3), and only R1CS row 0 is real:")
    print("  (A0 z0) * (B0 z0) = (C0 z0) row-wise says  x * y = out.")

    step(1, "the base selector matrices (rows 1..3 are 0 = 0 padding)")
    for name, M, what in [("A0", r1cs.A0, "slot 1 = x"),
                          ("B0", r1cs.B0, "slot 2 = y"),
                          ("C0", r1cs.C0, "slot 3 = out")]:
        rows = " ".join("[" + "".join(str(int(x)) for x in row) + "]" for row in M)
        note(f"{name} = {rows}    row 0 selects {what}")

    step(2, "a K = 4 hash-chain witness: x_(i+1) = out_i = x_i AND y_i")
    K, x0, ys = 4, 1, [1, 1, 0, 1]
    z, info = r1cs.gen_witness(K, x0, ys)
    note("i | x_i y_i | out_i = x_i AND y_i")
    for i in range(K):
        chain = f"-> chains into x_{i + 1}" if i < K - 1 else "(chain output)"
        note(f"{i} |  {info['xs'][i]}   {info['ys'][i]}  |    {info['outs'][i]}    {chain}")
    note(f"z = {bits(z)}    (one [1,x,y,out] block per instance)")

    step(3, "batched R1CS: A = I_K (x) A0, so Az just applies A0 to every block")
    a = r1cs.apply_base(r1cs.A0, z, K)
    b = r1cs.apply_base(r1cs.B0, z, K)
    c = r1cs.apply_base(r1cs.C0, z, K)
    note(f"Az        = {bits(a)}    (block i row 0 = x_i)")
    note(f"Bz        = {bits(b)}    (block i row 0 = y_i)")
    note(f"Cz        = {bits(c)}    (block i row 0 = out_i)")
    note(f"(Az)o(Bz) = {bits(a * b)}    elementwise == Cz  ok   (the Hadamard constraint)")
    assert r1cs.check_r1cs(z, K)

    step(4, "glue: Gz = 0 encodes the chain, one row per link (over F2, + is -)")
    for i in range(K - 1):
        note(f"G row {i}: z[{r1cs.out_index(i)}] (out_{i}) + z[{r1cs.in_index(i + 1)}] (x_{i + 1})  = 0")
    g = r1cs.glue_matrix(K) @ GF(z.astype(np.int64))
    note(f"G z = {bits(g)}  = 0  ok   (the cross-instance fact no single gate can see)")
    assert r1cs.check_glue(z, K)


# ------------------------------ end-to-end -------------------------------------

def show_flock_proof(proof):
    """Replay flock.verify() phase by phase, printing the verifier's math. The
    transcript operations are IDENTICAL to verify()'s (same absorbs, same order),
    so the challenges shown are the real ones, and every printed check is also
    asserted - the narration cannot drift from what verify() actually accepts."""
    cm, m, K = proof["cm"], proof["m"], proof["K"]
    tr = Transcript()
    tr.absorb_bytes(cm.root)

    step("phase 1/5", "COMMIT - the witness MLE z_hat, held only as a PCS root")
    note(f"K = {K} instances x {r1cs.BASE} slots = 2^{m} witness entries; root = {cm.root.hex()[:16]}..")

    step("phase 2/5", "ZEROCHECK - 0 = sum_b eq(r,b) * (Az(b)*Bz(b) - Cz(b))")
    r = tr.challenge_vec(m)
    note(f"verifier randomness r = {vec(r)}; the claim starts at literally 0")
    r_y, zc_expected = replay_sumcheck(proof["zc_rounds"], 3, GF(0), tr)
    va, vb, vc = GF(proof["va"]), GF(proof["vb"]), GF(proof["vc"])
    tr.absorb_fe_list([va, vb, vc])
    e_ry = eval_mle(eq_vector(r), r_y)
    lhs = e_ry * (va * vb - vc)
    note(f"prover claims a(r_y) = {fe(va)}, b(r_y) = {fe(vb)}, c(r_y) = {fe(vc)}")
    note(f"final point: eq(r,r_y)*(va*vb - vc) = {fe(lhs)}  == folded claim {fe(zc_expected)}  ok")
    assert lhs == zc_expected
    note("(va, vb, vc are so far only CLAIMS about Az, Bz, Cz - the lincheck earns them)")

    step("phase 3/5", "LINCHECK - reduce the three matrix claims to ONE claim about z")
    alpha = tr.challenge_vec(3)
    etaA, etaB, etaC = flock._matrix_marginals(r_y, K)
    eta = alpha[0] * etaA + alpha[1] * etaB + alpha[2] * etaC
    lc_claim = alpha[0] * va + alpha[1] * vb + alpha[2] * vc
    note(f"batching randomness alpha = {vec(alpha)}")
    note(f"claim = alpha0*va + alpha1*vb + alpha2*vc = {fe(lc_claim)}")
    note("eta = the same alpha-mix of the PUBLIC matrix marginals eq(r_y)^T M, which")
    note("the verifier computes itself on the small base matrices (the 4.1 collapse)")
    r_x, lc_expected = replay_sumcheck(proof["lc_rounds"], 2, lc_claim, tr)
    z_rx = GF(proof["z_rx"])
    tr.absorb_fe(z_rx)
    lhs = eval_mle(eta, r_x) * z_rx
    note(f"final point: eta(r_x) * z(r_x) = {fe(lhs)}  == folded claim {fe(lc_expected)}  ok")
    assert lhs == lc_expected
    note(f"-> surviving claim #1: z_hat(r_x) = {fe(z_rx)}   (the PCS settles it in phase 5)")

    step("phase 4/5", "GLUE - the hash-chain: 0 = sum_b (eq(tau)^T G)(b) * z(b)")
    tau = tr.challenge_vec(m)
    etaG = eq_vector(tau) @ r1cs.glue_matrix(K)
    note(f"tau = {vec(tau)};  etaG = eq(tau)^T G is public (built from the K-1 chain rows)")
    r_g, g_expected = replay_sumcheck(proof["g_rounds"], 2, GF(0), tr)
    z_rg = GF(proof["z_rg"])
    tr.absorb_fe(z_rg)
    lhs = eval_mle(etaG, r_g) * z_rg
    note(f"final point: etaG(r_g) * z(r_g) = {fe(lhs)}  == folded claim {fe(g_expected)}  ok")
    assert lhs == g_expected
    note(f"-> surviving claim #2: z_hat(r_g) = {fe(z_rg)}")

    step("phase 5/5", "OPEN - the PCS ties both surviving claims to the commitment")
    v_x = pcs.verify(cm, r_x, proof["open_x"], tr)
    note(f"open z_hat at r_x: v = {fe(v_x)}  == lincheck claim {fe(z_rx)}  ok")
    assert v_x == z_rx
    v_g = pcs.verify(cm, r_g, proof["open_g"], tr)
    note(f"open z_hat at r_g: v = {fe(v_g)}  == glue claim {fe(z_rg)}  ok")
    assert v_g == z_rg
    note("verifier accepts: the COMMITTED witness satisfies the gates AND the chain.")


def test_end_to_end():
    """Full pipeline (commit -> zerocheck -> lincheck -> glue -> open) accepts an
    honest hash-chain witness for batch sizes K = 2, 4, 8, 16."""
    banner("END-TO-END  commit -> zerocheck -> lincheck -> glue -> open (flock.py)")
    print("  Proving the SAME K = 4 chain witness from the R1CS section. Below is the")
    print("  verifier's complete view, replayed check by check from the actual proof.")
    z, _ = r1cs.gen_witness(4, 1, [1, 1, 0, 1])
    proof = flock.prove(z, 4)
    show_flock_proof(proof)
    assert flock.verify(proof)

    # The counted test: several batch sizes with (seeded-)random chains, quietly.
    for k in [1, 2, 3, 4]:
        K = 1 << k
        ys = [int(x) for x in np.random.default_rng(k).integers(0, 2, size=K)]
        z, _ = r1cs.gen_witness(K, x0=1, ys=ys)
        assert r1cs.check_r1cs(z, K) and r1cs.check_glue(z, K)
        assert flock.verify(flock.prove(z, K))
    note("(re-run quietly for K = 2, 4, 8, 16 with random chains: all accepted)", 2)
    ok("end-to-end prove/verify for K=2,4,8,16")


def test_soundness_bad_gate():
    """Flip one gate's output bit so x*y != out for instance 0. This violates the
    Hadamard constraint (Az) o (Bz) = Cz, so the ZEROCHECK must reject."""
    banner("SOUNDNESS  each attack is caught by the sub-protocol built to catch it")
    K = 4
    z, _ = r1cs.gen_witness(K, 1, [1, 1, 0, 1])
    step("attack 1", "corrupt ONE gate: flip out_0, so instance 0 claims 1 AND 1 = 0")
    note(f"z before: {bits(z)}")
    z[0 * r1cs.BASE + 3] ^= 1                 # corrupt out_0
    note(f"z after:  {bits(z)}    (block 0's out bit flipped)")
    note("now (Az)o(Bz) != Cz at instance 0, so the zerocheck's true sum is NOT 0;")
    note("its very first round message must satisfy g0(0)+g0(1) = 0 and cannot.")
    try:
        flock.verify(flock.prove(z, K))
        fail("bad-gate soundness", "accepted invalid witness")
    except ValueError as e:
        note(f"flock.verify raises: {e}")
        ok("corrupt AND-gate rejected (zerocheck)")


def test_soundness_broken_chain():
    """The subtler attack: every gate is individually valid (so the zerocheck
    passes) but instance 1's input is not instance 0's output. Only the GLUE
    check Gz = 0 - the batch-only cross-instance binding - can catch this."""
    K = 4
    z, _ = r1cs.gen_witness(K, 1, [1, 1, 0, 1])
    step("attack 2", "break the CHAIN: rewrite instance 1 as the perfectly valid gate 0 AND 1 = 0")
    note(f"z before: {bits(z)}")
    base = 1 * r1cs.BASE                        # instance 1: valid gate, wrong input
    z[base + 1], z[base + 2], z[base + 3] = 0, 1, 0
    assert (z[base + 1] & z[base + 2]) == z[base + 3]   # gate still valid
    note(f"z after:  {bits(z)}    (instance 1 fine in isolation, but x_1 = 0 != out_0 = 1)")
    note("every gate satisfies x*y = out, so the zerocheck sees NOTHING wrong;")
    note("but Gz != 0, so the glue sumcheck's claim of 0 is false and must fail.")
    try:
        flock.verify(flock.prove(z, K))
        fail("broken-chain soundness", "accepted broken chain")
    except ValueError as e:
        note(f"flock.verify raises: {e}")
        ok("broken hash-chain rejected (glue)")


if __name__ == "__main__":
    print("Flock (pedagogical) test suite")
    print("Each section prints the REAL math its test verifies: worked examples use")
    print("hand-checkable values; random 128-bit elements display as 0xHEAD..TAIL.")
    demo_field()
    test_mle_agrees_on_cube()
    test_sumcheck_roundtrip_and_soundness()
    test_pcs_roundtrip_and_soundness()
    demo_r1cs()
    test_end_to_end()
    test_soundness_bad_gate()
    test_soundness_broken_chain()
    print(f"\nAll {len(PASS)} checks passed.")
