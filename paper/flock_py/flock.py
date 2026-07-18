"""
flock.py - the end-to-end Flock PIOP (prover + verifier), pedagogical edition.

Pipeline (mirrors the paper's protocol, sections 3-4):

  1. Commit   : PCS-commit the witness MLE z_hat            (section 3, step 1)
  2. Zerocheck: sum_i eq(r,i)*(a[i]*b[i] - c[i]) = 0        (the rank-1 / Hadamard
                where a=Az, b=Bz, c=Cz                        constraint, section 3)
  3. Lincheck : reduce a(r_y),b(r_y),c(r_y) to ONE z-claim  (batched via alpha,
                using the block-diagonal collapse of 4.1)      sections 3, 4.1
  4. Glue     : prove the hash-chain Gz = 0 -> a z-claim     (section 4.6)
  5. Open     : PCS-open z_hat at the lincheck & glue points (section 3, step 5)

Everything is Fiat-Shamir'd through one shared SHA-256 transcript, so prove()
returns a self-contained non-interactive proof that verify() checks.

Sizes here are tiny for clarity; the structure is exactly the paper's.
"""

import numpy as np
from field import GF
from mle import eq_vector, eval_mle, num_vars
from transcript import Transcript
import sumcheck
import pcs
import r1cs


def _combine_zerocheck(ts):
    """Zerocheck summand eq(r,b) * (a(b)*b(b) - c(b)); each factor is multilinear,
    so a single variable appears in at most 3 factors -> degree 3 per round."""
    eqw, a, b, c = ts
    return eqw * (a * b - c)              # degree 3 in the bound variable


def _combine_product(ts):
    """Lincheck / glue summand eta(b) * z(b): two multilinear factors -> degree 2."""
    eta, z = ts
    return eta * z                        # degree 2 (lincheck / glue marginal)


def _matrix_marginals(r_y, K):
    """
    eq-weighted column marginals eta_M = eq_vector(r_y) @ (I_K (x) M0), computed via
    the block-diagonal collapse: eta_M = (E @ M0).flatten with E = eq(r_y) as K x BASE.
    Returns (etaA, etaB, etaC), each a length-2^m table. (Paper section 4.1: the
    outer K rows contribute only a cheap eq-factor; work is on the small base M0.)
    """
    E = eq_vector(r_y).reshape(K, r1cs.BASE)
    A0 = GF(r1cs.A0.astype(np.int64))
    B0 = GF(r1cs.B0.astype(np.int64))
    C0 = GF(r1cs.C0.astype(np.int64))
    etaA = (E @ A0).reshape(-1)
    etaB = (E @ B0).reshape(-1)
    etaC = (E @ C0).reshape(-1)
    return etaA, etaB, etaC


def prove(z_bits, K):
    z = GF(z_bits.astype(np.int64))
    m = num_vars(z)
    tr = Transcript()

    # 1. commit to the witness
    pcs_prover, cm = pcs.commit(z)
    tr.absorb_bytes(cm.root)

    # the three sides of the constraint, a=Az, b=Bz, c=Cz (block-diagonal apply)
    a = r1cs.apply_base(r1cs.A0, z_bits, K)
    b = r1cs.apply_base(r1cs.B0, z_bits, K)
    c = r1cs.apply_base(r1cs.C0, z_bits, K)

    # 2. zerocheck: prove a o b - c vanishes on the hypercube
    r = tr.challenge_vec(m)
    eqw = eq_vector(r)
    zc_rounds, r_y, zc_final = sumcheck.prove([eqw, a, b, c], 3, _combine_zerocheck, tr)
    _, va, vb, vc = zc_final                    # a(r_y), b(r_y), c(r_y)
    tr.absorb_fe_list([va, vb, vc])

    # 3. lincheck: batch the three claims and reduce to a single z-claim
    alpha = tr.challenge_vec(3)
    etaA, etaB, etaC = _matrix_marginals(r_y, K)
    eta = alpha[0] * etaA + alpha[1] * etaB + alpha[2] * etaC
    # (the prover never sends this claim - the verifier recomputes it from
    # va,vb,vc itself; shown here to mirror the verify() side line for line)
    lc_claim = alpha[0] * va + alpha[1] * vb + alpha[2] * vc
    lc_rounds, r_x, lc_final = sumcheck.prove([eta, z], 2, _combine_product, tr)
    _, z_rx = lc_final                           # z(r_x)
    tr.absorb_fe(z_rx)

    # 4. glue: prove the hash-chain Gz = 0 -> another z-claim
    G = r1cs.glue_matrix(K)
    tau = tr.challenge_vec(m)
    etaG = eq_vector(tau) @ G
    g_rounds, r_g, g_final = sumcheck.prove([etaG, z], 2, _combine_product, tr)
    _, z_rg = g_final                            # z(r_g)
    tr.absorb_fe(z_rg)

    # 5. open the committed witness at both points the protocol landed on
    open_x = pcs.open(pcs_prover, r_x, tr)
    open_g = pcs.open(pcs_prover, r_g, tr)

    return {
        "cm": cm, "m": m, "K": K,
        "zc_rounds": zc_rounds, "va": va, "vb": vb, "vc": vc,
        "lc_rounds": lc_rounds, "z_rx": z_rx,
        "g_rounds": g_rounds, "z_rg": z_rg,
        "open_x": open_x, "open_g": open_g,
    }


def verify(proof):
    cm = proof["cm"]
    m = proof["m"]
    K = proof["K"]
    tr = Transcript()
    tr.absorb_bytes(cm.root)

    # 2. zerocheck (initial claim 0: the summand must vanish on the whole cube)
    r = tr.challenge_vec(m)
    r_y, zc_expected = sumcheck.verify(proof["zc_rounds"], 3, GF(0), tr)
    va, vb, vc = GF(proof["va"]), GF(proof["vb"]), GF(proof["vc"])
    tr.absorb_fe_list([va, vb, vc])
    # final point: eq(r, r_y) * (va*vb - vc) must equal the folded sumcheck value
    e_ry = eval_mle(eq_vector(r), r_y)
    if e_ry * (va * vb - vc) != zc_expected:
        raise ValueError("zerocheck: final point relation failed")

    # 3. lincheck: recompute the (public) matrix marginals and check the z-claim
    alpha = tr.challenge_vec(3)
    etaA, etaB, etaC = _matrix_marginals(r_y, K)
    eta = alpha[0] * etaA + alpha[1] * etaB + alpha[2] * etaC
    lc_claim = alpha[0] * va + alpha[1] * vb + alpha[2] * vc
    r_x, lc_expected = sumcheck.verify(proof["lc_rounds"], 2, lc_claim, tr)
    z_rx = GF(proof["z_rx"])
    tr.absorb_fe(z_rx)
    if eval_mle(eta, r_x) * z_rx != lc_expected:
        raise ValueError("lincheck: final point relation failed")

    # 4. glue: recompute G's marginal at tau and check the chain z-claim
    G = r1cs.glue_matrix(K)
    tau = tr.challenge_vec(m)
    etaG = eq_vector(tau) @ G
    r_g, g_expected = sumcheck.verify(proof["g_rounds"], 2, GF(0), tr)
    z_rg = GF(proof["z_rg"])
    tr.absorb_fe(z_rg)
    if eval_mle(etaG, r_g) * z_rg != g_expected:
        raise ValueError("glue: final point relation failed")

    # 5. PCS: the opened evaluations must match the z-claims the sumchecks produced
    v_x = pcs.verify(cm, r_x, proof["open_x"], tr)
    v_g = pcs.verify(cm, r_g, proof["open_g"], tr)
    if v_x != z_rx:
        raise ValueError("PCS open at r_x disagrees with lincheck z-claim")
    if v_g != z_rg:
        raise ValueError("PCS open at r_g disagrees with glue z-claim")

    return True
