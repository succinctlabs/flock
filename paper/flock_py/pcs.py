"""
pcs.py - a real hash-based multilinear polynomial commitment (Ligero/Brakedown style).

This plays the role of Flock's PCS (Ligerito + ring-switching in the paper). It is
transparent and post-quantum: the ONLY cryptographic assumption is SHA-256
(Merkle commitments + Fiat-Shamir queries). No pairings, no trusted setup.

Scheme (classic Ligero tensor-query multilinear commitment):
  - A multilinear poly is stored as its 2^m evaluation table t, reshaped to an
    (n_rows x n_cols) matrix T, T[i][j] = t[i*n_cols + j].
  - Each ROW is Reed-Solomon encoded (coeffs -> evaluations at n_enc points),
    giving Enc (n_rows x n_enc). We Merkle-commit the COLUMNS of Enc; the root is
    the commitment.
  - The evaluation f_hat(r) factorizes through the tensor structure:
        f_hat(r) = eq_rows^T . T . eq_cols        (eq split into high/low vars)
    so the prover sends w = eq_rows^T . T (length n_cols) and the verifier checks
    v == <w, eq_cols>.
  - Soundness comes from opening a few random columns q (Fiat-Shamir): because the
    RS code is linear, eq_rows^T . Enc[:,q] must equal Enc(w)[q], and a random
    combination gamma^T . Enc[:,q] must equal Enc(gamma^T.T)[q]. These tie the
    sent row w to the committed matrix at random places.

This is pedagogical: conservative fixed parameters, O(N) verifier, not optimized.
"""

import hashlib
import numpy as np
from field import GF
from mle import eq_vector, num_vars
from transcript import fe_to_bytes

RATE_INV = 2       # codeword length = RATE_INV * n_cols  (rate 1/2, like the paper)
NUM_QUERIES = 24   # opened columns; soundness ~ (1/2)^Q on top of the field size


# ----------------------------- Merkle tree -------------------------------------

def _hash(data: bytes) -> bytes:
    """Internal-node hash: SHA-256 of the two child digests concatenated."""
    return hashlib.sha256(data).digest()


def _leaf_hash(col_values) -> bytes:
    """Hash one codeword column (a vector of field elements) into a leaf.
    The b"leaf" prefix domain-separates leaves from internal nodes, blocking
    the classic second-preimage trick of passing an internal node off as a leaf."""
    h = hashlib.sha256(b"leaf")
    for x in col_values:
        h.update(fe_to_bytes(x))
    return h.digest()


class Merkle:
    """Binary Merkle tree over a list of leaves (already hashed)."""

    def __init__(self, leaves):
        self.n = len(leaves)
        assert self.n & (self.n - 1) == 0, "leaf count must be a power of two"
        self.levels = [list(leaves)]
        cur = leaves
        while len(cur) > 1:
            nxt = [_hash(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)]
            self.levels.append(nxt)
            cur = nxt
        self.root = cur[0]

    def path(self, idx):
        """Authentication path for leaf `idx`: the sibling hash at every level.
        (idx ^ 1 flips the lowest bit - left child <-> right child.)"""
        siblings = []
        for level in self.levels[:-1]:
            siblings.append(level[idx ^ 1])
            idx >>= 1
        return siblings

    @staticmethod
    def verify(root, idx, leaf, siblings):
        """Recompute the root from a leaf and its authentication path.
        At each level, `idx & 1` says whether the current hash is the right or
        left child, i.e. which side the sibling concatenates on."""
        h = leaf
        for sib in siblings:
            if idx & 1:
                h = _hash(sib + h)
            else:
                h = _hash(h + sib)
            idx >>= 1
        return h == root


# --------------------------- Reed-Solomon encode -------------------------------

def _encode_rows(mat, n_enc):
    """
    RS-encode each row of `mat` (n_rows x n_cols): treat the row as polynomial
    coefficients and evaluate at the fixed points {0,1,...,n_enc-1} in F.
    Returns an (n_rows x n_enc) galois array.
    """
    n_rows, n_cols = mat.shape
    points = GF(np.arange(n_enc, dtype=np.uint64))
    # Vandermonde: V[p, j] = points[p]^j
    V = GF(np.ones((n_enc, n_cols), dtype=np.uint64))
    for j in range(1, n_cols):
        V[:, j] = V[:, j - 1] * points
    # Enc[i, p] = sum_j mat[i,j] * points[p]^j  =>  Enc = mat @ V^T
    return GF(mat) @ V.T


def _encode_vec(vec, n_enc):
    return _encode_rows(GF(vec).reshape(1, -1), n_enc)[0]


# ------------------------------ commit / open ----------------------------------

def _shape(m):
    """Split m variables into (row vars, col vars) as balanced as possible."""
    m2 = m // 2            # low vars -> columns
    m1 = m - m2            # high vars -> rows
    return m1, m2


class Commitment:
    """What the verifier holds: the Merkle root plus the (public) matrix shape.
    The root binds the prover to one specific encoded matrix; the shape lets the
    verifier recompute eq splits, re-encode rows, and reduce query indices."""

    def __init__(self, root, m, n_rows, n_cols, n_enc):
        self.root = root
        self.m = m
        self.n_rows, self.n_cols, self.n_enc = n_rows, n_cols, n_enc


class _Prover:
    """Holds the opened matrix so multiple evaluations can be proved."""

    def __init__(self, table):
        self.m = num_vars(table)
        self.m1, self.m2 = _shape(self.m)
        self.n_rows, self.n_cols = 1 << self.m1, 1 << self.m2
        self.n_enc = RATE_INV * self.n_cols
        self.T = GF(table).reshape(self.n_rows, self.n_cols)
        self.Enc = _encode_rows(self.T, self.n_enc)                 # n_rows x n_enc
        leaves = [_leaf_hash(self.Enc[:, q]) for q in range(self.n_enc)]
        self.tree = Merkle(leaves)
        self.commitment = Commitment(
            self.tree.root, self.m, self.n_rows, self.n_cols, self.n_enc
        )


def commit(table):
    """Commit to the multilinear polynomial given by `table` (its 2^m evaluations).
    Returns (prover_state, commitment): the prover keeps the state to answer
    open() later; only the commitment (root + shape) is sent to the verifier."""
    p = _Prover(table)
    return p, p.commitment


def open(prover, point, transcript):
    """Produce an evaluation proof that f_hat(point) = v (v returned in proof)."""
    P = prover
    r = GF(point)
    eq_cols = eq_vector(r[: P.m2])
    eq_rows = eq_vector(r[P.m2:])
    w = eq_rows @ P.T                       # length n_cols; v = <w, eq_cols>
    v = GF(np.sum(w * eq_cols))

    transcript.absorb_bytes(P.commitment.root)
    transcript.absorb_fe_list(r)
    transcript.absorb_fe_list(w)
    transcript.absorb_fe(v)

    # proximity random combination over the rows
    gamma = transcript.challenge_vec(P.n_rows)
    r_gamma = gamma @ P.T                    # length n_cols

    transcript.absorb_fe_list(r_gamma)

    # query columns
    queries = [int(transcript.challenge()) % P.n_enc for _ in range(NUM_QUERIES)]
    openings = []
    for q in queries:
        col = P.Enc[:, q]
        openings.append((q, col, P.tree.path(q)))

    return {
        "v": v, "w": w, "gamma": gamma, "r_gamma": r_gamma,
        "queries": queries, "openings": openings,
    }


def verify(commitment, point, proof, transcript):
    """Check an evaluation proof; returns the proven value v = f_hat(point) or
    raises ValueError. Three layers of checks, mirroring the module docstring:
    (1) the sent row w actually yields the claimed v, (2) Fiat-Shamir replay ties
    gamma and the query set to the transcript, (3) at each queried column, the
    Merkle path proves the column is committed, and the two linear-code identities
    tie w (eval-consistency) and gamma^T.T (proximity) to that committed data."""
    C = commitment
    r = GF(point)
    m1, m2 = _shape(C.m)
    eq_cols = eq_vector(r[:m2])
    eq_rows = eq_vector(r[m2:])
    w = GF(proof["w"])
    v = GF(proof["v"])

    # 1) claimed evaluation is consistent with the sent row w
    if GF(np.sum(w * eq_cols)) != v:
        raise ValueError("PCS: v != <w, eq_cols>")

    # replay Fiat-Shamir exactly as the prover did
    transcript.absorb_bytes(C.root)
    transcript.absorb_fe_list(r)
    transcript.absorb_fe_list(w)
    transcript.absorb_fe(v)
    gamma = transcript.challenge_vec(C.n_rows)
    if not np.array_equal(np.array(gamma), np.array(proof["gamma"])):
        raise ValueError("PCS: gamma mismatch (transcript desync)")
    r_gamma = GF(proof["r_gamma"])
    transcript.absorb_fe_list(r_gamma)
    queries = [int(transcript.challenge()) % C.n_enc for _ in range(NUM_QUERIES)]
    if queries != proof["queries"]:
        raise ValueError("PCS: query set mismatch")

    enc_w = _encode_vec(w, C.n_enc)
    enc_rg = _encode_vec(r_gamma, C.n_enc)

    for (q, col, path) in proof["openings"]:
        if q not in queries:
            raise ValueError("PCS: opened a non-queried column")
        # Merkle authenticity of the committed column
        if not Merkle.verify(C.root, q, _leaf_hash(col), path):
            raise ValueError("PCS: bad Merkle path")
        col = GF(col)
        # eval-consistency: eq_rows . Enc[:,q] == Enc(w)[q]
        if GF(np.sum(eq_rows * col)) != enc_w[q]:
            raise ValueError("PCS: eval-consistency check failed")
        # proximity: gamma . Enc[:,q] == Enc(gamma^T T)[q]
        if GF(np.sum(gamma * col)) != enc_rg[q]:
            raise ValueError("PCS: proximity check failed")

    return v
