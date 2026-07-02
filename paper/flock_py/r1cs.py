"""
r1cs.py - the batched circuit Flock proves.

Base circuit F (one AND gate):   out = x AND y  ==  x * y  over F2.
Per-instance witness layout (length 2^m0 = 4, so m0 = 2), little-endian slots:

    z0 = [ const1 , x , y , out ]
    idx     0       1   2    3

The single real R1CS row says  x * y = out :
    A0 selects x (idx 1), B0 selects y (idx 2), C0 selects out (idx 3).
Rows 1..3 are padding (0 = 0) so A0,B0,C0 are square 4x4, as the paper requires.

Batching (section 4.1): K = 2^k instances stacked block-diagonally,
    A = I_K (x) A0,  so  (A z) applies A0 to each instance independently.
Global index convention (little-endian): variables 0..m0-1 are the WITHIN-instance
bits i_in; variables m0..m-1 are the instance index i_out. So
    global_index = i_out * 2^m0 + i_in,   and  z.reshape(K, 2^m0)[i_out] = instance i_out.

Glue circuit G (a hash-chain): the instances form a chain
    x_{i+1} = out_i = x_i AND y_i,
enforced by the paper's shift argument (section 4.6). Slots used:
    out lives at within-instance index 3 -> boolean bits S_OUT = (1,1)
    x   lives at within-instance index 1 -> boolean bits S_IN  = (1,0)
"""

import numpy as np
from field import GF

M0 = 2                      # base variables; base dimension 2^M0 = 4
BASE = 1 << M0             # = 4

# within-instance slot bit-patterns (little-endian, length M0)
S_CONST = (0, 0)          # idx 0
S_IN = (1, 0)             # idx 1 (x, the chained input)
S_AUX = (0, 1)            # idx 2 (y)
S_OUT = (1, 1)            # idx 3 (out, the chained output)


def _selector_row(idx):
    """Unit row vector: as an R1CS matrix row it 'selects' witness slot `idx`."""
    row = np.zeros(BASE, dtype=np.uint8)
    row[idx] = 1
    return row


# Base matrices: only row 0 is a real constraint (x*y=out); rows 1..3 are zero.
A0 = np.zeros((BASE, BASE), dtype=np.uint8); A0[0] = _selector_row(1)  # picks x
B0 = np.zeros((BASE, BASE), dtype=np.uint8); B0[0] = _selector_row(2)  # picks y
C0 = np.zeros((BASE, BASE), dtype=np.uint8); C0[0] = _selector_row(3)  # picks out


def slot_index(slot_bits):
    """within-instance integer index of a slot given its little-endian bits."""
    return sum(b << v for v, b in enumerate(slot_bits))


def gen_witness(K, x0, ys):
    """
    Build a satisfying batched witness for a hash-chain of K AND-gates.
      x0  : initial input bit (0/1)
      ys  : list of K aux bits (0/1), one per instance
    Returns (z_bits as np.uint8 length K*BASE, info dict).
    Chain: x_0 = x0 ; out_i = x_i AND y_i ; x_{i+1} = out_i.
    """
    assert len(ys) == K
    z = np.zeros(K * BASE, dtype=np.uint8)
    x = x0 & 1
    xs, outs = [], []
    for i in range(K):
        y = ys[i] & 1
        out = x & y
        base = i * BASE
        z[base + 0] = 1        # const 1
        z[base + 1] = x        # input x_i
        z[base + 2] = y        # aux y_i
        z[base + 3] = out      # output out_i
        xs.append(x); outs.append(out)
        x = out                # chain
    info = {"xs": xs, "ys": [y & 1 for y in ys], "outs": outs, "x_final": x}
    return z, info


def apply_base(M0mat, z, K):
    """Compute (I_K (x) M0mat) @ z  by applying the small M0mat to each instance."""
    Z = GF(z.astype(np.int64)).reshape(K, BASE)          # (K, BASE)
    M = GF(M0mat.astype(np.int64))                       # (BASE, BASE)
    out = (M @ Z.T).T                                    # (K, BASE)
    return out.reshape(K * BASE)


def transpose_apply_base(M0mat, vec, K):
    """Compute (I_K (x) M0mat)^T @ vec = (I_K (x) M0mat^T) @ vec, blockwise."""
    V = GF(vec).reshape(K, BASE)
    MT = GF(M0mat.astype(np.int64)).T
    out = (MT @ V.T).T
    return out.reshape(K * BASE)


def check_r1cs(z, K):
    """Directly verify (Az) o (Bz) = Cz over F2 for the batched witness."""
    a = apply_base(A0, z, K)
    b = apply_base(B0, z, K)
    c = apply_base(C0, z, K)
    return bool(np.all(a * b == c))


# ------------------------------- glue G ----------------------------------------
# Hash-chain relation out_i = x_{i+1} for i = 0..K-2, as a sparse linear map G with
# Gz = 0. This is the concrete instantiation of the paper's section 4.6 glue; Flock
# proves it with a tailored shift argument, we prove Gz=0 with the same lincheck
# sumcheck used for A,B,C (a "generic IO via slot-aligned regions", section 4.6).

def out_index(i):
    return i * BASE + slot_index(S_OUT)   # within-instance idx 3


def in_index(i):
    return i * BASE + slot_index(S_IN)    # within-instance idx 1


def glue_matrix(K):
    """(K*BASE x K*BASE) matrix G with row i enforcing out_i - x_{i+1} = 0 (F2)."""
    N = K * BASE
    G = GF(np.zeros((N, N), dtype=np.int64))
    for i in range(K - 1):
        G[i, out_index(i)] = 1            # out_i
        G[i, in_index(i + 1)] = 1         # x_{i+1}   (+ == - in F2)
    return G


def check_glue(z, K):
    """Directly verify the hash-chain: Gz = 0 over F2."""
    g = glue_matrix(K) @ GF(z.astype(np.int64))
    return bool(np.all(np.array(g) == 0))
