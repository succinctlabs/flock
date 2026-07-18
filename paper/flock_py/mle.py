"""
mle.py - multilinear extensions and the eq polynomial.

A function f : {0,1}^m -> F is stored simply as its table of 2^m values.
Index convention (little-endian): the value at boolean point b = (b_0,...,b_{m-1})
lives at integer index  i = sum_v b_v * 2^v,  i.e. variable v is bit (i >> v) & 1.

The multilinear extension  f_hat(x) = sum_b f(b) * eq(b, x)  is what the whole
protocol runs on: it turns a table of bits into a polynomial the verifier can
probe at a random field point r in F^m.
"""

import numpy as np
from field import GF, ONE


def num_vars(table):
    """m such that len(table) == 2^m."""
    n = len(table)
    m = n.bit_length() - 1
    assert 1 << m == n, "table length must be a power of two"
    return m


def eq_vector(point):
    """
    Return the length-2^m vector [ eq(b, point) for b in {0,1}^m ], little-endian.

    Built as a tensor product, one variable at a time:
        eq(b, x) = prod_v ( x_v if b_v==1 else (1 - x_v) )
    This is exactly the multilinear 'eq' from the paper - a product of the tiny
    1-D Lagrange bases {(1-x_v), x_v} on the node set {0,1}, one per coordinate.
    """
    w = GF([1])
    for xv in point:                       # append variable v as the new top bit
        w = np.concatenate([w * (ONE - xv), w * xv])
    return w


def eval_mle(table, point):
    """
    Evaluate the multilinear extension of `table` at `point` in F^m.
    Uses  f_hat(point) = <table, eq_vector(point)>  (O(2^m), maximally clear).
    """
    assert len(point) == num_vars(table)
    if len(table) == 1:
        return table[0]
    return GF(np.sum(GF(table) * eq_vector(point)))


def bind_first(table, x):
    """
    Bind variable 0 (the least-significant bit) to the field value x, returning a
    table over the remaining m-1 variables (half the length).

    Pairs (2j, 2j+1) share their higher bits and differ only in bit 0, so this is
    a 1-D linear interpolation between the b_0=0 and b_0=1 slices:
        out[j] = table[2j] * (1 - x) + table[2j+1] * x
    This is the single fold step used in every sumcheck round.
    """
    t = GF(table)
    even = t[0::2]        # b_0 = 0 slice
    odd = t[1::2]         # b_0 = 1 slice
    return even * (ONE - x) + odd * x
