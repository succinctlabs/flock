"""
field.py - the binary field F = GF(2^128) that Flock computes in.

The paper works over F2 (bits) for the *data* (witness, matrices) and lifts to
the big extension field F = F_{2^128} for the verifier's random challenges, which
is where soundness comes from (Schwartz-Zippel over a large field).

galois has no Conway polynomial for degree 128, so we hand it the same
irreducible polynomial the paper uses: the GHASH polynomial

    x^128 + x^7 + x^2 + x + 1.

Everything downstream imports GF from here so the whole codebase shares one field.
"""

import galois

# The GHASH reduction polynomial, exactly as in the Flock paper (section 4.3).
IRREDUCIBLE = "x^128 + x^7 + x^2 + x + 1"

# verify=False: skip galois's primitivity check. We only need a *field* (the poly
# is irreducible); we never rely on the chosen element being a multiplicative
# generator, so primitivity is irrelevant here.
GF = galois.GF(2**128, irreducible_poly=IRREDUCIBLE, verify=False)

# Shared field constants. CAUTION (classic galois trap): galois scalars are 0-d
# numpy arrays, so in-place operators mutate them. Never write `acc = ONE` and
# then `acc *= y` - that silently corrupts the global ONE for the whole program.
# Always start accumulators from a fresh GF(0)/GF(1) and use non-in-place ops
# (`acc = acc * y`). See sumcheck._lagrange_interpolate_eval for an example.
ZERO = GF(0)
ONE = GF(1)


def bit(b):
    """Lift a Python 0/1 into the field as the subfield element 0 or 1."""
    return ONE if b else ZERO
