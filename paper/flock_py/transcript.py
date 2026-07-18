"""
transcript.py - Fiat-Shamir transcript backed by SHA-256.

This is the *only* cryptographic assumption in the whole system (exactly as the
paper states: "the remaining soundness assumption is the security of SHA-256,
used internally for both Merkle commitments and Fiat-Shamir"). We hash the
running protocol transcript and squeeze out verifier challenges in F = GF(2^128),
turning the interactive protocol into a non-interactive one.
"""

import hashlib
from field import GF

FE_BYTES = 16  # 128 bits per field element


def fe_to_bytes(x):
    """Serialize a single GF(2^128) element to 16 big-endian bytes."""
    return int(x).to_bytes(FE_BYTES, "big")


class Transcript:
    """
    A running SHA-256 hash of every message exchanged so far.

    The prover ABSORBS each protocol message before squeezing the CHALLENGE that
    would have been the verifier's reply; the verifier replays the exact same
    absorb/challenge sequence, so both sides derive identical challenges iff they
    saw identical messages. Any tampering desynchronizes the challenges and some
    later check fails. Prover and verifier must therefore call these methods in
    EXACTLY the same order - compare prove() and verify() in flock.py line by line.
    """

    def __init__(self, label=b"flock"):
        # Domain-separation label: proofs for different protocols (or protocol
        # versions) hash differently even on identical messages.
        self.h = hashlib.sha256()
        self.h.update(label)

    def absorb_bytes(self, data: bytes):
        """Mix raw bytes (e.g. a Merkle root) into the transcript."""
        # length-prefixed so distinct messages can't be ambiguously concatenated
        # (absorb(b"ab"); absorb(b"c") must differ from absorb(b"a"); absorb(b"bc"))
        self.h.update(len(data).to_bytes(8, "big"))
        self.h.update(data)
        return self

    def absorb_fe(self, x):
        """Mix one GF(2^128) element into the transcript."""
        return self.absorb_bytes(fe_to_bytes(x))

    def absorb_fe_list(self, xs):
        """Mix a sequence of field elements, in order."""
        for x in xs:
            self.absorb_fe(x)
        return self

    def challenge(self):
        """Squeeze one field element; also re-key so the next challenge differs."""
        digest = self.h.digest()               # 32 bytes
        # fold the 256-bit digest to 128 bits (XOR halves); any 128-bit string is
        # a valid field element, so no modular bias to worry about
        val = int.from_bytes(digest[:FE_BYTES], "big") ^ int.from_bytes(
            digest[FE_BYTES:], "big"
        )
        self.h.update(b"squeeze")              # ratchet the state
        return GF(val % (2**128))

    def challenge_vec(self, n):
        """Squeeze n field elements as one galois array (e.g. a random point in F^n).

        Note the int() round-trip: GF([...]) wants integers, not 0-d galois
        scalars - another instance of the field-element-vs-integer distinction
        that galois makes you handle explicitly."""
        return GF([int(self.challenge()) for _ in range(n)])
