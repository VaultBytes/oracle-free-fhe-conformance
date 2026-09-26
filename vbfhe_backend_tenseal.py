#!/usr/bin/env python3
"""TenSEAL backend, a third CKKS implementation for the conformance suite.

Why this exists. The suite previously scored two engines: OpenFHE, which we did not write, and
SoftwareCKKS, which we did. A reviewer observed that one independent implementation is thin
evidence, and that scoring our own reference with our own oracles is not independence in the strong
sense. TenSEAL wraps Microsoft SEAL, written by a different team again, so a third column tests the
suite against a second body of code we had no hand in.

It also exercises something the other two do not. TenSEAL rescales and relinearizes automatically,
so `rescale` here is a no-op while OpenFHE and SoftwareCKKS both require an explicit call. An
engine whose rescale policy differs from the reference is exactly the kind of legitimate variation a
conformance suite must not mistake for an error, and if the suite could only score engines that
manage levels the way ours does, it would be testing a convention rather than the scheme.

Coverage note, stated rather than hidden. TenSEAL 0.3.16 exposes no rotation on `CKKSVector`, so
this backend has no `rotate` method and the suite reports no keyswitch invariant for it. That is a
limit of the library's Python surface, not a fault in the engine, and the suite records absence as
absence rather than scoring it zero.
"""
from __future__ import annotations

import numpy as np
import tenseal as ts


class TenSEALBackend:
    """CKKS over Microsoft SEAL via TenSEAL, exposing the suite's backend protocol."""

    def __init__(self, ring_dim: int = 8192, scale_bits: int = 40, depth: int = 2,
                 first_mod_size: int = 60):
        self.N = int(ring_dim)
        self.slots = self.N // 2
        self.scale_bits = int(scale_bits)
        self.delta = 1 << self.scale_bits
        self.q = 1 << (self.scale_bits * depth + first_mod_size)

        chain = [first_mod_size] + [self.scale_bits] * depth + [first_mod_size]
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=self.N,
                              coeff_mod_bit_sizes=chain)
        self.ctx.global_scale = float(self.delta)
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

    # ---- encoding -------------------------------------------------------------------------
    def encode(self, vec, scale=None, qm=None):
        """Pad to the slot count, because TenSEAL vectors carry a fixed length.

        OpenFHE and SoftwareCKKS pad internally, so a caller may hand them a short vector and
        multiply the result by a slot-length plaintext. A TenSEAL CKKSVector keeps the length it
        was built with and refuses a mismatched operand, so the padding has to happen here or every
        caller has to know which engine it is talking to.
        """
        v = np.asarray(vec, dtype=float).ravel()[:self.slots]
        if v.size < self.slots:
            v = np.concatenate([v, np.zeros(self.slots - v.size)])
        return [float(x) for x in v]

    def encrypt(self, pt_vec):
        return ts.ckks_vector(self.ctx, list(pt_vec))

    def decrypt(self, ct):
        return np.asarray(ct.decrypt(), dtype=float)

    def decode_ct(self, ct, scale=None):
        return np.asarray(ct.decrypt(), dtype=float)

    # ---- evaluation -----------------------------------------------------------------------
    def add(self, x, y):
        return x + y

    def add_plain(self, ct, pt_vec):
        return ct + list(pt_vec)

    def mul_plain(self, ct, pt_vec):
        return ct * list(pt_vec)

    def mul(self, x, y):
        return x * y

    def rescale(self, ct, factor=None):
        """A no-op. TenSEAL rescales automatically after every multiplication.

        The suite calls rescale explicitly because OpenFHE and SoftwareCKKS both require it. An
        engine that manages its own level chain is still a correct CKKS engine, so the right
        response is to accept the call and do nothing rather than to fail or to rescale twice.
        """
        return ct

    # ---- identity -------------------------------------------------------------------------
    def serialize_ct(self, ct) -> bytes:
        return ct.serialize()

    def ct_equal(self, x, y) -> bool:
        return x.serialize() == y.serialize()

    def ct_modulus(self, ct) -> int:
        return int(self.q)

    def __repr__(self) -> str:
        return f"TenSEALBackend(N={self.N}, scale_bits={self.scale_bits})"
