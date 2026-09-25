#!/usr/bin/env python3
"""
vbfhe SDK — REAL CKKS backend over OpenFHE.

Drop-in replacement for SoftwareCKKS that runs the SDK (and therefore `certify()`) against a
production FHE library — OpenFHE's CKKS — instead of the pure-Python teaching backend. Same op
protocol (encode/encrypt/decrypt/decode_ct/add/add_plain/mul_plain/mul/rescale + N/slots/delta/
scale_bits/q), so the Session and the conformance kernel run unchanged.

Two extra hooks let the conformance layer handle OpenFHE's opaque ciphertext (not [c0,c1,qm]):
  serialize_ct(ct) -> bytes   (for result-binding digests)
  ct_equal(x, y)   -> bool     (for the blind add-commutativity invariant)

Uses FIXEDMANUAL scaling so the SDK's explicit `rescale(mul(...))` calls behave as written, and
HEStd_NotSet so small demo ring dimensions are allowed (mirrors the private attention harness (not included):174-191).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

import openfhe as ofhe


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


class OpenFHEBackend:
    """Real CKKS backend. `certify()` against this is a production-library conformance run."""

    def __init__(self, max_batch: int = 8, scale_bits: int = 40, depth: int = 4,
                 ring_dim: int | None = None, first_mod_size: int = 60):
        self.N = int(ring_dim or max(1024, _next_pow2(2 * max_batch)))
        self.slots = self.N // 2
        self.scale_bits = int(scale_bits)
        self.delta = 1 << self.scale_bits
        # representative modulus size (for the backend descriptor / derive_floors uses N+scale only)
        self.q = 1 << (self.scale_bits * depth + first_mod_size)

        p = ofhe.CCParamsCKKSRNS()
        p.SetSecurityLevel(ofhe.HEStd_NotSet)
        p.SetRingDim(self.N)
        p.SetMultiplicativeDepth(depth)
        p.SetScalingModSize(self.scale_bits)
        p.SetFirstModSize(first_mod_size)
        p.SetScalingTechnique(ofhe.FIXEDMANUAL)
        p.SetBatchSize(self.slots)
        cc = ofhe.GenCryptoContext(p)
        for f in (ofhe.PKE, ofhe.LEVELEDSHE, ofhe.ADVANCEDSHE):
            cc.Enable(f)
        kp = cc.KeyGen()
        cc.EvalMultKeyGen(kp.secretKey)
        self.rot_steps = [1, 2, 3, 4, 5, 6, 7, 8]         # rotation (keyswitch) keys the CA may challenge
        cc.EvalRotateKeyGen(kp.secretKey, self.rot_steps)
        self.cc, self._pk, self._sk = cc, kp.publicKey, kp.secretKey

    # ---- encode: pass the raw values through; OpenFHE encodes at plaintext-make time ----
    def encode(self, vec, scale=None, qm=None):
        return np.asarray(vec, dtype=float)

    def _pt(self, vec):
        return self.cc.MakeCKKSPackedPlaintext([float(v) for v in np.atleast_1d(vec)])

    # ---- encryption ----
    def encrypt(self, pt_vec):
        return self.cc.Encrypt(self._pk, self._pt(pt_vec))

    def decrypt(self, ct):
        d = self.cc.Decrypt(self._sk, ct)
        d.SetLength(self.slots)
        return np.array(d.GetRealPackedValue()[:self.slots])

    def decode_ct(self, ct, scale=None):
        return self.decrypt(ct)

    # ---- homomorphic ops (op names + call order mirror SoftwareCKKS) ----
    def add(self, x, y):
        return self.cc.EvalAdd(x, y)

    def add_plain(self, ct, pt_vec):
        return self.cc.EvalAdd(ct, self._pt(pt_vec))

    def mul_plain(self, ct, pt_vec):
        return self.cc.EvalMult(ct, self._pt(pt_vec))

    def mul(self, x, y):
        return self.cc.EvalMult(x, y)          # OpenFHE relinearizes internally

    def rescale(self, ct, factor=None):
        return self.cc.Rescale(ct)

    def ct_modulus(self, raw):
        """OpenFHE tracks levels internally; plaintext ops auto-align, so no explicit modulus is
        needed (the software backend returns its per-level modulus here)."""
        return None

    def poly_eval(self, raw, coeffs):
        """Native polynomial evaluation p(x)=Σ c_k x^k on a ciphertext (OpenFHE EvalPoly handles the
        level management). Lets the SDK's poly_eval run a real degree-2 activation on real CKKS."""
        return self.cc.EvalPoly(raw, [float(c) for c in coeffs])

    def rotate(self, raw, k):
        """Rotate slots left by k — a KEYSWITCH operation (uses a rotation/Galois key). EvalRotate(k)
        maps slot i -> slot (i+k), i.e. the plaintext is np.roll(vec, -k). Exposing this lets the
        attested profile certify the keyswitch pipeline (the primitive every FHE accelerator advertises)."""
        return self.cc.EvalRotate(raw, int(k))

    # ---- conformance hooks for the opaque OpenFHE ciphertext ----
    def serialize_ct(self, ct) -> bytes:
        fd, fn = tempfile.mkstemp(suffix=".ct")     # mkstemp (not mktemp) — no TOCTOU race
        os.close(fd)
        try:
            ofhe.SerializeToFile(fn, ct, ofhe.BINARY)
            with open(fn, "rb") as fh:
                return fh.read()
        finally:
            if os.path.exists(fn):
                os.remove(fn)

    def ct_equal(self, x, y) -> bool:
        return self.serialize_ct(x) == self.serialize_ct(y)


class DegradedOpenFHEBackend(OpenFHEBackend):
    """A real OpenFHE backend with a NON-distributive multiply bug (adds a small constant offset in
    ct×ct). Linear ops stay correct; `ctmul_distributive` must catch it — a realistic 'engine computes
    the wrong thing' fault, on a real library."""
    def mul(self, x, y):
        prod = self.cc.EvalMult(x, y)
        return self.cc.EvalAdd(prod, self._pt(np.full(self.slots, 0.05)))
