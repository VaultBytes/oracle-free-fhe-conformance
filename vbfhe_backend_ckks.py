#!/usr/bin/env python3
"""
vbfhe SDK — software CKKS backend (real RLWE, runnable now, no board, no external FHE lib).

A compact but REAL leveled CKKS over Z_q[X]/(X^N+1): canonical-embedding encode/decode, RLWE
encrypt/decrypt, ciphertext add, plaintext multiply, ciphertext×ciphertext multiply with
relinearization, and rescale (a true modulus switch q -> q/Δ; each ciphertext carries its current
modulus). Power-of-two modulus chain q = Δ^L·q0 so Δ | q at every level. Negacyclic multiply via
exact big-int convolution. Enough for the LinearScoreCartridge: linear matvec via FEATURE-MAJOR
packing => NO rotations, + one degree-2 square. board = 0.

Honest scope: real lattice encryption (RLWE), real canonical embedding, real relin/rescale with
per-level modulus — but a power-of-two demo modulus (not a production prime/RNS chain) and pure-Python
big-int convolution (slow; OpenFHE/Lattigo are ~100-1000× faster, the chip more). It exists to prove
the ADOPTION LAYER + a bit-exact-within-precision cartridge on real encryption, not to be fast/secure.

Ciphertext = [c0, c1, qm] where qm is the current (per-level) modulus.
"""
import math
import numpy as np

rng = np.random.default_rng(20260615)


class SoftwareCKKS:
    def __init__(self, logN=8, scale_bits=22, depth=3, q0_bits=34, seed=20260615):
        # per-INSTANCE RNG: independent SoftwareCKKS instances must not share/correlate key material
        # via a module-global generator (also makes a session's key material reproducible from `seed`).
        self._rng = np.random.default_rng(seed)
        self.N = 1 << logN
        self.slots = self.N // 2
        self.scale_bits = scale_bits
        self.delta = 1 << scale_bits
        self.q = 1 << (scale_bits * depth + q0_bits)       # top modulus, power of two (Δ | q)
        M = 2 * self.N
        w = np.exp(2j * np.pi / M)
        jk = np.outer(2 * np.arange(self.N) + 1, np.arange(self.N))
        self.V = w ** jk
        self.Vh = self.V.conj().T
        self.s = self._ternary()
        self._relin_base = 1 << 20
        self._relin_digits = math.ceil(self.q.bit_length() / 20)
        self.rlk = self._gen_relin()

    # ---- sampling / mod ----
    def _ternary(self):
        return self._rng.integers(-1, 2, size=self.N, dtype=np.int64).astype(object)

    def _smallE(self, sigma=3.2):
        return np.round(self._rng.normal(0, sigma, size=self.N)).astype(np.int64).astype(object)

    @staticmethod
    def _mod(a, qm):
        m = qm - 1
        return np.array([int(v) & m for v in np.asarray(a, dtype=object)], dtype=object)

    @staticmethod
    def _center(a, qm):
        h = qm >> 1
        return np.array([(int(v) - qm if int(v) > h else int(v)) for v in np.asarray(a, dtype=object)], dtype=object)

    def _negmul(self, a, b, qm):
        lin = np.convolve(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
        res = np.zeros(self.N, dtype=object); N = self.N
        for k in range(len(lin)):
            if k < N: res[k] += int(lin[k])
            else:     res[k - N] -= int(lin[k])
        return self._mod(res, qm)

    # ---- encode / decode ----
    def encode(self, vec, scale=None, qm=None):
        scale = scale or self.delta; qm = qm or self.q
        v = np.zeros(self.N, dtype=complex); m = min(len(vec), self.slots)
        for j in range(m):
            v[j] = vec[j]; v[self.N - 1 - j] = vec[j]
        coeffs = (self.Vh @ v) / self.N
        return np.array([int(round(c.real * scale)) & (qm - 1) for c in coeffs], dtype=object)

    def decode(self, poly, scale, qm):
        c = self._center(poly, qm).astype(float)
        return np.real(self.V @ c)[:self.slots] / scale

    # ---- encryption ----
    def encrypt(self, pt_poly):
        a = self._mod([int(x) * 7919 for x in self._rng.integers(0, 1 << 60, size=self.N, dtype=np.int64).astype(object)], self.q)
        e = self._smallE()
        c0 = self._mod(-self._negmul(a, self.s, self.q) + np.asarray(pt_poly, dtype=object) + e, self.q)
        return [c0, a, self.q]

    def decrypt(self, ct):
        qm = ct[2]
        return self._mod(np.asarray(ct[0], dtype=object) + self._negmul(ct[1], self.s, qm), qm)

    def decode_ct(self, ct, scale):
        return self.decode(self.decrypt(ct), scale, ct[2])

    def add(self, x, y):
        qm = min(x[2], y[2])
        return [self._mod(x[0] + y[0], qm), self._mod(x[1] + y[1], qm), qm]

    def add_plain(self, ct, pt_poly):
        return [self._mod(ct[0] + np.asarray(pt_poly, dtype=object), ct[2]), ct[1], ct[2]]

    def mul_plain(self, ct, pt_poly):
        qm = ct[2]
        return [self._negmul(ct[0], pt_poly, qm), self._negmul(ct[1], pt_poly, qm), qm]

    # ---- ct x ct ----
    def _gen_relin(self):
        s2 = self._negmul(self.s, self.s, self.q); keys = []
        for i in range(self._relin_digits):
            a = self._mod([int(x) * 7919 for x in self._rng.integers(0, 1 << 60, size=self.N, dtype=np.int64).astype(object)], self.q)
            e = self._smallE()
            b = self._mod(-self._negmul(a, self.s, self.q) + e + s2 * (self._relin_base ** i), self.q)
            keys.append((b, a))
        return keys

    def _relinearize(self, d0, d1, d2, qm):
        c0, c1 = self._mod(d0, qm), self._mod(d1, qm)
        d2c = [int(v) for v in self._mod(d2, qm)]; T = self._relin_base
        for i in range(self._relin_digits):
            digit = np.array([(v // (T ** i)) % T for v in d2c], dtype=object)
            b, a = self.rlk[i]
            c0 = self._mod(c0 + self._negmul(digit, b, qm), qm)
            c1 = self._mod(c1 + self._negmul(digit, a, qm), qm)
        return [c0, c1, qm]

    def mul(self, x, y):
        qm = min(x[2], y[2])
        d0 = self._negmul(x[0], y[0], qm)
        d1 = self._mod(self._negmul(x[0], y[1], qm) + self._negmul(x[1], y[0], qm), qm)
        d2 = self._negmul(x[1], y[1], qm)
        return self._relinearize(d0, d1, d2, qm)

    def rescale(self, ct, factor=None):
        factor = factor or self.delta
        qm = ct[2]; qp = qm // factor
        def idiv(v):
            v = int(v)
            return (v + factor // 2) // factor if v >= 0 else -((-v + factor // 2) // factor)
        def rs(p):
            return np.array([idiv(v) & (qp - 1) for v in self._center(p, qm)], dtype=object)
        return [rs(ct[0]), rs(ct[1]), qp]


if __name__ == "__main__":
    ck = SoftwareCKKS(logN=8, scale_bits=22, depth=4, q0_bits=34)
    print(f"N={ck.N} slots={ck.slots} q~2^{ck.q.bit_length()} delta=2^{ck.scale_bits} digits={ck._relin_digits}")
    v = rng.normal(0, 2, size=ck.slots); u = rng.normal(0, 2, size=ck.slots); w = rng.normal(0, 1, size=ck.slots)
    D = ck.delta
    ct = ck.encrypt(ck.encode(v)); ctu = ck.encrypt(ck.encode(u))
    print(f"[encode/decode]   max err = {np.max(np.abs(ck.decode(ck.encode(v), D, ck.q) - v)):.2e}")
    print(f"[encrypt/decrypt] max err = {np.max(np.abs(ck.decode_ct(ct, D) - v)):.2e}")
    print(f"[ct+ct]           max err = {np.max(np.abs(ck.decode_ct(ck.add(ct, ctu), D) - (v+u))):.2e}")
    ctm = ck.rescale(ck.mul_plain(ct, ck.encode(w)))
    print(f"[ct*plain]        max err = {np.max(np.abs(ck.decode_ct(ctm, D) - (v*w))):.2e}")
    cts = ck.rescale(ck.mul(ct, ct))
    print(f"[ct*ct square]    max err = {np.max(np.abs(ck.decode_ct(cts, D) - (v*v))):.2e}")
    a2, a1, a0 = 0.5, -1.0, 2.0
    sq = ck.rescale(ck.mul(ct, ct))
    t2 = ck.rescale(ck.mul_plain(sq, ck.encode(np.full(ck.slots, a2), qm=sq[2])))
    t1 = ck.rescale(ck.mul_plain(ct, ck.encode(np.full(ck.slots, a1))))
    t1 = [t1[0] & 0, t1[1] & 0, t1[2]] if False else t1
    cal = ck.add_plain(ck.add(t2, t1), ck.encode(np.full(ck.slots, a0), qm=min(t2[2], t1[2])))
    ref = a2 * v * v + a1 * v + a0
    print(f"[degree-2 calib]  max err = {np.max(np.abs(ck.decode_ct(cal, D) - ref)):.2e}")
