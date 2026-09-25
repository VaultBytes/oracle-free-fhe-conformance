"""Negacyclic NTT over Z_q[X]/(X^N + 1), merged psi form.

Convention:
  * psi = primitive 2N-th root of unity mod q (psi^N == -1).
  * FORWARD: Cooley-Tukey decimation-in-time (Longa-Naehrig Alg.1). Input in
    NATURAL order, output in BIT-REVERSED order. Twiddles psi^{brv(k)}.
  * INVERSE: Gentleman-Sande decimation-in-frequency (Alg.2). Input BIT-REVERSED,
    output NATURAL, twiddles psi^{-brv(k)}, final scale by N^{-1}.
  * The psi twist is folded into the twiddles ("merged") -- no separate
    pre/post psi-multiply pass, matching the spec's NTT pipeline.

This pair is self-inverse (iNTT . NTT == identity) and equals the definitional
negacyclic transform A[j] = sum_i a_i * psi^{(2j+1) i} (in bit-reversed order),
both checked by tests/.
"""
from __future__ import annotations
from .modarith import modinv


def bit_reverse(x: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _log2(n: int) -> int:
    b = n.bit_length() - 1
    assert (1 << b) == n, "N must be a power of two"
    return b


def psi_powers_bitrev(psi: int, n: int, q: int) -> list[int]:
    """tab[k] = psi^{brv(k, log2 n)} mod q, k = 0..n-1."""
    logn = _log2(n)
    # straight powers first, then permute by bit-reversal
    pw = [1] * n
    for i in range(1, n):
        pw[i] = pw[i - 1] * psi % q
    return [pw[bit_reverse(k, logn)] for k in range(n)]


def ntt_forward(a: list[int], psi_rev: list[int], q: int) -> list[int]:
    """In-place-style CT forward NTT. Returns a NEW list (bit-reversed order)."""
    a = list(a)
    n = len(a)
    t = n
    m = 1
    while m < n:
        t >>= 1
        for i in range(m):
            j1 = 2 * i * t
            w = psi_rev[m + i]
            for j in range(j1, j1 + t):
                u = a[j]
                v = a[j + t] * w % q
                a[j] = (u + v) % q
                a[j + t] = (u - v) % q
        m <<= 1
    return a


def ntt_inverse(a: list[int], inv_psi_rev: list[int], n_inv: int, q: int) -> list[int]:
    """In-place-style GS inverse NTT. Returns a NEW list (natural order)."""
    a = list(a)
    n = len(a)
    t = 1
    m = n
    while m > 1:
        j1 = 0
        h = m >> 1
        for i in range(h):
            w = inv_psi_rev[h + i]
            for j in range(j1, j1 + t):
                u = a[j]
                v = a[j + t]
                a[j] = (u + v) % q
                a[j + t] = (u - v) * w % q
            j1 += 2 * t
        t <<= 1
        m = h
    return [x * n_inv % q for x in a]


def naive_negacyclic_forward(a: list[int], psi: int, q: int) -> list[int]:
    """Definitional oracle (O(N^2)), NATURAL-order output:
    A[j] = sum_i a_i * psi^{(2j+1) i} mod q.  For small-N verification only."""
    n = len(a)
    out = []
    for j in range(n):
        base = pow(psi, 2 * j + 1, q)
        acc = 0
        p = 1
        for i in range(n):
            acc = (acc + a[i] * p) % q
            p = p * base % q
        out.append(acc)
    return out
