"""Modular arithmetic + deterministic NTT-prime / psi-root selection.

All arithmetic uses Python arbitrary-precision ints, so 50-60-bit primes and
their 120-bit products are exact by construction -- this is the bit-exact oracle
the RTL (60-bit Barrett modmul) must match. Nothing here depends on machine word
width. Determinism: every search scans candidates in a fixed order, no RNG.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Primality (Miller-Rabin). With this witness set {2,3,...,37} the test is
# DETERMINISTIC and exact for all n < 2^64, which covers every prime this model
# uses (<= 60-bit). Above 2^64 it is a strong-probable-prime test, NOT a proof:
# e.g. 318665857834031151167461 (> 2^64) is composite yet passes all 12 bases.
# ---------------------------------------------------------------------------
_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_WITNESSES:
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_ntt_primes(count: int, bits: int, two_n: int) -> list[int]:
    """`count` distinct primes q < 2^bits with q == 1 (mod two_n), scanning the
    largest such candidates downward. Deterministic given (count, bits, two_n)."""
    # largest k with k*two_n + 1 < 2^bits
    k = ((1 << bits) - 1 - 1) // two_n
    out: list[int] = []
    while len(out) < count and k > 0:
        q = k * two_n + 1
        if q.bit_length() == bits and is_prime(q):
            out.append(q)
        k -= 1
    if len(out) < count:
        raise ValueError(f"only found {len(out)}/{count} {bits}-bit primes == 1 mod {two_n}")
    return out


def modinv(a: int, q: int) -> int:
    return pow(a % q, q - 2, q)  # q prime (Fermat)


def modinv_general(a: int, m: int) -> int:
    """Inverse of a mod m for ANY modulus m (extended Euclid). Used where the
    modulus is a composite group product (gadget construction)."""
    a %= m
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError(f"no inverse: gcd({a},{m})={g}")
    return x % m


def _egcd(a: int, b: int):
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        qd = old_r // r
        old_r, r = r, old_r - qd * r
        old_s, s = s, old_s - qd * s
        old_t, t = t, old_t - qd * t
    return old_r, old_s, old_t


def find_psi(q: int, two_n: int) -> int:
    """Smallest primitive `two_n`-th root of unity mod q (order exactly two_n),
    scanning generator candidates g = 2,3,5,... deterministically.

    psi^N == q-1 (i.e. -1) guarantees the negacyclic structure for X^N + 1."""
    n = two_n // 2
    exp = (q - 1) // two_n
    g = 2
    while True:
        psi = pow(g, exp, q)
        if pow(psi, n, q) == q - 1:  # order is exactly 2N => psi^N == -1
            # also confirm psi^(2N) == 1 and no smaller order dividing 2N via the
            # N-th-power test above (sufficient: psi^N == -1 forces order | 2N and
            # order does not divide N, so order == 2N).
            return psi
        g += 1
        while not is_prime(g):
            g += 1
