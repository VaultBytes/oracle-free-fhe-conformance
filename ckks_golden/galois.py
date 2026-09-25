"""Standalone Galois automorphism (rotation) reference -- bit-exact, deterministic.

OFC primitive P3. Previously the automorphism was only validated *inside* keyswitch
(the rotation that follows it); the conformance standard wants it as an isolated,
certifiable op. This is that oracle.

Convention: the ring is Z_q[X]/(X^N + 1) (negacyclic, matching ntt.py). The Galois
group is (Z/2N)^* acting by sigma_g : a(X) -> a(X^g), g odd and coprime to 2N. On
coefficients, a_i X^i maps to a_i X^{(g*i) mod 2N}, reduced in X^N+1 (so X^N = -1):
the target slot is e = (g*i) mod 2N; if e < N it lands at +a_i in slot e, else at
-a_i in slot e-N. g is invertible mod 2N so this is a signed permutation of slots --
an EXACT integer op (Class A): two conformant implementations must agree bit-for-bit.

WHICH g ROTATES SLOTS BY k
--------------------------
A slot rotation by k is sigma_{b^k mod 2N} -- the k-th POWER of a fixed base b, not an
affine function of k. b must be admissible:

    ord(b mod 2N) == N/2        (its powers reach every one of the N/2 slot rotations)
    -1 not in <b>               (equivalently 2N-1 not in <b>: the subgroup must miss
                                 conjugation, so that <b> and {+-1} split (Z/2N)^*
                                 and each rotation has ONE exponent, unmixed with conj)

b = 5 is admissible for every N = 2^m with m >= 2 and is what HEAAN, OpenFHE and
Lattigo use; SEAL uses b = 3. This repo uses 5 throughout -- see
the RTL staged-golden generator (not included) (GEN = 5, rotGroup_L[j] = 5^j mod 4L).

    rotation by k slots  ==  sigma_g  with  g = pow(b, k, 2*N)          (b = 5 here)
    conjugation          ==  sigma_g  with  g = 2N-1

g = 2*k+1 is NOT a rotation by k. It is an arbitrary odd unit whose discrete log base 5
has no relation to k, and for half of all k it additionally carries conjugation. This
docstring previously asserted the opposite; the claim is false and was propagated into
the BSGS rotation-key exporter. Measured at N=256/1024/4096/16384 (25/25 compositions
wrong), e.g. at N=1024: g=3 is conj . rot(+163), not rot(+1); g=5 is rot(+1), not
rot(+2); g=9 is rot(+326), not rot(+4). The rotation cross-check is not included in this distribution
from the repository root for the full adjudication table.

The op implemented below is independent of packing -- it is a pure coefficient
permutation, which is exactly what makes it cleanly certifiable. The packing only
enters when you ask WHICH g to pass, which is what the paragraph above answers.
"""
from __future__ import annotations
from .modarith import modinv_general  # extended-Euclid inverse (2N is composite)


def automorphism_coeff(a: list[int], g: int, q: int) -> list[int]:
    """sigma_g applied to coefficient-domain poly `a` (length N), mod q.
    Returns a NEW length-N list. `g` must be odd (coprime to 2N)."""
    n = len(a)
    two_n = 2 * n
    assert g % 2 == 1, "Galois exponent g must be odd (a unit mod 2N)"
    out = [0] * n
    for i in range(n):
        e = (g * i) % two_n
        if e < n:
            out[e] = (out[e] + a[i]) % q
        else:
            out[e - n] = (out[e - n] - a[i]) % q
    return out


def automorphism_index_map(n: int, g: int) -> list[tuple[int, int]]:
    """Returns the signed permutation as a list `m` of length N where coefficient i
    goes to slot m[i][0] with sign m[i][1] in {+1,-1}. Pure structure (q-independent)
    -- lets an RTL/silicon implementation precompute the wiring and a verifier check
    the permutation without arithmetic."""
    two_n = 2 * n
    m = []
    for i in range(n):
        e = (g * i) % two_n
        m.append((e, 1) if e < n else (e - n, -1))
    return m


def inverse_g(g: int, n: int) -> int:
    """The exponent realizing sigma_g^{-1}: g_inv with g*g_inv == 1 (mod 2N).
    2N is composite -> extended Euclid (Fermat's little theorem does not apply)."""
    return modinv_general(g % (2 * n), 2 * n)


if __name__ == "__main__":
    # Self-check: sigma_g is a bijection and sigma_{g^-1} . sigma_g == id.
    q = (1 << 60) - 569  # any modulus
    N = 16
    a = [(7 * i + 3) % q for i in range(N)]
    for g in (3, 5, 2 * N - 1):
        b = automorphism_coeff(a, g, q)
        ginv = inverse_g(g, N)
        a_back = automorphism_coeff(b, ginv, q)
        assert a_back == a, f"sigma_{g} not invertible by sigma_{ginv}"
    print("galois.py self-check OK (automorphism is an exact invertible signed perm)")
