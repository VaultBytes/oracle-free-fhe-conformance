"""CKKS canonical-embedding encoder / decoder -- FFT-based, deterministic.

WHY THIS MODULE EXISTS
----------------------
Everything else in `ckks_golden` operates on *coefficients*: the NTT, the
key-switch, the automorphism.  None of it can express the sentence "slot k of
the message".  Without a canonical-embedding encoder the package cannot state,
let alone test, any claim about *slots* -- which is why a packing claim could sit
unchallenged in a docstring: there was no apparatus that could contradict it.
This module is that apparatus.  It is a new file; it changes nothing.

THE EMBEDDING
-------------
The ring is R = Z[X]/(X^N + 1) (negacyclic, the ring `ntt.py` implements).  Its
roots are the primitive 2N-th roots of unity

    zeta = exp(i*pi/N),   zeta^t for t odd,  t in (Z/2N)^*

and the canonical embedding of m in R is the vector (m(zeta^t))_{t odd}.  m has
REAL coefficients, so m(zeta^{-t}) = conj(m(zeta^t)): the N evaluations come in
N/2 conjugate pairs and carry only N/2 independent complex numbers.  Those are
the slots.

WHICH t IS SLOT j
-----------------
(Z/2N)^* is not cyclic; for N a power of two it factors as

    (Z/2N)^* = {+1,-1} x <5>,      ord(5 mod 2N) = N/2,   -1 not in <5>

so <5> is a set of coset representatives, one per conjugate pair -- exactly one
representative per slot.  The convention taken here (HEAAN, OpenFHE, Lattigo) is

    slot j  <->  evaluation at zeta^(5^j mod 2N),   j = 0 .. N/2-1

5 is not magic: any base b with ord(b mod 2N) = N/2 and -1 not in <b> gives an
equally valid slot ordering, and SEAL in fact uses 3 (GaloisTool::generator_ = 3
in native/src/seal/util/galois.h) with its own matching order.
the rotation cross-check (not included) tabulates which bases qualify.

This ordering is not cosmetic.  The Galois automorphisms sigma_g : a(X) ->
a(X^g) compose by MULTIPLYING exponents, sigma_{g1} . sigma_{g2} = sigma_{g1 g2}.
Applying sigma_{5^k} to m evaluates at zeta^{5^k * 5^j} = zeta^{5^{j+k}}, so it
moves slot j+k to slot j: a cyclic rotation of the slot vector by k, and
rotations compose additively because 5^{k1} * 5^{k2} = 5^{k1+k2}.  Any
assignment k -> g(k) that is not exponential in k cannot realise rotation,
because rotation is additive in k and the group law on exponents is
multiplicative.  `the rotation cross-check (not included)` settles this experimentally rather than by
assertion.

SPARSE PACKING -- THE TILING RULE, PRECISELY
--------------------------------------------
A caller may ask for n_slots = s logical slots with s < N/2 (the pilot kit asks
for 16).  s must be a power of two and must divide N/2.  The logical vector z of
length s is TILED -- repeated, not block-expanded -- into the full N/2 slots:

    z_full[j] = z[j mod s],        j = 0 .. N/2-1          (REPEAT)
    NOT
    z_full[j] = z[j // ((N/2)/s)]  (block-expand -- WRONG, do not use)

and the full vector is then encoded normally.  `decode(..., n_slots=s)` returns
z_full[0:s].

The repeat rule is forced, not chosen.  Period-s repetition in j is exactly the
statement that the evaluation vector t -> m(zeta^t) is invariant under the
subgroup <5^s> of (Z/2N)^* (it is also invariant under the conjugation partner,
since z_full is repeated on both cosets).  A polynomial whose canonical
embedding is <5^s>-invariant is fixed by the automorphisms sigma_{5^s}, i.e. it
lies in the fixed subring

    Z[X^{N/(2s)}]/(X^N + 1)  ~=  Z[Y]/(Y^{2s} + 1),   Y = X^{N/(2s)}

so ONLY every (N/(2s))-th coefficient of the encoded polynomial is non-zero.
That structural consequence is checkable and is checked (the encode cross-check (not included), part
C): if the tiling rule were block-expansion the sparsity would not appear.  The
block-expanded vector is not <5^s>-invariant and encodes to a dense polynomial.

Rotation interacts with the tiling in the obvious way: rotating the full N/2-slot
vector by k rotates each tile, so on the logical length-s vector sigma_{5^k}
realises a rotation by (k mod s).

PRECISION AND DETERMINISM
-------------------------
The transform is a single length-2N complex FFT in each direction -- O(N log N),
one complex128 array of 2N entries (512 KiB at N=16384).  A dense Vandermonde
matrix would be O(N^2) and ~4 GiB at N=16384; `the encode cross-check (not included)` part A
cross-checks this FFT path against that dense definition at small N, where the
dense form is affordable, so the fast path is tied to the textbook one.

Rounding to integers is round-half-up, floor(x + 1/2), matching the rounding
already used elsewhere in the package ((2v + D)//(2D)).  numpy's FFT is
single-threaded pocketfft and bit-reproducible for a fixed build, so encode() is
deterministic on a given machine; it is NOT guaranteed bit-identical across
numpy builds or architectures, because it is floating point.  The integer output
is nevertheless stable unless a coefficient lands within ~1e-9 of a half
integer.  Treat encode() as a Class-B (approximate) reference, not a Class-A
bit-exact one like the NTT or the automorphism.

float64 carries 53 mantissa bits, so a coefficient magnitude above 2^53 cannot
be rounded meaningfully.  Both directions refuse (ValueError) rather than
silently returning noise.

USAGE
    from .keyswitch import Context
    from .encode import encode, decode
    ctx   = Context(N=4096, L=3, alpha=1)
    limbs = encode(ctx, [1+2j, 3-1j, ...], scale=2**40)   # L lists of N ints
    back  = decode(ctx, limbs, scale=2**40)
"""
from __future__ import annotations

import numpy as np

from .modarith import modinv

# Above this coefficient magnitude float64 cannot represent the integer part.
_MANTISSA_LIMIT = 1 << 53

_CRT_CACHE: dict[tuple[int, ...], tuple[int, list[int]]] = {}


# ---------------------------------------------------------------------------
# slot index set
# ---------------------------------------------------------------------------
def slot_indices(n: int) -> np.ndarray:
    """The CKKS slot index set for ring degree `n`: [5^j mod 2n] for
    j = 0 .. n/2-1.  Slot j is the evaluation of m at zeta^(5^j),
    zeta = exp(i*pi/n).  Length n/2, all entries odd and distinct."""
    _require_pow2(n, "N")
    two_n = 2 * n
    out = np.empty(n // 2, dtype=np.int64)
    x = 1
    for j in range(n // 2):
        out[j] = x
        x = (x * 5) % two_n
    return out


def _require_pow2(v: int, name: str) -> None:
    if v < 2 or (v & (v - 1)) != 0:
        raise ValueError(f"{name} must be a power of two >= 2, got {v}")


def _resolve(n: int, n_slots: int | None, indices) -> tuple[int, np.ndarray]:
    """Validate the slot count and produce the (possibly overridden) index set."""
    _require_pow2(n, "N")
    n_full = n // 2
    if n_slots is None:
        n_slots = n_full
    if n_slots < 1 or (n_slots & (n_slots - 1)) != 0:
        raise ValueError(f"n_slots must be a power of two >= 1, got {n_slots}")
    if n_slots > n_full or n_full % n_slots != 0:
        raise ValueError(
            f"n_slots={n_slots} must divide N/2={n_full} and not exceed it")
    idx = slot_indices(n) if indices is None else np.asarray(indices, dtype=np.int64)
    if idx.shape != (n_full,):
        raise ValueError(f"index set must have length N/2={n_full}, got {idx.shape}")
    return n_slots, idx


# ---------------------------------------------------------------------------
# coefficient-domain encode / decode (no RNS)
# ---------------------------------------------------------------------------
def encode_poly(n: int, slots, scale, n_slots: int | None = None,
                indices=None) -> list[int]:
    """Canonical-embedding encode into ONE integer polynomial of degree < n.

    `slots` is a length-n_slots sequence of complex (or real) values; it is
    tiled to N/2 slots by the REPEAT rule documented at module level.  Returns
    n CENTERED Python ints (may be negative): the coefficients of
    round(scale * sigma^{-1}(z_full)).

    `indices` exists ONLY so the control suite can corrupt the transform
    through the production code path (the encode cross-check (not included) part D).  Leave it None.
    """
    n_slots, idx = _resolve(n, n_slots, indices)
    two_n = 2 * n

    z = np.asarray(slots, dtype=np.complex128).ravel()
    if z.size != n_slots:
        raise ValueError(f"expected {n_slots} slot values, got {z.size}")
    z_full = np.tile(z, (n // 2) // n_slots) * float(scale)

    # C[t] = m(zeta^t): the chosen representative gets the slot value, its
    # conjugate partner 2N-t gets the conjugate.  That Hermitian symmetry is
    # what makes the coefficients real.
    c = np.zeros(two_n, dtype=np.complex128)
    c[idx % two_n] = z_full
    c[(two_n - idx) % two_n] = np.conj(z_full)

    # m_i = (1/N) * sum_{t} C[t] * zeta^{-t i}  ==  fft_{2N}(C)[i] / N,
    # because numpy's fft uses exp(-2*pi*i*k*n/M).  Entries N..2N-1 come out as
    # -m_{i-N} (that is X^N = -1) and are dropped.
    coeff = np.fft.fft(c)[:n].real / n

    peak = float(np.max(np.abs(coeff))) if n else 0.0
    if peak >= _MANTISSA_LIMIT:
        raise ValueError(
            f"coefficient magnitude {peak:.3e} >= 2^53: scale={scale} is too "
            f"large for the float64 transform; rounding would be meaningless")
    return np.floor(coeff + 0.5).astype(np.int64).tolist()


def decode_poly(n: int, coeffs, scale, n_slots: int | None = None,
                indices=None) -> list[complex]:
    """Canonical-embedding decode of a CENTERED integer coefficient vector.

    Returns the first n_slots slot values (the full N/2 when n_slots is None).
    Inverse of encode_poly up to the rounding the scale implies."""
    n_slots, idx = _resolve(n, n_slots, indices)
    two_n = 2 * n

    coeffs = list(coeffs)
    if len(coeffs) != n:
        raise ValueError(f"expected {n} coefficients, got {len(coeffs)}")
    if _needs_bigint(coeffs):
        raise ValueError(
            "coefficient magnitude >= 2^53: this polynomial cannot be decoded "
            "in float64; reduce the scale or the level")
    a = np.zeros(two_n, dtype=np.complex128)
    a[:n] = np.asarray(coeffs, dtype=np.float64)

    # fft(A)[t] = sum_i m_i zeta^{-t i} = m(zeta^{-t}),
    # so slot j = m(zeta^{5^j}) sits at index (-5^j) mod 2N.
    ev = np.fft.fft(a)
    vals = ev[(two_n - idx) % two_n] / float(scale)
    return vals[:n_slots].tolist()


def _needs_bigint(coeffs) -> bool:
    for c in coeffs:
        if c >= _MANTISSA_LIMIT or c <= -_MANTISSA_LIMIT:
            return True
    return False


# ---------------------------------------------------------------------------
# RNS limb plumbing (the layout make_inputs / ckks_mul_relin_check use:
# a list of L lists of N ints, limb i reduced mod ctx.q[i], coefficient domain)
# ---------------------------------------------------------------------------
def _crt_consts(q: list[int]) -> tuple[int, list[int]]:
    key = tuple(q)
    got = _CRT_CACHE.get(key)
    if got is None:
        big = 1
        for qi in q:
            big *= qi
        got = (big, [(big // qi) * modinv((big // qi) % qi, qi) for qi in q])
        _CRT_CACHE[key] = got
    return got


def limbs_from_poly(ctx, coeffs: list[int]) -> list[list[int]]:
    """Split a centered integer polynomial into ctx's L Q-limbs (coefficient
    domain), the layout keyswitch.make_inputs builds."""
    return [[c % qi for c in coeffs] for qi in ctx.q]


def poly_from_limbs(ctx, limbs: list[list[int]]) -> list[int]:
    """CRT-reconstruct the L Q-limbs and center into (-Q/2, Q/2]."""
    big, crt = _crt_consts(ctx.q)
    half = big // 2
    out = []
    for i in range(ctx.N):
        v = sum(limbs[l][i] * crt[l] for l in range(len(ctx.q))) % big
        out.append(v - big if v > half else v)
    return out


def encode(ctx, slots, scale, n_slots: int | None = None,
           indices=None) -> list[list[int]]:
    """Canonical-embedding encode into ctx's RNS limbs (coefficient domain).
    Returns L lists of N ints, limb i reduced mod ctx.q[i]."""
    return limbs_from_poly(ctx, encode_poly(ctx.N, slots, scale, n_slots, indices))


def decode(ctx, poly_or_limbs, scale, n_slots: int | None = None,
           indices=None) -> list[complex]:
    """Canonical-embedding decode.  Accepts either a centered integer
    coefficient polynomial (list[int], length N) or RNS limbs (list of L lists
    of N ints) -- the limb form is CRT-reconstructed and centered first."""
    if len(poly_or_limbs) and isinstance(poly_or_limbs[0], (list, tuple)):
        coeffs = poly_from_limbs(ctx, poly_or_limbs)
    else:
        coeffs = list(poly_or_limbs)
    return decode_poly(ctx.N, coeffs, scale, n_slots, indices)


if __name__ == "__main__":
    # Smoke self-check: round trip through the coefficient domain at N=64.
    import math
    N, D, S = 64, 1 << 40, 8
    z = [complex(0.5 + j / (2 * S), 0.25 - j / (3 * S)) for j in range(S)]
    p = encode_poly(N, z, D, n_slots=S)
    w = decode_poly(N, p, D, n_slots=S)
    err = max(abs(a - b) for a, b in zip(z, w))
    stride = N // (2 * S)
    dense = [i for i, c in enumerate(p) if c != 0 and i % stride != 0]
    print(f"encode.py self-check: N={N} slots={S} max err {err:.3e} "
          f"({-math.log2(err):.1f} bits), off-stride nonzeros {len(dense)}")
    assert err < 1e-8 and not dense, "encode.py self-check FAILED"
    print("encode.py self-check OK")
