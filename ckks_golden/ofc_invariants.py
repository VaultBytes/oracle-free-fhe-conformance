"""Oracle-free conformance -- certify FHE hardware by the SCHEME'S OWN ALGEBRAIC LAWS,
with NO trusted golden reference. (An original OFC contribution, not an adaptation.)

Every existing conformance approach (NIST CAVP, golden-vector suites, our own ofc_verify)
certifies a device by comparing its output to a TRUSTED REFERENCE ("match these bytes").
That has a fatal adoption flaw: a vendor can object "why must my chip match *your* chip?"
-- it puts our reference, not the math, at the center.

This module flips it: a device is conformant iff it satisfies the INTRINSIC ALGEBRAIC
INVARIANTS that hold for ANY correct FHE implementation, on random inputs -- invariants
the device's OWN outputs must satisfy *internally*. The oracle is FHE algebra itself:

  NTT is Z_q-LINEAR:           ntt(a+b) == ntt(a) + ntt(b)  ;  ntt(c.a) == c.ntt(a)
  NTT round-trips:             intt(ntt(a)) == a
  NTT diagonalizes convolution: intt(ntt(f).ntt(g)) == f (x) g   (negacyclic)
  automorphism is a HOMOMORPHISM: sigma_g(a+b) == sigma_g(a) + sigma_g(b)
  automorphisms form a GROUP:   sigma_{g.h} == sigma_g . sigma_h ; sigma_{g^-1} . sigma_g == id
  automorphism ROTATES SLOTS:   decode(rot_k(m))[j] == decode(m)[j+k]
  sigma_{2N-1} CONJUGATES SLOTS: decode(sigma_{2N-1}(m))[j] == conj(decode(m)[j])
  keyswitch IDENTITY:           out_b + out_a.s1 == d.s2  (mod Q)
  modmul is a COMMUTATIVE RING: comm, assoc, distributive

WHAT THE INVARIANTS DO **NOT** DO (read this before quoting a score)
-------------------------------------------------------------------
An invariant suite certifies the laws it states, not "correctness". Two failures of this
module were found and fixed on 2026-09-20; both are recorded here because the shape of
the mistake recurs:

  1. SILENCE READ AS SUCCESS. `check_keyswitch_identity` used to call the module-level
     `hybrid_keyswitch(dev.ctx, ...)` -- it took the device's *context* and never its
     *answers*, and there was no device seam for keyswitch at all. A device that returns
     all-zeros from every operation, and has no keyswitch method whatsoever, scored 5/8
     INCLUDING "keyswitch identity". A missing operation is now NOT TESTED, never PASS,
     and NOT TESTED is not conformant. See `Verdict`.

  2. ORDER-BLIND AUTOMORPHISM CHECKS. Homomorphism, group law and inverse hold for ANY
     multiplicative endomorphism of the exponent, so a device that rotates the WRONG WAY
     (g -> g^-1) certified clean. None of the three mentions a slot, so none can
     adjudicate a claim about slots -- which is how `galois.py` came to document
     "g = 2k+1 rotates by k" (false) without a single test contradicting it. Neither a
     round trip nor the homomorphism can adjudicate it either: encoding AND decoding with
     the wrong index set {2j+1} round-trips at 2.4e-11, because any N/2 distinct odd
     residues give an invertible transform, and evaluation at ANY point set is a ring
     homomorphism. ONLY rotation adjudicates slot order. `check_slot_rotation` is that
     invariant, and it depends on the ordering.

  3. HALF THE GALOIS GROUP UNTESTED. Found while enumerating what each invariant still
     admits -- by argument, then confirmed by building the device, not by a failing test.
     (Z/2N)^* = {+1,-1} x <b>, and the rotation invariant only ever exercises <b>. Since
     g -> |g| is a group homomorphism onto <b>, a device that silently drops the -1 part
     satisfies homomorphism, group law, inverse AND rotation: measured 10/10 CONFORMANT
     while disagreeing with the honest device on half of all exponents. Closed by
     `check_slot_conjugation`; the device is kept as DropsConjugationDevice.

Invariants that STILL admit a concrete wrong implementation are enumerated in
`KNOWN_HOLES` below, each with a control device that demonstrates the hole by passing.
An invariant that no control can fail is not an invariant.

THE DEVICE PROTOCOL
-------------------
Previously implicit (the suite simply reached for `dev.ctx`, `dev.q`, `dev.two_n`,
`dev.ntt`, `dev.intt`, `dev.modmul`, `dev.auto`). Stated explicitly so a vendor knows
what to implement and the suite knows what is missing rather than guessing:

  ATTRIBUTES (required)
    dev.ctx          keyswitch.Context -- the public parameters (N, L, alpha, primes)
    dev.q            the single modulus the poly ops below work in (conventionally ctx.q[0])
    dev.two_n        2 * ctx.N

  OPERATIONS (required)
    dev.ntt(a)       -> list[int]        forward NTT mod dev.q, length N
    dev.intt(e)      -> list[int]        inverse NTT mod dev.q, length N
    dev.modmul(a, b) -> list[int]        pointwise product mod dev.q, length N
    dev.auto(a, g)   -> list[int]        sigma_g on coefficient-domain a, mod dev.q

  OPERATIONS (optional; ABSENT => that invariant reports NOT TESTED, never PASS)
    dev.keyswitch(d_ntt, evk_a, evk_b) -> (out_a, out_b)
                     one hybrid key-switch, L limbs each, NTT domain. The suite verifies
                     out_b + out_a.s1 == d.s2 (mod Q) ON THE DEVICE'S OWN out_a/out_b.
    dev.rotate(a, k) -> list[int]        slot rotation by k on coefficient-domain a.
                     A device that does not override it is held to the profile rule
                     rot_k == sigma_{ROT_BASE^k mod 2N}, applied through its own dev.auto.
                     Override it to declare a different derivation of the exponent -- that
                     derivation is then what `check_slot_rotation` adjudicates, which is
                     exactly where the 2k+1 bug lived (the host derived the exponent; the
                     chip applied sigma_g faithfully; both agreed and both were wrong).

Run:  python3 -m ckks_golden.ofc_invariants      (from runtime/vbfhe/)
"""
from __future__ import annotations

from .encode import decode_poly, encode_poly
from .galois import automorphism_coeff, inverse_g
from .keyswitch import (Context, _det_poly, hybrid_keyswitch, make_inputs,
                        verify_keyswitch)
from .modarith import find_psi, modinv
from .ntt import ntt_forward, ntt_inverse, psi_powers_bitrev

# The packing base this profile certifies against. It is NOT a free choice: it must be
# admissible (ord(b mod 2N) == N/2 and -1 not in <b>) AND it must be the base the slot
# ordering in encode.slot_indices() is built from, or "slot j" means two different things
# on the two sides of the check. encode.py fixes 5 (HEAAN / OpenFHE / Lattigo; SEAL uses
# 3 with its own matching order). check_slot_rotation asserts admissibility rather than
# taking it on faith.
ROT_BASE = 5
ROT_SCALE = 1 << 40      # encode scale for the slot probe
ROT_TOL = 1e-6           # a correct rotation lands at ~1.5e-12; a wrong one at ~1e0


# --- verdicts: PASS / FAIL / NOT TESTED -- silence must not read as success ----------
class Verdict:
    """Three-valued result. `bool(v)` is True ONLY for PASS, so a caller that reduces
    with `all()` or `bool()` fails closed on NOT TESTED -- an untested invariant can
    never be counted as a passing one."""
    __slots__ = ("code", "detail")

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail

    def __bool__(self) -> bool:
        return self.code == "PASS"

    def __eq__(self, other) -> bool:
        if isinstance(other, Verdict):
            return self.code == other.code
        if isinstance(other, str):
            return self.code == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.code)

    def __repr__(self) -> str:
        return self.code + (f" ({self.detail})" if self.detail else "")

    @property
    def tested(self) -> bool:
        return self.code != "NOT TESTED"


PASS = Verdict("PASS")
FAIL = Verdict("FAIL")
NOT_TESTED = Verdict("NOT TESTED")


def _fail(detail: str) -> Verdict:
    return Verdict("FAIL", detail)


def _untested(detail: str) -> Verdict:
    return Verdict("NOT TESTED", detail)


def _v(ok: bool) -> Verdict:
    return PASS if ok else FAIL


# --- device-under-test: each method is the device's own implementation of an op -------
class Device:
    """A correct device. A vendor subclasses and replaces these with their hardware calls.
    The invariants below are checked ONLY against the device's own outputs -- no reference.
    See THE DEVICE PROTOCOL in the module docstring for what must be provided."""
    name = "honest"

    def __init__(self, ctx):
        self.ctx = ctx
        q = ctx.q[0]
        self.q = q
        self.two_n = ctx.two_n
        self.psi = find_psi(q, self.two_n)
        self.psi_rev = psi_powers_bitrev(self.psi, ctx.N, q)
        self.ipsi_rev = psi_powers_bitrev(modinv(self.psi, q), ctx.N, q)
        self.ninv = modinv(ctx.N, q)

    def ntt(self, a):  return ntt_forward(a, self.psi_rev, self.q)
    def intt(self, e): return ntt_inverse(e, self.ipsi_rev, self.ninv, self.q)
    def auto(self, a, g): return automorphism_coeff(a, g, self.q)
    def modmul(self, a, b): return [(x * y) % self.q for x, y in zip(a, b)]

    def rotate(self, a, k):
        """This device's slot rotation by k. The profile rule: rot_k == sigma_{b^k mod 2N}
        for the admissible base b the packing is built from."""
        return self.auto(a, pow(ROT_BASE, k, self.two_n))

    def keyswitch(self, d_ntt, evk_a, evk_b):
        """One hybrid key-switch. Returns (out_a, out_b), L limbs each, NTT domain."""
        out_a, out_b, _ = hybrid_keyswitch(self.ctx, d_ntt, evk_a, evk_b)
        return out_a, out_b


# --- the invariant checks: each returns a Verdict on the device's OWN outputs ---------
def _addv(a, b, q): return [(x + y) % q for x, y in zip(a, b)]


def _odd_exponents(seeds: int, two_n: int, count: int) -> list[int]:
    """`count` odd residues mod 2N drawn from the round seed. 2N is a power of two, so
    every odd residue is a unit -- the full Galois group, not a hardcoded corner of it."""
    half = two_n // 2
    return [2 * r + 1 for r in _det_poly(seeds ^ 0x5EED, count, half)]


def check_ntt_linear(dev, seeds):
    a = _det_poly(seeds, dev.ctx.N, dev.q)
    b = _det_poly(seeds + 1, dev.ctx.N, dev.q)
    lhs = dev.ntt(_addv(a, b, dev.q))
    rhs = _addv(dev.ntt(a), dev.ntt(b), dev.q)
    return _v(lhs == rhs)


def check_ntt_roundtrip(dev, seeds):
    a = _det_poly(seeds, dev.ctx.N, dev.q)
    return _v(dev.intt(dev.ntt(a)) == a)


def check_ntt_convolution(dev, seeds):
    # intt(ntt(f).ntt(g)) == negacyclic product f (x) g  (no reference NTT used -- device's own)
    N, q = dev.ctx.N, dev.q
    f = _det_poly(seeds, N, q)
    g = _det_poly(seeds + 2, N, q)
    dev_prod = dev.intt(dev.modmul(dev.ntt(f), dev.ntt(g)))
    # negacyclic schoolbook reference computed from f,g directly (algebra, not a golden table)
    naive = [0] * N
    for i in range(N):
        for j in range(N):
            k = i + j
            if k < N:
                naive[k] = (naive[k] + f[i] * g[j]) % q
            else:
                naive[k - N] = (naive[k - N] - f[i] * g[j]) % q
    return _v(dev_prod == naive)


def check_auto_homomorphism(dev, seeds, g=5):
    """sigma_g(a+b) == sigma_g(a) + sigma_g(b), for the pinned g AND for exponents drawn
    from the round seed. The pinned g is kept so no previously-caught device escapes."""
    a = _det_poly(seeds, dev.ctx.N, dev.q)
    b = _det_poly(seeds + 3, dev.ctx.N, dev.q)
    ab = _addv(a, b, dev.q)
    for gg in [g] + _odd_exponents(seeds, dev.two_n, 3):
        if dev.auto(ab, gg) != _addv(dev.auto(a, gg), dev.auto(b, gg), dev.q):
            return _fail(f"g={gg}")
    return PASS


def check_auto_group_law(dev, seeds, g1=3, g2=5):
    """sigma_{g1} . sigma_{g2} == sigma_{g1.g2 mod 2N}, for the pinned pair AND for pairs
    drawn from the round seed."""
    a = _det_poly(seeds, dev.ctx.N, dev.q)
    ex = _odd_exponents(seeds, dev.two_n, 4)
    pairs = [(g1, g2), (ex[0], ex[1]), (ex[2], ex[3])]
    for h1, h2 in pairs:
        if dev.auto(dev.auto(a, h2), h1) != dev.auto(a, (h1 * h2) % dev.two_n):
            return _fail(f"g1={h1} g2={h2}")
    return PASS


def check_auto_inverse(dev, seeds, g=5):
    """sigma_{g^-1} . sigma_g == id, for the pinned g AND for exponents from the seed."""
    a = _det_poly(seeds, dev.ctx.N, dev.q)
    for gg in [g] + _odd_exponents(seeds, dev.two_n, 3):
        if dev.auto(dev.auto(a, gg), inverse_g(gg, dev.ctx.N)) != a:
            return _fail(f"g={gg}")
    return PASS


def check_modmul_distributive(dev, seeds):
    q, N = dev.q, dev.ctx.N
    a = _det_poly(seeds, N, q); b = _det_poly(seeds + 4, N, q); c = _det_poly(seeds + 5, N, q)
    lhs = dev.modmul(a, _addv(b, c, q))
    rhs = _addv(dev.modmul(a, b), dev.modmul(a, c), q)
    return _v(lhs == rhs)


# --- the slot-order invariant: the only one that can adjudicate a packing claim -------
def _rot_base_admissible(n: int, b: int) -> bool:
    """ord(b mod 2N) == N/2 and -1 not in <b>: the two conditions that make <b> a set of
    coset representatives, one per slot, with each rotation carrying ONE exponent."""
    two_n, x, seen = 2 * n, 1, set()
    for _ in range(n // 2):
        x = x * b % two_n
        seen.add(x)
    return len(seen) == n // 2 and (two_n - 1) not in seen


def _slot_vector(n_slots: int, seeds: int) -> list[complex]:
    """Slot values with pairwise-distinct magnitudes AND pairwise-distinct arguments,
    none real and none the conjugate of another, so that a permutation and a conjugation
    are both identifiable from the decoded vector alone (the same construction
    rotation_check.py uses). The phase offset is stirred by the round seed."""
    import math
    off = 0.3 + (seeds % 977) / 977.0
    out = []
    for j in range(n_slots):
        r = 0.5 + 0.5 * j / n_slots
        t = off + 1.7 * j / n_slots
        out.append(complex(r * math.cos(t), r * math.sin(t)))
    return out


def _device_rotate(dev, a, k):
    """The device's own slot rotation by k. A device may expose it directly (dev.rotate);
    otherwise it is held to the profile rule rot_k == sigma_{ROT_BASE^k}, applied through
    the device's own automorphism. Returns None when the device implements neither."""
    rot = getattr(dev, "rotate", None)
    if rot is not None:
        return rot(a, k)
    auto = getattr(dev, "auto", None)
    if auto is None:
        return None
    return auto(a, pow(ROT_BASE, k, dev.two_n))


def _decoded_image(dev, produce, N, q, label):
    """Run one device-produced coefficient vector through the decoder. Returns
    (slot values, None) or (None, Verdict) -- output the decoder cannot represent is a
    FAILED invariant, not a harness crash: a signed permutation of a plaintext cannot
    grow a coefficient, so junk output is itself the violation."""
    try:
        out = produce()
    except Exception as exc:
        return None, _fail(f"{label}: raised {type(exc).__name__}: {exc}")
    if out is None:
        return None, _untested("device implements neither rotate() nor auto()")
    if len(out) != N:
        return None, _fail(f"{label}: returned {len(out)} coefficients, expected {N}")
    centered = [v - q if v > q // 2 else v for v in out]
    try:
        return decode_poly(N, centered, ROT_SCALE), None
    except (ValueError, TypeError) as exc:
        return None, _fail(f"{label}: output is not decodable ({exc})")


def check_slot_rotation(dev, seeds):
    """THE ORDER-SENSITIVE INVARIANT. Encode a known slot vector, apply the DEVICE's
    rotation by k, decode, and require slot j to hold what slot j+k held.

    Why the other automorphism invariants cannot replace this: homomorphism, group law and
    inverse are satisfied by every multiplicative endomorphism of the exponent, and a round
    trip is satisfied by every invertible transform -- including one built on the wrong slot
    index set. This check is the only one whose outcome changes when the slot ORDER changes.

    The shifts tested are k=1 and k=n_slots-1 (fixed, so coverage of the two generators
    does not depend on the draw) plus two drawn from the round seed. All exclude the two
    degenerate values k=0 and k=n_slots/2, where rot_k == rot_{-k} and an exponent-inverting
    device would pass by coincidence rather than by being right.
    """
    N, q = dev.ctx.N, dev.q
    n_slots = N // 2
    if n_slots < 4:
        return _untested(f"N={N} too small for a non-degenerate rotation")
    if not _rot_base_admissible(N, ROT_BASE):
        return _fail(f"profile base {ROT_BASE} is not admissible at N={N}")

    # k with 2k !== 0 (mod n_slots): excludes k=0 and k=n_slots/2, the two shifts that are
    # their own inverse and so cannot tell rot(+k) from rot(-k).
    usable = [k for k in range(1, n_slots) if (2 * k) % n_slots != 0]
    drawn = [usable[i] for i in _det_poly(seeds ^ 0x1207, 2, len(usable))]
    shifts = sorted({1, n_slots - 1, *drawn} & set(usable))

    z = _slot_vector(n_slots, seeds)
    coeffs = encode_poly(N, z, ROT_SCALE)
    if max(abs(c) for c in coeffs) >= q // 2:
        return _untested("scale too large for this modulus")
    reduced = [c % q for c in coeffs]

    for k in shifts:
        got, bad = _decoded_image(dev, lambda: _device_rotate(dev, list(reduced), k),
                                  N, q, f"k={k}")
        if bad is not None:
            return bad
        err = max(abs(got[j] - z[(j + k) % n_slots]) for j in range(n_slots))
        if err > ROT_TOL:
            return _fail(f"k={k}: slot j did not receive slot j+k (max err {err:.2e})")
    return PASS


def check_slot_conjugation(dev, seeds):
    """sigma_{2N-1} must CONJUGATE every slot: decode(sigma_{2N-1}(m))[j] == conj(z[j]),
    because m has real coefficients and m(zeta^-t) == conj(m(zeta^t)).

    Why this is a separate law and not a corner of the one above: (Z/2N)^* splits as
    {+1,-1} x <b>, and `check_slot_rotation` only ever exercises <b> -- it says nothing
    about the other coset, which is HALF of every exponent the device will be asked for.
    The projection g -> |g| onto <b> is itself a group homomorphism, so a device that
    silently DROPS the -1 part (applying sigma_{+b^a} where sigma_{-b^a} was asked)
    satisfies homomorphism, group law, inverse AND the rotation invariant. Measured at
    N=16: such a device certified 9/9 CONFORMANT while disagreeing with the honest one at
    g=3 and g=2N-1. See DropsConjugationDevice."""
    N, q = dev.ctx.N, dev.q
    n_slots = N // 2
    auto = getattr(dev, "auto", None)
    if auto is None:
        return _untested("device has no auto()")
    z = _slot_vector(n_slots, seeds)
    coeffs = encode_poly(N, z, ROT_SCALE)
    if max(abs(c) for c in coeffs) >= q // 2:
        return _untested("scale too large for this modulus")
    reduced = [c % q for c in coeffs]
    got, bad = _decoded_image(dev, lambda: auto(reduced, dev.two_n - 1), N, q, "conjugation")
    if bad is not None:
        return bad
    err = max(abs(got[j] - z[j].conjugate()) for j in range(n_slots))
    if err > ROT_TOL:
        return _fail(f"sigma_(2N-1) did not conjugate the slots (max err {err:.2e})")
    return PASS


def check_keyswitch_identity(dev, seeds):
    """The keyswitch identity out_b + out_a.s1 == d.s2 (mod Q) ON THE DEVICE'S OWN OUTPUTS,
    up to the INHERENT hybrid-ModDown rounding residual (an intrinsic bound ~O(N.alpha),
    NOT a golden tolerance). A correct device lands well within it; a wrong keyswitch breaks
    the identity by ~Q (2^59 per limb here), so the invariant still catches arithmetic faults.

    A device with no keyswitch seam is NOT TESTED. It used to be PASS: the check called the
    module-level hybrid_keyswitch(dev.ctx, ...), taking the device's *context* and never its
    *answers*, so an all-zeros device with no keyswitch method certified on this line."""
    ks = getattr(dev, "keyswitch", None)
    if ks is None:
        return _untested("device has no keyswitch()")
    inp = make_inputs(dev.ctx, seed=seeds, e_bound=0)
    try:
        got = ks(inp["d_ntt"], inp["evk_a"], inp["evk_b"])
    except Exception as exc:                       # a device that throws is not conformant
        return _fail(f"keyswitch raised {type(exc).__name__}: {exc}")
    if got is None:
        return _fail("keyswitch returned None")
    try:
        out_a, out_b = got[0], got[1]
        err = verify_keyswitch(dev.ctx, inp, out_a, out_b)
    except Exception as exc:
        return _fail(f"keyswitch output unusable: {type(exc).__name__}: {exc}")
    bound = dev.ctx.N * dev.ctx.alpha   # inherent ModDown rounding bound (intrinsic to the scheme)
    if err <= bound:
        return PASS
    return _fail(f"identity residual {err} > bound {bound}")


INVARIANTS = [
    ("NTT linearity", check_ntt_linear),
    ("NTT round-trip", check_ntt_roundtrip),
    ("NTT diagonalizes convolution", check_ntt_convolution),
    ("automorphism homomorphism", check_auto_homomorphism),
    ("automorphism group law", check_auto_group_law),
    ("automorphism inverse", check_auto_inverse),
    ("automorphism rotates slots", check_slot_rotation),
    ("automorphism conjugates slots", check_slot_conjugation),
    ("modmul distributivity", check_modmul_distributive),
    ("keyswitch identity", check_keyswitch_identity),
]


def certify(device, rounds=8):
    """Certify a device purely by algebraic invariants on `rounds` deterministic inputs
    each. Returns {invariant: Verdict}. NO golden reference is consulted anywhere.

    Aggregation over rounds: any FAIL is FAIL; otherwise any NOT TESTED is NOT TESTED;
    otherwise PASS. `bool(Verdict)` is True only for PASS, so `all(res.values())` is a
    fail-closed conformance test for callers that keep using it."""
    res = {}
    for name, fn in INVARIANTS:
        agg = PASS
        for r in range(rounds):
            v = fn(device, 0x1000 * r + 7)
            if not isinstance(v, Verdict):
                v = _v(bool(v))
            if v.code == "FAIL":
                agg = v
                break
            if v.code == "NOT TESTED" and agg.code == "PASS":
                agg = v
        res[name] = agg
    return res


def verdict(res) -> str:
    """CONFORMANT only when every invariant was tested and passed."""
    if any(v.code == "FAIL" for v in res.values()):
        return "CAUGHT"
    if any(v.code == "NOT TESTED" for v in res.values()):
        return "INCOMPLETE"      # not conformant: an untested law is not a satisfied law
    return "CONFORMANT"


# =====================================================================================
# CONTROL DEVICES. Each one is a concrete wrong implementation; the suite must reject it.
# An invariant that no control can fail is not an invariant.
# =====================================================================================
class BrokenLinearityDevice(Device):
    """A device whose NTT matches a golden on a FIXED input (passes naive golden test) but
    adds a constant to one slot -- VIOLATING linearity. Golden-matching on one vector would
    not catch this; the algebraic invariant does, from the device's own outputs alone."""
    name = "broken:nonlinear-ntt"

    def ntt(self, a):
        out = super().ntt(a)
        out[1] = (out[1] + 7) % self.q   # a fixed offset: breaks ntt(a+b)=ntt(a)+ntt(b)
        return out


class BrokenGroupDevice(Device):
    """Automorphism correct for single g but wrong composition (a real rotation-key bug
    class). Per-op golden on g=5 passes; the GROUP-LAW invariant catches it."""
    name = "broken:automorphism-group"

    def auto(self, a, g):
        out = super().auto(a, g)
        if g == 15:                       # sabotage only the composed generator
            out[0] = (out[0] + 1) % self.q
        return out


class ZeroDevice:
    """Every operation returns all-zeros, and there is NO keyswitch method at all. This is
    the device that exposed the bug: before 2026-09-20 it scored 5/8 including "keyswitch
    identity". It is deliberately NOT a Device subclass -- it is exactly the object a third
    party would hand the suite, which is the point: the suite must not fill in for it."""
    name = "zero:no-keyswitch"

    def __init__(self, ctx):
        self.ctx = ctx
        self.q = ctx.q[0]
        self.two_n = ctx.two_n

    def ntt(self, a, *r, **k):       return [0] * len(a)
    def intt(self, a, *r, **k):      return [0] * len(a)
    def modmul(self, a, b, *r, **k): return [0] * len(a)
    def auto(self, a, g, *r, **k):   return [0] * len(a)


class ZeroKeyswitchDevice(ZeroDevice):
    """The same all-zeros device, but it DOES answer keyswitch -- with zeros. This is the
    control for the keyswitch invariant proper: NOT TESTED above is an absence of evidence,
    a zero answer is evidence of a fault, and the two must be reported differently."""
    name = "zero:zero-keyswitch"

    def keyswitch(self, d_ntt, evk_a, evk_b):
        z = [[0] * self.ctx.N for _ in range(self.ctx.L)]
        return z, [list(x) for x in z]


class InvertedExponentDevice(Device):
    """"Rotates the wrong way": applies sigma_{g^-1} where sigma_g was asked. g -> g^-1 is
    a multiplicative endomorphism of (Z/2N)^*, so homomorphism, group law AND inverse are
    all satisfied -- this device certified clean before the slot-rotation invariant existed."""
    name = "broken:inverted-exponent"

    def auto(self, a, g):
        return super().auto(a, inverse_g(g, self.ctx.N))


class AffineRotationDevice(Device):
    """The bug that shipped. The automorphism engine is correct; the EXPONENT DERIVATION is
    wrong -- the host derived g = 2k+1 for a rotation by k (galois.py's docstring said so).
    Chip and host agreed with each other all the way onto silicon, and both were wrong,
    because agreement between two sides of one derivation says nothing about whether it
    rotates anything. Only a check that decodes slots can tell."""
    name = "broken:affine-2k+1"

    def rotate(self, a, k):
        return self.auto(a, (2 * k + 1) % self.two_n)


class PerturbedKeyswitchDevice(Device):
    """A keyswitch that is correct except for +1 on one coefficient of one limb. The output
    is byte-indistinguishable from correct on 6143 of 6144 values; the identity residual
    blows up to ~Q/q_0 after centered CRT reconstruction."""
    name = "broken:keyswitch-1limb"

    def keyswitch(self, d_ntt, evk_a, evk_b):
        out_a, out_b = super().keyswitch(d_ntt, evk_a, evk_b)
        out_b[0][0] = (out_b[0][0] + 1) % self.ctx.q[0]
        return out_a, out_b


class DropsConjugationDevice(Device):
    """Applies sigma_{+b^a} where sigma_{-b^a} was asked -- the conjugation half of the
    Galois group silently discarded. It disagrees with the honest device on HALF of all
    exponents (g=3 and g=2N-1 among them) and still certified 9/9 CONFORMANT until
    `automorphism conjugates slots` existed, because g -> |g| is a group homomorphism onto
    <b>: homomorphism, group law, inverse and the rotation invariant are all blind to it.
    Found by enumerating what each invariant still admits, not by a failing test."""
    name = "broken:drops-conjugation"

    def auto(self, a, g):
        rot = {pow(ROT_BASE, i, self.two_n) for i in range(self.ctx.N // 2)}
        if g % self.two_n not in rot:          # (Z/2N)^* = {+1,-1} x <b>
            g = (-g) % self.two_n
        return super().auto(a, g)


class JunkRotationDevice(Device):
    """An automorphism that returns full-width junk instead of a signed permutation. The
    control for the harness itself: such output is outside the plaintext range the decoder
    can represent, and that must be reported as a FAILED invariant, not raised as a crash
    that takes the whole certification run down with it."""
    name = "broken:rotation-junk"

    def auto(self, a, g):
        out = super().auto(a, g)
        return [(v + 0x0123456789ABCDEF) % self.q for v in out]


class NoKeyswitchDevice(Device):
    """Honest on every op it implements, but implements no keyswitch. Its verdict must be
    INCOMPLETE, not CONFORMANT: a law nobody tested is not a law the device satisfies."""
    name = "honest:no-keyswitch"
    keyswitch = None       # the protocol member is absent, not broken


CONTROLS = [
    BrokenLinearityDevice, BrokenGroupDevice, ZeroDevice, ZeroKeyswitchDevice,
    InvertedExponentDevice, AffineRotationDevice, DropsConjugationDevice,
    JunkRotationDevice,
    PerturbedKeyswitchDevice, NoKeyswitchDevice,
]


# =====================================================================================
# KNOWN HOLES. Wrong implementations that the invariants above STILL certify.
# These are asserted to PASS, so the list cannot rot into a comfortable fiction: if an
# invariant is ever strengthened enough to catch one, this module fails and the entry
# must be moved to CONTROLS.
# =====================================================================================
class PermutedNTTDevice(Device):
    """WRONG BUT CERTIFIED. Its NTT evaluates at the right points in the WRONG ORDER (here:
    reversed), and its iNTT undoes the same permutation. Linearity, round-trip AND the
    convolution law all hold, because pointwise multiplication commutes with any permutation
    of the evaluation points -- so the three NTT invariants together do not pin the output
    ORDER. This is the classic interop fault (natural vs bit-reversed vs any other order):
    two such devices agree with the algebra and disagree with each other, byte for byte.
    Closing it needs a check that names a point, e.g. ntt(a)[i] == a(psi^{2i+1})."""
    name = "hole:permuted-ntt-order"

    def _perm(self, v):
        return list(reversed(v))

    def ntt(self, a):  return self._perm(super().ntt(a))
    def intt(self, e): return super().intt(self._perm(e))


class ScaledModmulDevice(Device):
    """WRONG BUT CERTIFIED BY ITS OWN INVARIANT. modmul(a,b) = 2ab is distributive (and
    commutative, and associative), so `modmul distributivity` alone cannot see it. In this
    suite it is caught only because `NTT diagonalizes convolution` also routes through
    modmul -- i.e. the ring invariant is carried by a different check than the one named
    after it. Listed so the credit is assigned to the check that actually does the work."""
    name = "hole:modmul-scaled-2x"

    def modmul(self, a, b):
        return [(2 * x * y) % self.q for x, y in zip(a, b)]


class SyzygyKeyswitchDevice(Device):
    """WRONG BUT CERTIFIED. It adds g_1.(a_0,b_0) - g_0.(a_1,b_1) to the correct output,
    where g_l are the gadget idempotents. Because g_0.g_1 == 0 (mod Q) the perturbation
    lies exactly in the kernel of the identity, so out_b + out_a.s1 == d.s2 holds EXACTLY
    while out_a is a completely different mask.

    What this proves: the identity at e_bound=0 certifies DECRYPTABILITY, not the output.
    A device may return any point of the eval key's syzygy lattice. With a noisy eval key
    (e_bound > 0, as every real one is) the same perturbation multiplies the key noise by
    ~Q and the ciphertext is destroyed -- which the check never looks at, because it pins
    e_bound=0. Closing it needs a NOISE-GROWTH invariant: run at e_bound > 0 and require
    the residual to stay inside the scheme's own noise bound. main() measures both."""
    name = "hole:keyswitch-syzygy"

    def keyswitch(self, d_ntt, evk_a, evk_b):
        out_a, out_b = super().keyswitch(d_ntt, evk_a, evk_b)
        ctx = self.ctx
        if ctx.dnum < 2:
            return out_a, out_b
        g0 = ctx.gadget_modall(0)
        g1 = ctx.gadget_modall(1)
        for i in range(ctx.L):
            qi = ctx.q[i]
            c0, c1 = g1[i], g0[i]
            for n in range(ctx.N):
                out_a[i][n] = (out_a[i][n] + c0 * evk_a[0][i][n] - c1 * evk_a[1][i][n]) % qi
                out_b[i][n] = (out_b[i][n] + c0 * evk_b[0][i][n] - c1 * evk_b[1][i][n]) % qi
        return out_a, out_b


KNOWN_HOLES = [PermutedNTTDevice, SyzygyKeyswitchDevice]

# Wrong implementations that pass the invariant NAMED for them, but are caught elsewhere in
# the suite. Kept separate: they are not holes in the suite, they are misattributed credit.
MISATTRIBUTED = [(ScaledModmulDevice, "modmul distributivity",
                  "NTT diagonalizes convolution")]


# =====================================================================================
def _row(name, res, names, width=13):
    cell = {"PASS": "ok", "FAIL": "FAIL", "NOT TESTED": "n.t."}
    return f"{name:<28}" + "".join(f"{cell[res[n].code]:<{width}}" for n in names)


def main():
    ctx = Context(N=16, L=6, alpha=3, prime_bits=60)
    names = [n for n, _ in INVARIANTS]
    print("=== Oracle-free conformance: certify by FHE's own algebra (no golden reference) ===\n")
    print(f"{'device':<28}" + "".join(f"{n[:11]:<13}" for n in names) + "VERDICT")

    honest = certify(Device(ctx))
    print(_row("honest", honest, names) + verdict(honest))
    ok = verdict(honest) == "CONFORMANT"
    if not ok:
        print("  !! the honest device did not certify:",
              {k: repr(v) for k, v in honest.items() if v.code != "PASS"})

    print("\n--- controls: every one of these is a concrete wrong device and must be rejected ---")
    for cls in CONTROLS:
        r = certify(cls(ctx))
        vd = verdict(r)
        print(_row(cls.name, r, names) + vd)
        if vd == "CONFORMANT":
            ok = False
            print(f"  !! {cls.name} CERTIFIED -- this control no longer fails anything")

    print("\n--- known holes: wrong devices the invariants above STILL certify (see KNOWN_HOLES) ---")
    for cls in KNOWN_HOLES:
        r = certify(cls(ctx))
        vd = verdict(r)
        print(_row(cls.name, r, names) + vd)
        if vd != "CONFORMANT":
            ok = False
            print(f"  !! {cls.name} is now CAUGHT -- move it from KNOWN_HOLES to CONTROLS")

    for cls, named, caught_by in MISATTRIBUTED:
        r = certify(cls(ctx))
        print(_row(cls.name, r, names) + verdict(r))
        if not (r[named].code == "PASS" and r[caught_by].code == "FAIL"):
            ok = False
            print(f"  !! {cls.name}: expected {named}=PASS caught by {caught_by}")

    # The syzygy hole, measured: exact at e_bound=0, destroyed at e_bound>0. This is the
    # number that says what the keyswitch identity does and does not certify.
    inp0 = make_inputs(ctx, seed=7, e_bound=0)
    inp1 = make_inputs(ctx, seed=7, e_bound=1)
    hon, syz = Device(ctx), SyzygyKeyswitchDevice(ctx)
    line = []
    for tag, inp in (("e=0", inp0), ("e=1", inp1)):
        for dv in (hon, syz):
            a, b = dv.keyswitch(inp["d_ntt"], inp["evk_a"], inp["evk_b"])
            line.append(f"{tag}/{dv.name.split(':')[0]:<6} {verify_keyswitch(ctx, inp, a, b):>26}")
    print("\nkeyswitch identity residual (bound = N*alpha = "
          f"{ctx.N * ctx.alpha}):")
    for s in line:
        print("   ", s)

    print("\nNo reference bytes were consulted -- conformance = satisfying FHE's algebraic laws.")
    print("NOT TESTED is not CONFORMANT: an operation the device does not implement is")
    print("reported as untested, never as a pass. Silence does not read as success.")
    print(f"\nself-test (honest conformant; every control caught; every known hole still open): {ok}")
    assert ok, "oracle-free conformance self-test failed"
    print("ofc_invariants self-test OK.")


if __name__ == "__main__":
    main()
