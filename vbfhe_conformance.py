#!/usr/bin/env python3
"""
vbfhe SDK — ORACLE-FREE CONFORMANCE layer  ·  the differentiating client verb.

Certify that the engine which produced an encrypted result computed it correctly — WITHOUT decrypting
the user's data, and WITHOUT trusting a golden reference. The oracle is FHE algebra itself
(ckks_golden/ofc_invariants.py, the patented method; PCT filed).

Certificate binds four things, all signed:
  * VERDICT from oracle-free algebraic invariants run on the session's backend (service-owned probes)
  * DERIVED precision floors from CKKS noise theory (vendor-neutral — you clear the math's bound)
  * ANTI-SANDBAG workload-representative probe profile (random-only probes can be gamed)
  * RESULT BINDING — a digest of the exact output ciphertexts, so a cert can't be reused for another result
  * (optional) PRIMITIVE-LAYER attestation of the RNS op family (NTT/automorphism/keyswitch laws)

OPEN vs CLOSED:
  OPEN   (client): certify() verb, Certificate, ConformanceService interface, LocalOracleFreeService.
  CLOSED (moat)  : RemoteConformanceService submits a request (descriptor + trace + seeds — NO data)
                   to the paid VaultBytes service running the full patented judging engine. It also
                   ingests a FHETCH/OpenFHE trace, so you can "certify any FHE build, incl. Niobium".
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

OFC_METHOD = "oracle-free-algebraic-invariants/v0.2"
CERT_VERSION = "vbfhe-conformance-cert/0.3"   # 0.3 renames `ceilings` and adds the upper band
CONFORMANCE_TOLERANCE_BITS = 6.0     # a conformant device must land within this many bits of the
                                     # noise-theory ceiling for each op (the "conformance slack")
# The known-answer (attested) mode compares the device output to the CA's TRUE plaintext, so the score
# also carries the reference's own encode/decode error — inherently ~2-3 bits below the self-consistency
# score. This margin is a property of the measurement (encode+decode of the reference), NOT per-device
# tuning. A rigorous [worst,avg] acceptance band replaces both tolerances in the v0.2 spec (see SECURITY.md).
ATTESTED_KNOWN_ANSWER_MARGIN = 3.0

# ------------------------------------------------------------------------------------------------
# #2 Derived, vendor-neutral precision floors (from CKKS noise theory, not hand-set constants)
# ------------------------------------------------------------------------------------------------
def _import_noise_model():
    """noise_model lives in the sibling ckks_golden package; add runtime/vbfhe to the path."""
    here = os.path.dirname(os.path.abspath(__file__))
    vbfhe = os.path.abspath(os.path.join(here, ".."))
    if vbfhe not in sys.path:
        sys.path.append(vbfhe)                    # APPEND, never insert(0, ...)
    from ckks_golden import noise_model            # type: ignore
    return noise_model


def derive_floors(be=None, tolerance: float = CONFORMANCE_TOLERANCE_BITS, *, N=None, scale_bits=None):
    """Precision floor (bits) each op must meet or beat, DERIVED from CKKS noise theory for the given
    (N, scale_bits). floor = worst-case theoretical ceiling − tolerance. Pass a backend `be` OR the
    params directly (the conformance server derives floors from the request descriptor, not a backend).
    The second element used to be called `ceilings`, which is what it was written into every
    certificate as. It is not a ceiling. `mode="worst"` is the worst-case GUARANTEED precision of a
    correct implementation, so it is a lower bound a correct engine must meet or beat, and a correct
    engine exceeding it is the expected case rather than a suspicious one. Reading it as a maximum
    is how a delegate reporting 47.9 bits came to look like it was 19.9 bits above what the same
    certificate called physically carriable. The upper bound is `attainable_absolute_bits`.

    Returns (floors_dict, worst_case_bits_dict, source)."""
    if be is not None:
        N, sb = int(be.N), int(be.scale_bits)
    else:
        N, sb = int(N), int(scale_bits)
    try:
        nm = _import_noise_model()
        ceil = {
            "add_homomorphism":      nm.encode_bits(N, sb, mode="worst"),
            "plainmul_distributive": nm.rescale_bits(N, sb, levels=1, mode="worst"),
            "ctmul_distributive":    nm.arithmetic_ceiling_bits(N, sb, depth=1, mode="worst"),
            # rotation is a keyswitch on a fresh (level-0) ciphertext — a linear op, ~encode precision
            "keyswitch_rotation":    nm.encode_bits(N, sb, mode="worst"),
        }
        source = "ckks_golden.noise_model(worst)-tol"
    except Exception as e:
        # REFUSE. This used to fall back to a scale_bits-derived ceiling, which was LAXER than the
        # noise-theoretic one (6.0/5.0/5.0 instead of 6.0/6.0/6.0 at N=256, scale 22). A conformance
        # authority that cannot derive its own acceptance threshold has nothing to certify against,
        # and silently substituting a different threshold means the certificate's floor_source field
        # describes a computation that did not happen. An absent dependency must refuse, not degrade.
        raise RuntimeError(
            f"cannot derive precision floors: the ckks_golden noise model is unavailable "
            f"({type(e).__name__}: {e}). Floors are the basis of every verdict this authority "
            f"issues; refusing rather than substituting a fallback threshold.") from e
    floors = {k: round(max(1.0, v - tolerance), 2) for k, v in ceil.items()}
    floors["add_commutative"] = None                              # blind / exact
    ceil = {k: round(v, 2) for k, v in ceil.items()}
    return floors, ceil, source


def attainable_absolute_bits(scale_bits: int) -> float:
    """Derived upper bound on ABSOLUTE decoded precision. Refuses if the noise model is absent."""
    try:
        nm = _import_noise_model()
        return float(nm.attainable_bits(int(scale_bits)))
    except Exception as e:                                          # noqa: BLE001
        raise RuntimeError(
            f"cannot derive the attainable precision bound: the ckks_golden noise model is "
            f"unavailable ({type(e).__name__}: {e}). The upper band decides whether a response is "
            f"physically producible at all; refusing rather than substituting a hand-set edge."
        ) from e


def relative_upper_bits(scale_bits: int, dyn: float) -> float:
    """The attainable bound expressed in the units the measurement actually reports.

    The score is relative to max|expected|, so the absolute bound gains log2 of that same quantity.
    This one line is the whole of the units fix: the suite previously compared a relative score to
    an absolute model bound and to a hand-set scale_bits + 8, and neither one was in the units of
    the number it was judging.
    """
    return attainable_absolute_bits(scale_bits) + math.log2(max(float(dyn), 1e-300))


# ------------------------------------------------------------------------------------------------
# Result binding (#1) — digest the exact output ciphertexts (ciphertext leaks nothing)
# ------------------------------------------------------------------------------------------------
def ct_digest(items) -> str:
    """sha256 binding a certificate to THIS result. Each item is either pre-serialized ciphertext
    BYTES (real backends, e.g. OpenFHE) or a raw limb triple [c0, c1, qm] (SoftwareCKKS). The verifier
    holds the same ciphertexts and re-derives the digest; user plaintext is never involved."""
    h = hashlib.sha256()
    for it in items:
        if isinstance(it, (bytes, bytearray)):
            h.update(b"B"); h.update(it)
        else:
            c0, c1, qm = it
            h.update(b"L"); h.update(str(qm).encode())
            h.update(str([int(x) for x in c0]).encode())
            h.update(str([int(x) for x in c1]).encode())
    return h.hexdigest()


def ct_items(be, raw_cts):
    """Canonical form for digesting a backend's ciphertexts: serialized bytes if the backend exposes
    serialize_ct (real libraries), else the raw limb triple (SoftwareCKKS)."""
    if hasattr(be, "serialize_ct"):
        return [be.serialize_ct(r) for r in raw_cts]
    return list(raw_cts)


# --------------------------------------------------------------------------------------------------
# Certificate
# --------------------------------------------------------------------------------------------------
@dataclass
class InvariantResult:
    name: str
    blind: bool
    passed: bool
    achieved_bits: Optional[float]
    floor_bits: Optional[float]


@dataclass
class Certificate:
    version: str
    method: str
    issued_at: float
    session_id: str
    backend: dict
    op_trace: list
    rounds: int
    probe_profile: str                 # "workload" (anti-sandbag) | "random"
    floor_source: str
    worst_case_bits: dict           # worst-case GUARANTEED precision, a lower bound (see derive_floors)
    precision_band: Optional[dict]  # derived upper band, per law, in the units the scores use
    result_digest: Optional[str]       # binds the cert to the exact output ciphertexts
    primitive_conformance: Optional[dict]   # optional RNS primitive-family attestation
    application_conformance: Optional[dict] # LLM-level: does the model still pick the right token?
    composition: Optional[dict]             # pipeline-composed floor (per-op pass != chain correct)
    challenge_seed: Optional[int]           # server-issued blinded challenge (anti-precompute/replay)
    adversarial_audit: Optional[dict]       # ofc_adversarial 5-axis anti-sandbag audit
    error_coupling: Optional[dict]          # DIAGNOSTIC: does pmul error carry w*add error?
    invariants: list
    verdict: str
    manifest_sha256: str
    signer: str
    signature_algo: str
    public_key: Optional[str]
    signature: str
    attested: bool = False             # True = known-answer challenge; False = self-reported (SECURITY.md)
    n_report: Optional[int] = None     # slots the AUTHORITY examined (was device-chosen)

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True, default=float, allow_nan=False)

    @staticmethod
    def from_json(s: str) -> "Certificate":
        """Parse a certificate, REFUSING any document the signature does not actually cover.

        This used to drop unknown keys and hand the survivors to the constructor, commented
        "forward-compatible". It was not: `verify()` re-serialises the parsed subset and hashes
        that, so anything the parser discarded was outside the signature. A certificate carrying
        `{"NOTE": "this certificate is REVOKED", "achieved_bits_override": 999}` verified VALID,
        and a document with the key `verdict` twice -- FAIL then PASS -- verified VALID while any
        first-occurrence parser read FAIL. Both are signature-evasion, not forward compatibility.

        A field genuinely added by a later version cannot be covered by a signature this version
        knows how to compute, so the only sound answer is to refuse the document and say why.
        """
        import dataclasses

        def _no_dupes(pairs):
            seen = set()
            for k, _ in pairs:
                if k in seen:
                    raise ValueError(f"duplicate key {k!r} in certificate: "
                                     f"parsers disagree on which value is authoritative")
                seen.add(k)
            return dict(pairs)

        data = json.loads(s, object_pairs_hook=_no_dupes)
        known = {f.name for f in dataclasses.fields(Certificate)}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f"certificate carries field(s) the signature does not cover: "
                             f"{', '.join(extra)}")
        missing = sorted(known - set(data) - {"attested", "n_report"})
        if missing:
            raise ValueError(f"certificate is missing signed field(s): {', '.join(missing)}")
        return Certificate(**{k: v for k, v in data.items() if k in known})

    @property
    def is_pass(self) -> bool:
        return self.verdict == "PASS"

    def _body(self) -> dict:
        d = asdict(self)
        d.pop("signature", None)
        d.pop("manifest_sha256", None)
        return d

    def canonical_bytes(self) -> bytes:
        return json.dumps(self._body(), sort_keys=True, separators=(",", ":"),
                          default=float, allow_nan=False).encode()

    def verify(self, hmac_key: bytes | None = None) -> bool:
        """Re-derive the manifest hash and check the signature. Ed25519 => third-party check with NO
        shared secret (embedded public key). HMAC fallback => pass the key."""
        from vbfhe_signing import verify_signature
        body = self.canonical_bytes()
        if hashlib.sha256(body).hexdigest() != self.manifest_sha256:
            return False
        return verify_signature(self.signature_algo, body, self.signature, self.public_key, hmac_key=hmac_key)

    def binds_result(self, raw_cts) -> bool:
        """True iff this certificate was issued for exactly these output ciphertexts."""
        return self.result_digest is not None and self.result_digest == ct_digest(raw_cts)


# --------------------------------------------------------------------------------------------------
# Oracle-free probe kernel (+#3 anti-sandbag workload profile)
# --------------------------------------------------------------------------------------------------
def _bits_dyn(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    """Achieved bits and the dynamic range they were measured against.

    The score is RELATIVE: err / max|expected|. That is the usual effective-precision number, but it
    means the score carries a log2(max|expected|) term that belongs to the authority's probe draw and
    not to the engine. Any bound compared against this score has to carry the same term, so the
    caller needs the dynamic range as well as the score. Returning one without the other is how the
    suite ended up comparing a relative measurement to an absolute model bound.
    """
    err = float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
    dyn = float(np.max(np.abs(np.asarray(rhs)))) + 1e-12
    if err <= 0:
        return 60.0, dyn
    return max(0.0, -math.log2(err / dyn + 1e-18)), dyn


def _bits(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return _bits_dyn(lhs, rhs)[0]


def _cteq(x, y) -> bool:
    return (x[2] == y[2]
            and np.array_equal(np.asarray(x[0], dtype=object), np.asarray(y[0], dtype=object))
            and np.array_equal(np.asarray(x[1], dtype=object), np.asarray(y[1], dtype=object)))


def _probe_vectors(rng, S, profile):
    """Return (a,b,c,w). 'workload' widens dynamic range and injects STRUCTURED inputs (constant,
    alternating, ramp, near-max) — a device tuned to pass small randoms is caught here (anti-sandbag;
    cf. ofc_adversarial: random probes miss what workload-representative probes catch)."""
    if profile in ("workload", "strict"):
        scale = rng.choice([0.5, 2.0, 6.0, 20.0] if profile == "strict" else [0.5, 2.0, 6.0])
        a = rng.normal(0, scale, size=S)
        # inject a structured stress pattern into b
        pat = rng.integers(0, 4)
        if pat == 0:   b = np.full(S, scale)                       # constant
        elif pat == 1: b = scale * (np.arange(S) % 2 * 2 - 1.0)    # alternating ±
        elif pat == 2: b = np.linspace(-scale, scale, S)          # ramp
        else:          b = rng.normal(0, scale, size=S)
        c = rng.normal(0, scale, size=S)
        w = rng.normal(0, scale, size=S)
    else:
        a = rng.normal(0, 1.0, size=S); b = rng.normal(0, 1.0, size=S)
        c = rng.normal(0, 1.0, size=S); w = rng.normal(0, 1.0, size=S)
    return a, b, c, w


def probe_measure(be, rounds: int, seed: int, profile: str) -> dict:
    """CLIENT-SIDE (accredited-lab) measurement: run the oracle-free probes on the device `be` and
    return the achieved bits per invariant (+ the blind commutativity result). No user data, no
    floors, no verdict — just what the device achieved on throwaway probes. This is what a device
    submits to the conformance server; the SERVER sets the floors and judges."""
    if rounds < 1:
        raise ValueError("probe_measure requires rounds >= 1")
    rng = np.random.default_rng(seed)
    S, D = be.slots, be.delta
    comm_ok = True
    hom_bits, pmul_bits, cmul_bits = [], [], []

    for _ in range(rounds):
        a, b, c, w = _probe_vectors(rng, S, profile)
        ea = be.encrypt(be.encode(a)); eb = be.encrypt(be.encode(b)); ec = be.encrypt(be.encode(c))

        _eq = be.ct_equal if hasattr(be, "ct_equal") else _cteq
        comm_ok &= _eq(be.add(ea, eb), be.add(eb, ea))                         # blind (no decryption)

        lhs = be.decode_ct(be.add(ea, eb), D); rhs = be.decode_ct(ea, D) + be.decode_ct(eb, D)
        hom_bits.append(_bits(lhs, rhs))

        wpt = be.encode(w)
        lhs = be.decode_ct(be.rescale(be.mul_plain(be.add(ea, eb), wpt)), D)
        rhs = be.decode_ct(be.add(be.rescale(be.mul_plain(ea, wpt)),
                                  be.rescale(be.mul_plain(eb, wpt))), D)
        pmul_bits.append(_bits(lhs, rhs))

        lhs = be.decode_ct(be.rescale(be.mul(be.add(ea, eb), ec)), D)
        rhs = be.decode_ct(be.add(be.rescale(be.mul(ea, ec)),
                                  be.rescale(be.mul(eb, ec))), D)
        cmul_bits.append(_bits(lhs, rhs))

    return {"add_commutative": bool(comm_ok),
            "add_homomorphism": round(float(min(hom_bits)), 2),
            "plainmul_distributive": round(float(min(pmul_bits)), 2),
            "ctmul_distributive": round(float(min(cmul_bits)), 2)}


# ------------------------------------------------------------------------------------------------
# ATTESTED (known-answer) mode — closes the self-report fabrication AND the consistently-wrong-engine
# gap (SECURITY.md Findings 2 & 3). The SERVER issues a blinded challenge seed; the device derives the
# probe PLAINTEXTS from that seed, encrypts them under its own key, computes, DECRYPTS the throwaway
# probes, and returns the decoded outputs. The server re-derives the SAME probes from the seed (so it
# KNOWS the expected answers) and scores the device's outputs against them. A device cannot fabricate
# (it can't know a fresh seed's probes) and cannot be consistently-wrong (a wrong key/scale/offset
# makes the decoded output disagree with the server's known plaintext). No user data is involved —
# only server-chosen throwaway probes are encrypted/decrypted.
_ATTEST_N = 64   # slots reported per op per round (small wire; enough for a precision estimate)


def _slot_window(seed: int, r: int, S: int, n: int):
    """Which slots the authority examines this round, derived from ITS seed.

    This used to be a hardcoded [0:n] on both sides. The examined index set was therefore fixed,
    public and unchallenged, so an engine correct on slots 0..63 and arbitrary everywhere else was
    certified -- 64 of 4096 slots at the headline parameters, i.e. wrong on 98.4% of its output.
    Deriving the window from the seed costs nothing and the device cannot know it in advance.
    """
    if n >= S:
        return np.arange(S)
    return np.sort(np.random.default_rng((int(seed) << 8) ^ (r + 1)).choice(S, size=int(n), replace=False))


def attested_response(be, seed: int, rounds: int, profile: str, n_report: int = _ATTEST_N) -> dict:
    """DEVICE side: run the server's seeded probes and return decoded outputs (throwaway probes)."""
    if rounds < 1:
        raise ValueError("attested_response requires rounds >= 1")
    rng = np.random.default_rng(seed)
    S, D = be.slots, be.delta
    n = min(int(n_report), S)
    has_ks = hasattr(be, "rotate")                            # keyswitch (rotation) coverage if supported
    out = {"add": [], "pmul": [], "cmul": []}
    if has_ks:
        out["ks"] = []
    for r in range(rounds):
        a, b, c, w = _probe_vectors(rng, S, profile)          # server will reproduce these exactly
        idx = _slot_window(seed, r, S, n)                     # CA-chosen slots, not always [0:n]
        ea, eb, ec = be.encrypt(be.encode(a)), be.encrypt(be.encode(b)), be.encrypt(be.encode(c))
        s1 = be.add(ea, eb)
        out["add"].append([float(x) for x in np.asarray(be.decode_ct(s1, D))[idx]])
        pm = be.rescale(be.mul_plain(s1, be.encode(w)))
        out["pmul"].append([float(x) for x in np.asarray(be.decode_ct(pm, D))[idx]])
        cm = be.rescale(be.mul(s1, ec))
        out["cmul"].append([float(x) for x in np.asarray(be.decode_ct(cm, D))[idx]])
        if has_ks:
            k = _ks_step(seed, r)                             # CA-chosen rotation amount (keyswitch)
            out["ks"].append([float(x) for x in np.asarray(be.decode_ct(be.rotate(be.encrypt(be.encode(a)), k), D))[idx]])
    # rotation_supported is DECLARED, so a device that can rotate cannot silently drop the law:
    # `has_ks = "ks" in outputs` meant an engine with a wrong rotation just omitted the key and the
    # invariant vanished from a signed PASS. Declaring false is still possible, but it is recorded.
    return {"attested": True, "rounds": int(rounds), "profile": profile, "n_report": n,
            "rotation_supported": bool(has_ks), "outputs": out}


def _ks_step(seed: int, r: int) -> int:
    """CA-chosen rotation amount for round r (1..8), derived from the challenge seed so both the device
    and the server compute the same value without disturbing the a/b/c/w probe stream."""
    return int(np.random.default_rng([int(seed) & 0xFFFFFFFF, r, 777]).integers(1, 9))


def attested_measurements(response: dict, N: int, seed: int, profile: str,
                          scale_bits: int | None = None) -> dict:
    """SERVER side: re-derive the probes from the seed (KNOWN answers) and score the device's decoded
    outputs against them. Returns the same measurement shape as probe_measure so judge_measurements
    applies the floors identically — but these bits are KNOWN-ANSWER, not self-consistency.

    With `scale_bits` the authority also applies the derived upper band, per round and per law. A
    round is refused when its score exceeds what the declared scale can carry against that round's
    own dynamic range. The band has to be per round because the score is relative and the probe
    amplitude is redrawn each round, so a single scalar edge is either too loose for the quiet
    rounds or too tight for the loud ones. The old hand-set edge of scale_bits + 8 was too loose for
    every round at these parameters.
    """
    rounds = int(response["rounds"]); n = int(response["n_report"]); S = int(N) // 2
    if response.get("profile") != profile:
        raise ValueError("attested response profile mismatch")
    rng = np.random.default_rng(seed)
    has_ks = "ks" in response.get("outputs", {})
    hom, pmul, cmul, ks = [], [], [], []
    ea_all, ep_all, w_all = [], [], []
    band_top: dict[str, list[float]] = {}

    def _score(acc, law, r, got, exp):
        """Score one law in one round, and refuse a score the parameters cannot produce."""
        bits, dyn = _bits_dyn(got, exp)
        if scale_bits is not None:
            upper = relative_upper_bits(scale_bits, dyn)
            if bits > upper:
                raise ValueError(
                    f"implausible measurement: {law} round {r} reports {bits:.2f} bits against a "
                    f"dynamic range of {dyn:.3f}, and a scale of 2^{scale_bits} carries at most "
                    f"{upper:.2f} bits there. A correct approximate computation cannot be this "
                    f"accurate, so the response is refused rather than scored.")
            band_top.setdefault(law, []).append(round(upper, 2))
        acc.append(bits)
    for r in range(rounds):
        a, b, c, w = _probe_vectors(rng, S, profile)
        idx = _slot_window(seed, r, S, n)
        exp_add, exp_pmul = (a + b)[idx], (w * (a + b))[idx]
        got_add = np.asarray(response["outputs"]["add"][r], dtype=float)
        got_pmul = np.asarray(response["outputs"]["pmul"][r], dtype=float)
        _score(hom, "add_homomorphism", r, response["outputs"]["add"][r], exp_add)
        _score(pmul, "plainmul_distributive", r, response["outputs"]["pmul"][r], exp_pmul)
        _score(cmul, "ctmul_distributive", r, response["outputs"]["cmul"][r], ((a + b) * c)[idx])
        # Error ALGEBRA, not error magnitude. `pmul` is evaluated on the very ciphertext that
        # produced `add` (s1 = add(ea, eb); pm = rescale(mul_plain(s1, w))), so an honest engine's
        # pmul error necessarily carries w * err_add. A respondent that fabricates each answer
        # independently -- including one that paints noise of exactly the right magnitude -- has no
        # such coupling. The authority already holds every term, so this costs nothing to check.
        ea_all.append(got_add - exp_add); ep_all.append(got_pmul - exp_pmul); w_all.append(w[idx])
        if has_ks:
            k = _ks_step(seed, r)
            _score(ks, "keyswitch_rotation", r, response["outputs"]["ks"][r],
                   np.roll(a, -k)[idx])                              # rotate maps to roll(-k)
    # add_commutative is NOT emitted here. In attested mode nothing about commutativity is
    # challenged, sent or checked -- it used to be hardcoded True and signed, so it read PASS for a
    # respondent that ran no FHE at all AND for an engine the same certificate failed. A field that
    # cannot fail is not evidence. Add correctness is covered by the known-answer add check.
    meas = {"add_homomorphism": round(float(min(hom)), 2),
            "plainmul_distributive": round(float(min(pmul)), 2),
            "ctmul_distributive": round(float(min(cmul)), 2),
            "error_coupling_r": _error_coupling(ea_all, ep_all, w_all)}
    if has_ks:
        meas["keyswitch_rotation"] = round(float(min(ks)), 2)   # KNOWN-ANSWER keyswitch coupling
    if band_top:
        # Recorded so a reader can see which band each score was judged against. A band that is
        # applied but not published is a threshold the respondent cannot check.
        meas["precision_band_top"] = {k: v for k, v in sorted(band_top.items())}
    return meas


def _error_coupling(err_add, err_pmul, weights) -> float:
    """Pearson r between the reported pmul error and w * (reported add error), pooled over rounds.

    An honest CKKS engine derives pmul from the add result, so this is strongly positive. An
    independent fabricator sits at zero however carefully it matches the error MAGNITUDE. This does
    not make the suite sound: a delegate can carry one error vector per probe and push it through
    the plaintext circuit, which is still O(n). It raises the cost of the fabrication from matching
    a distribution to reproducing the circuit's error algebra, and it is reported, not enforced,
    because we have not characterised its false-refusal rate on real engines.
    """
    x = np.concatenate([np.asarray(w, dtype=float) * np.asarray(e, dtype=float)
                        for w, e in zip(weights, err_add)])
    y = np.concatenate([np.asarray(e, dtype=float) for e in err_pmul])
    if x.size < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


# The self-report path receives a number and no probes, so it cannot know the dynamic range the
# number was measured against and cannot derive an exact upper bound. This allowance is the widest
# log2(max|expected|) the workload profile plausibly produces, and it is the last hand-set constant
# in the acceptance rule. The attested path derives its band per round from probes it holds and does
# not use this edge. A bound this loose is a reason to prefer the attested path, not a defence.
_UNKNOWN_DYNAMIC_RANGE_BITS = 8.0


def _plausible_bits(x, scale_bits) -> float:
    """Reject implausible / adversarial self-reported bit counts (inf/nan/negative, or more than the
    scale can physically carry) on the SELF-REPORT path, where no probes are available.

    This REFUSES rather than clamps. It used to return min(v, ceiling), which silently rewrote an
    impossible measurement into a passing one: a respondent answering in exact arithmetic reports
    hundreds of bits, was clamped, and PASSED. A precision above what the declared scale can carry
    is not a good result, it is evidence the approximate computation was not performed.

    The edge itself used to be scale_bits + 8, hand-set and in the wrong units. Its base is now
    derived, and what remains hand-set is only the allowance for a dynamic range this path cannot
    observe.
    """
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"implausible measurement: {x!r}")
    edge = attainable_absolute_bits(scale_bits) + _UNKNOWN_DYNAMIC_RANGE_BITS
    if v > edge:
        raise ValueError(
            f"implausible measurement: {v:.2f} bits exceeds the {edge:.1f}-bit edge that a scale of "
            f"2^{scale_bits} can carry even at the widest dynamic range this profile draws; a "
            f"correct approximate computation cannot be this accurate")
    return v


def judge_measurements(measurements: dict, floors: dict, scale_bits: int = 64,
                       mode: str = "known-answer") -> list[InvariantResult]:
    """SERVER-SIDE (certification authority): apply the derived floors to the device's measurements.
    The server owns the pass/fail bar — a device cannot pick an easy floor — and rejects physically
    implausible self-reported values (see SECURITY.md: the remote path still needs device attestation
    to bind these numbers to a real execution).

    `mode` decides whether an upper bound applies at all, and it has to, because the two modes
    measure different things. A known-answer score compares the device's output to a plaintext the
    authority computed, so the error is the device's full decode error and the attainable bound
    applies. A self-consistency score compares two of the device's OWN outputs, whose errors share
    most of their terms and cancel, so the same honest engine scores far higher: 50.76 bits against
    14.36 for the same run at N=256, scale 2^22. Applying the known-answer bound there refuses an
    honest device. It was applied there, and the only reason no honest device was refused is that
    the one caller left `scale_bits` at its default of 64, which put the edge out of reach. A bound
    that is only harmless because it is unreachable is not a control.
    """
    if mode not in ("known-answer", "self-consistency"):
        raise ValueError(f"unknown judging mode {mode!r}")

    def _res(name, blind):
        if blind:
            return InvariantResult(name, True, bool(measurements[name]), None, None)
        if mode == "known-answer":
            achieved = _plausible_bits(measurements[name], scale_bits)
        else:
            achieved = float(measurements[name])
            if not math.isfinite(achieved) or achieved < 0.0:
                raise ValueError(f"implausible measurement: {measurements[name]!r}")
        floor = floors[name]
        return InvariantResult(name, False, achieved >= floor, round(achieved, 2), floor)
    res = []
    if "add_commutative" in measurements:          # self-consistency mode only; see attested_measurements
        res.append(_res("add_commutative", True))
    res += [_res("add_homomorphism", False),
            _res("plainmul_distributive", False), _res("ctmul_distributive", False)]
    if "keyswitch_rotation" in measurements:                  # attested keyswitch coverage (if supported)
        res.append(_res("keyswitch_rotation", False))
    return res


def run_invariants(be, rounds: int, seed: int, profile: str, floors: dict) -> list[InvariantResult]:
    """Co-located measure+judge (the LocalOracleFreeService path)."""
    return judge_measurements(probe_measure(be, rounds, seed, profile), floors,
                              mode="self-consistency")


# ------------------------------------------------------------------------------------------------
# #4 Optional primitive-layer attestation (RNS op family: NTT/automorphism/keyswitch laws)
# ------------------------------------------------------------------------------------------------
def primitive_attestation(rounds: int = 6, challenge_seed=None) -> Optional[dict]:
    """Run the oracle-free RNS-primitive invariants (ckks_golden/ofc_invariants) on the reference op
    family. Attests the PRIMITIVE family conforms (NTT linearity, automorphism group law, keyswitch
    identity). Honestly labelled: this is the reference RNS family, not the SDK's software backend."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        vbfhe = os.path.abspath(os.path.join(here, ".."))
        if vbfhe not in sys.path:
            sys.path.append(vbfhe)
        from ckks_golden.ofc_invariants import Device, certify, verdict   # type: ignore
        from ckks_golden.keyswitch import Context                         # type: ignore
        ctx = Context(N=16, L=6, alpha=3, prime_bits=60)
        res = certify(Device(ctx), rounds=rounds, seed=challenge_seed)
        # certify() returns three-valued Verdicts (PASS / FAIL / NOT TESTED). Record the code, not
        # a bool: collapsing NOT TESTED into a bool is how an untested law came to read as a passing
        # one. `verdict()` is CONFORMANT only when every invariant was actually tested and passed.
        # SUBJECT is the reference RNS family — NOT the device under test. This attests the primitive
        # laws hold for a correct implementation; it does not test the certified backend. (See SECURITY.md.)
        return {"subject": "reference-RNS (NOT the device under test)", "profile": "CKKS-N16-SMOKE",
                "invariants": {k: v.code for k, v in res.items()},
                "verdict": verdict(res),
                "conformant": verdict(res) == "CONFORMANT"}
    except Exception as e:
        return {"subject": "reference-RNS (NOT the device under test)",
                "error": f"{type(e).__name__}: {e}", "conformant": None}


def adversarial_attestation(challenge_seed: int, public_seed: int = 0xABCDEF) -> Optional[dict]:
    """Run the FHE-ACVP anti-sandbag audit (ckks_golden/ofc_adversarial) with the SERVER's blinded
    challenge seed. Five axes: fixed public-seed, blinded-challenge, per-stage manifest,
    precision-plausibility, and measured-floor on workload-band inputs.

    IMPORTANT (see SECURITY.md): the audit is run against the REFERENCE HonestDevice, NOT the device
    under test — the current SDK does not wire the certified backend into ofc_adversarial. So this
    section attests that the reference family is sandbag-resistant to the server's challenge; it is NOT
    an attestation about the certified device. It is recorded for transparency and is NOT folded into
    the verdict. Binding this audit to the real device (challenge-response over its own outputs) is
    tracked as required future work."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        vbfhe = os.path.abspath(os.path.join(here, ".."))
        if vbfhe not in sys.path:
            sys.path.append(vbfhe)
        from ckks_golden.ofc_adversarial import certify as adv_certify, HonestDevice  # type: ignore
        from ckks_golden.keyswitch import Context                                     # type: ignore
        ctx = Context(N=16, L=6, alpha=3, prime_bits=60)
        cs = int(challenge_seed) & 0x7FFFFFFF
        res = adv_certify(HonestDevice(), ctx, public_seed, cs, audit_band=(0.6, 1.0), n_audit=400)
        return {"subject": "reference-RNS (NOT the device under test)", "challenge_seed": cs,
                "axes": {k: bool(v) for k, v in res.items()}, "resistant": bool(all(res.values()))}
    except Exception as e:
        return {"subject": "reference-RNS (NOT the device under test)",
                "error": f"{type(e).__name__}: {e}", "resistant": None}


# ------------------------------------------------------------------------------------------------
# #Phase2 Application-level (LLM) conformance — does the model still pick the right token?
# ------------------------------------------------------------------------------------------------
_INFERENCE_OPS = {"matvec", "poly_eval", "attention", "inner_product",
                  "sr_mul", "multiply", "mul", "ckks_mul"}   # + FHETCH/OpenFHE primitive op names


def application_conformance(invariants, op_trace, app_tol: float = 0.99) -> Optional[dict]:
    """Bridge the measured arithmetic precision to the APPLICATION question ("does the LLM still get
    the right token?") via ckks_golden/ofc_e2e. Only meaningful for an inference-shaped workload, so
    returns None unless the op-trace contains a matvec/attention-class op. device_bits = the measured
    ct×ct precision (the matvec arithmetic bottleneck)."""
    if not any(t.get("op") in _INFERENCE_OPS for t in op_trace):
        return None
    ctmul = next((i for i in invariants if i.name == "ctmul_distributive"), None)
    if ctmul is None or ctmul.achieved_bits is None:
        return None
    device_bits = max(1, min(16, int(ctmul.achieved_bits)))     # argmax saturates ~100% above ~14b
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        vbfhe = os.path.abspath(os.path.join(here, ".."))
        if vbfhe not in sys.path:
            sys.path.append(vbfhe)
        from ckks_golden.ofc_e2e import e2e_conformance          # type: ignore
        r = e2e_conformance(device_bits, app_tol=app_tol)
        return {"metric": "argmax-agreement", "device_bits": device_bits,
                "agreement": round(float(r["agreement"]), 4), "app_tol": app_tol,
                "note": "INDICATIVE: plaintext-quantization proxy, not an encrypted LLM run; saturates "
                        "above ~14 bits (see SECURITY.md)",
                "conformant": bool(r["conformant"])}
    except Exception as e:
        return {"metric": "argmax-agreement", "device_bits": device_bits,
                "error": f"{type(e).__name__}: {e}", "conformant": None}


# ------------------------------------------------------------------------------------------------
# #Phase3 Pipeline-composition conformance — per-op pass != chain correct (errors accumulate)
# ------------------------------------------------------------------------------------------------
_MUL_OPS = {"matvec", "poly_eval", "attention", "inner_product",
            "sr_mul", "multiply", "mul", "ckks_mul"}          # + FHETCH/OpenFHE primitive op names


def composition_conformance(invariants, op_trace, floors) -> Optional[dict]:
    """Certify the COMPOSED pipeline, not just isolated ops: independent per-op errors accumulate
    (~0.5·log2(chain) bits lost), so the chain output must still clear the floor. chain_len = number
    of multiplicative/keyswitch-consuming ops in the user's trace. Reuses ckks_golden/ofc_composition."""
    ctmul = next((i for i in invariants if i.name == "ctmul_distributive"), None)
    if ctmul is None or ctmul.achieved_bits is None:
        return None
    chain_len = sum(1 for t in op_trace if t.get("op") in _MUL_OPS)
    if chain_len < 2:
        # Nothing was composed. This used to clamp to 1 and emit conformant=true, which restated
        # the ctmul verdict under a name that implies a pipeline result. Report not-applicable.
        return {"chain_len": chain_len, "conformant": None,
                "note": "no multiplicative chain in op_trace; composition not exercised"}
    floor = floors["ctmul_distributive"]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        vbfhe = os.path.abspath(os.path.join(here, ".."))
        if vbfhe not in sys.path:
            sys.path.append(vbfhe)
        from ckks_golden.ofc_composition import composed_pass     # type: ignore
        ok, composed_bits = composed_pass(ctmul.achieved_bits, floor, chain_len)
        return {"chain_len": chain_len, "per_op_bits": round(ctmul.achieved_bits, 2),
                "composed_bits": round(float(composed_bits), 2), "floor_bits": floor,
                "conformant": bool(ok)}
    except Exception as e:
        return {"chain_len": chain_len, "error": f"{type(e).__name__}: {e}", "conformant": None}


# --------------------------------------------------------------------------------------------------
# Service interface (OPEN) + local reference judge (OPEN) + remote submit / FHETCH interop (CLOSED)
# --------------------------------------------------------------------------------------------------
def assemble_certificate(*, descriptor, session_id, op_trace, rounds, profile, floors,
                         worst_case_bits, floor_source, invs, result_digest, with_primitive, signer,
                         signer_name, challenge_seed=None, with_adversarial=True, attested=False,
                         error_coupling=None, n_report=None, precision_band=None):
    """Build + sign a Certificate from judged invariants. Shared by the local judge and the
    conformance server so both emit identical certificates (only the signer differs)."""
    prim = primitive_attestation(challenge_seed=challenge_seed) if with_primitive else None
    adv = adversarial_attestation(challenge_seed) if (with_adversarial and challenge_seed is not None) else None
    app = application_conformance(invs, op_trace)
    comp = composition_conformance(invs, op_trace, floors)
    # FAIL-CLOSED: a section that is None means "not applicable" (OK); but if a section RAN and did
    # not return conformant==True (i.e. it failed OR errored to conformant==None), it must not pass.
    # Three states, not two. A section can be ABSENT (not requested), NOT APPLICABLE (it ran and
    # had nothing to exercise -- conformant is None with a reason and no error), or BROKEN (it tried
    # and errored). Absent and not-applicable are fine; broken must not pass, because otherwise a
    # missing dependency yields a signed PASS with the check silently skipped.
    def _section_ok(sec):
        if sec is None:
            return True
        if "error" in sec:
            return False
        return sec.get("conformant") is not False
    invariants_ok = all(i.passed for i in invs)
    app_ok = _section_ok(app)
    comp_ok = _section_ok(comp)
    # A section that ERRORED must not be waved through. `prim` and `adv` are attestations about the
    # reference RNS family and not about the device, so they do not decide whether the device is
    # conformant -- but if either failed to RUN, the certificate would assert a check that never
    # happened. That is how a missing dependency produced a signed PASS with the adversarial audit
    # silently errored out. An absent section is fine; a broken one is not.
    prim_ran = (prim is None) or ("error" not in prim and prim.get("conformant") is not False)
    adv_ran = (adv is None) or ("error" not in adv and adv.get("resistant") is not False)
    verdict = "PASS" if (invariants_ok and app_ok and comp_ok and prim_ran and adv_ran) else "FAIL"
    cert = Certificate(
        version=CERT_VERSION, method=OFC_METHOD, issued_at=round(time.time(), 3),
        session_id=session_id, backend=descriptor, op_trace=list(op_trace), rounds=int(rounds),
        probe_profile=profile, floor_source=floor_source, worst_case_bits=worst_case_bits,
        precision_band=precision_band, result_digest=result_digest,
        primitive_conformance=prim, application_conformance=app, composition=comp,
        challenge_seed=(int(challenge_seed) if challenge_seed is not None else None),
        adversarial_audit=adv, error_coupling=error_coupling, n_report=n_report,
        attested=bool(attested),
        invariants=[asdict(i) for i in invs], verdict=verdict, manifest_sha256="",
        signer=signer_name, signature_algo=signer.algo, public_key=signer.public_key_hex, signature="")
    body = cert.canonical_bytes()
    cert.manifest_sha256 = hashlib.sha256(body).hexdigest()
    cert.signature = signer.sign(body)
    return cert


class ConformanceService:
    def certify(self, session, op_trace, rounds=8) -> Certificate:
        raise NotImplementedError


class LocalOracleFreeService(ConformanceService):
    """Reference oracle-free judge — the runnable local certifier that ships OPEN with the client.
    Certifies the session's actual backend and signs the result (Ed25519 => third-party verifiable)."""
    signer_name = "vbfhe-local-oracle-free/reference"

    def __init__(self, signer=None, seed: int = 20260701, profile: str = "workload",
                 tolerance: float = CONFORMANCE_TOLERANCE_BITS, with_primitive: bool = True,
                 strict: bool = False):
        from vbfhe_signing import default_signer
        self._signer = signer or default_signer()
        self._seed = seed
        # strict = anti-sandbag hardening (wider production-band probes + a tighter floor); design
        # basis: ckks_golden/ofc_adversarial measured-floor audit on workload-band inputs.
        self._profile = "strict" if strict else profile
        self._tol = 4.0 if strict else tolerance
        self._with_primitive = with_primitive

    def certify(self, session, op_trace, rounds=8, result_digest=None) -> Certificate:
        be = session.be
        floors, worst_case_bits, source = derive_floors(be, self._tol)
        invs = run_invariants(be, rounds=rounds, seed=self._seed, profile=self._profile, floors=floors)
        desc = {"scheme": "CKKS", "backend_class": type(be).__name__, "N": int(be.N),
                "slots": int(be.slots), "scale_bits": int(be.scale_bits), "q_bits": int(be.q.bit_length())}
        return assemble_certificate(
            descriptor=desc, session_id=session.session_id(), op_trace=op_trace, rounds=rounds,
            profile=self._profile, floors=floors, worst_case_bits=worst_case_bits,
            floor_source=source, invs=invs,
            result_digest=result_digest, with_primitive=self._with_primitive,
            signer=self._signer, signer_name=self.signer_name)


class RemoteConformanceService(ConformanceService):
    """CLOSED seam — the paid path. Packages a request (backend descriptor + op trace + probe seeds;
    NO ciphertexts, NO keys) and submits it to the VaultBytes conformance service, which runs the full
    patented judging engine and returns a signed certificate. Inject transport to point at the endpoint.

    #6 Interop: `from_fhetch_trace()` builds a request from a FHETCH/OpenFHE trace, so you can
    'certify any FHE build, including Niobium/FHETCH traces.'"""
    signer = "vbfhe-conformance-service/remote"

    def __init__(self, endpoint="https://conformance.vaultbytes.com/v0/certify",
                 api_key: Optional[str] = None, transport=None, profile: str = "workload"):
        self.endpoint = endpoint; self.api_key = api_key; self._transport = transport
        self._profile = profile

    PROBE_SEED = 20260701

    def build_request(self, session, op_trace, rounds, result_digest=None) -> dict:
        """The device MEASURES its own precision on throwaway probes (client side) and submits the
        achieved bits + descriptor + op-trace — NO ciphertexts, NO keys, NO plaintext. The server sets
        the floors and judges. (Anti-fabrication of the measurements is the ofc_adversarial
        stage-manifest layer; the reference server judges the submitted measurements.)"""
        be = session.be
        measurements = probe_measure(be, rounds=rounds, seed=self.PROBE_SEED, profile=self._profile)
        return {"endpoint": self.endpoint, "scheme": "CKKS", "backend_class": type(be).__name__,
                "N": int(be.N), "slots": int(be.slots), "scale_bits": int(be.scale_bits),
                "q_bits": int(be.q.bit_length()), "session_id": session.session_id(),
                "op_trace": list(op_trace), "rounds": int(rounds), "profile": self._profile,
                "probe_seed": self.PROBE_SEED, "measurements": measurements, "result_digest": result_digest}

    @staticmethod
    def from_fhetch_trace(trace: dict) -> dict:
        """Build a conformance request from a FHETCH/OpenFHE Polynomial-IR trace. Accepts a dict with
        keys like {'params': {'ring_dim'|'N', 'scale_bits'?}, 'ops': [{'op': 'sr_ntt'|...}, ...]}.
        Extracts only the SHAPE (params + op names) — no ciphertexts, no keys."""
        p = trace.get("params", {})
        N = int(p.get("N") or p.get("ring_dim") or 0)
        ops = [{"op": o.get("op") or o.get("kind") or "?"} for o in trace.get("ops", [])]
        return {"source": "fhetch", "scheme": trace.get("scheme", "CKKS"), "N": N,
                "scale_bits": int(p.get("scale_bits", 0)), "op_trace": ops, "rounds": 8,
                "probe_seed": 20260701}

    def certify(self, session, op_trace, rounds=8, result_digest=None) -> Certificate:
        req = self.build_request(session, op_trace, rounds, result_digest=result_digest)
        if self._transport is None:
            raise NotImplementedError(
                "RemoteConformanceService is the paid VaultBytes conformance service (closed). "
                "Inject transport=... (e.g. vbfhe_server.in_process_transport) to submit, or use "
                "LocalOracleFreeService for the open reference judge.")
        return Certificate.from_json(self._transport(req))


class AttestedRemoteConformanceService(ConformanceService):
    """SOUND remote path (SECURITY.md Findings 2 & 3 closed). Two-step known-answer challenge:
      1. fetch a fresh BLINDED challenge seed from the server (device can't precompute it),
      2. run the server's seeded probes on the device (attested_response) — encrypt/compute/DECRYPT
         throwaway probes, no user data,
      3. submit the decoded outputs; the server re-derives the probes (it KNOWS the answers), scores
         them, and signs. A device cannot fabricate numbers and cannot be consistently-wrong.
    The transport handles two message kinds: {"kind":"challenge"} and {"kind":"attest", ...}."""
    def __init__(self, transport=None, profile: str = "workload"):
        self._transport = transport
        self._profile = profile

    def certify(self, session, op_trace, rounds=8, result_digest=None) -> Certificate:
        if self._transport is None:
            raise NotImplementedError("inject transport=... (e.g. vbfhe_server.in_process_transport)")
        be = session.be
        ch = json.loads(self._transport({"kind": "challenge", "profile": self._profile, "rounds": rounds}))
        resp = attested_response(be, seed=int(ch["seed"]), rounds=int(ch["rounds"]),
                                 profile=ch["profile"])
        req = {"kind": "attest", "challenge_id": ch["challenge_id"], "scheme": "CKKS",
               "backend_class": type(be).__name__, "N": int(be.N), "slots": int(be.slots),
               "scale_bits": int(be.scale_bits), "q_bits": int(be.q.bit_length()),
               "session_id": session.session_id(), "op_trace": list(op_trace),
               "response": resp, "result_digest": result_digest}
        return Certificate.from_json(self._transport(req))


# ------------------------------------------------------------------------------------------------
# #7 Hardware backend seam — the protocol a silicon-proven CKKS backend implements so certify() runs
# against the real chip instead of SoftwareCKKS. certify() is already backend-agnostic: it only calls
# encode / encrypt / add / add_plain / mul_plain / mul / rescale / decode_ct + reads .slots/.delta/.N/
# .scale_bits/.q. Any object honouring that protocol (a chip host, a GPU backend) plugs straight in.
# ------------------------------------------------------------------------------------------------
class HardwareBackend:
    """Protocol stub. A real implementation wraps the AWS-F2 host driver (or GLIDE CUDA backend) and
    fulfils the same op surface as SoftwareCKKS. Left unimplemented on purpose (needs the pod/chip)."""
    def __getattr__(self, name):
        raise NotImplementedError(
            f"HardwareBackend.{name} not wired — implement the SoftwareCKKS op protocol against the "
            f"silicon-proven CKKS host (AWS-F2 / GLIDE). certify() runs unchanged once it does.")
