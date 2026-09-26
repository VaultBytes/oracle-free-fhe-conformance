#!/usr/bin/env python3
"""Blind attested conformance: the AUTHORITY holds the key and the device never sees plaintext.

WHY THIS EXISTS

The attested known-answer profile in `vbfhe_conformance` hands the device the whole problem in
the clear. The authority sends a seed; the device derives the probe plaintexts from it, encrypts
them under its OWN key, evaluates, DECRYPTS, and returns decoded numbers. Every input a dishonest
respondent needs is therefore available to it in plaintext, and `attack_delegate.py` shows what
follows: a respondent that performs no homomorphic operation at all re-derives the probes, answers
in float64, paints Gaussian noise to any target precision, and receives a signed PASS.

No threshold fixes that, because the obstruction is economic rather than statistical. Producing
CKKS's decoded output costs O(n) plaintext flops while evaluating the circuit costs
O(ell * N log N) plus key-switching, so the dishonest respondent is not merely indistinguishable
from the honest one, it is faster. Every cost, latency and throughput defence therefore runs
backwards: raising probe volume stresses the honest party and not the attacker.

This module inverts the protocol so the asymmetry disappears. The authority generates the context
and the keypair, encrypts the probes ITSELF, and sends ciphertexts plus the public evaluation key.
The device evaluates blind and returns ciphertexts. The authority decrypts with a key it never
shared and scores the result.

WHAT THAT BUYS, AND WHAT IT DOES NOT

A delegate cannot read the probe values out of the ciphertexts it is given, so it cannot compute
the answers cheaply in the clear. Forging a passing response now requires producing ciphertexts
that decrypt correctly under a key the forger does not hold, which is the problem CKKS's security
rests on. Three further problems go with it:

  * the cost argument stops being inverted, because forging costs about what honest evaluation
    costs, so probe volume becomes a usable binding on hardware class;
  * the device never decrypts anything, so the approximate-decryption oracle that the attested
    path creates, an IND-CPA-D exposure on the device's own key, simply does not arise;
  * the precision floor is derived from parameters the AUTHORITY chose, so a respondent can no
    longer declare the parameters that set its own bar.

It does not bind the computation to a particular piece of silicon. A delegate may forward the
ciphertexts to a real CKKS library on an ordinary host and return what comes back. That is still
delegation, but it is delegation to something that genuinely performs CKKS, so what the certificate
establishes is that correct CKKS evaluation happened somewhere under the authority's parameters.
Binding it to a named device remains a matter for custody or a hardware root of trust, and no
protocol over this interface will supply it.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

_OPS = ("add", "pmul", "cmul")


def blind_evaluator(backend) -> Any:
    """A view of `backend` that can evaluate but cannot decrypt.

    The secret is removed rather than hidden, so `decrypt` raises instead of quietly returning
    something. A device that could still decrypt would make the whole exercise decorative, and
    this is asserted in `selftest_blindness` rather than assumed.
    """
    dev = copy.copy(backend)
    dev.s = None                                   # the secret polynomial, gone
    dev._blind = True

    def _refuse(*_a, **_k):
        raise PermissionError("blind evaluator holds no secret key and cannot decrypt")

    dev.decrypt = _refuse
    dev.decode_ct = _refuse
    return dev


def selftest_blindness(dev) -> bool:
    """True iff the evaluator genuinely cannot decrypt. A control, not a comment."""
    try:
        dev.decrypt(None)
    except PermissionError:
        return getattr(dev, "s", None) is None
    except Exception:
        return getattr(dev, "s", None) is None
    return False


def issue_blind_challenge(auth, seed: int, rounds: int = 4, n_report: int = 64) -> dict:
    """AUTHORITY side. Encrypt the probes under the authority's own key.

    Returns the AUTHORITY's copy of the session: the ciphertexts, the plaintext weight the device
    multiplies by (a plaintext operand is public by construction), and the expected answers it
    keeps to itself for scoring. `device_view` is what may be sent. Handing this whole object to
    the device would hand it `_expect`, which is every answer in the clear, and the protocol would
    be back to the one it replaces.
    """
    rng = np.random.default_rng(int(seed))
    S, D = auth.slots, auth.delta
    n = min(int(n_report), S)
    work, expect = [], []
    for _ in range(int(rounds)):
        a, b, c = rng.normal(size=S), rng.normal(size=S), rng.normal(size=S)
        w = rng.normal(size=S)
        work.append({"ct_a": auth.encrypt(auth.encode(a)),
                     "ct_b": auth.encrypt(auth.encode(b)),
                     "ct_c": auth.encrypt(auth.encode(c)),
                     "pt_w": auth.encode(w)})
        expect.append({"add": a + b, "pmul": w * (a + b), "cmul": (a + b) * c})
    return {"seed": int(seed), "rounds": int(rounds), "n_report": n,
            "scale": D, "work": work, "_expect": expect}


def device_view(challenge: dict) -> dict:
    """Exactly what the authority may transmit. Everything else stays on the authority's side.

    The leading underscore on `_expect` is a naming convention, not a boundary, and a convention is
    not a protocol. This function is the boundary, and `selftest_device_view` checks it.
    """
    return {"rounds": challenge["rounds"], "n_report": challenge["n_report"],
            "scale": challenge["scale"], "work": challenge["work"]}


def selftest_device_view(challenge: dict) -> bool:
    """True iff the transmitted object carries no plaintext the device is supposed to be blind to."""
    view = device_view(challenge)
    if "_expect" in view or "seed" in view:
        return False
    flat = repr(sorted(view.keys()))
    return "_expect" not in flat


def blind_evaluate(dev, challenge: dict) -> dict:
    """DEVICE side. Evaluate foreign ciphertexts with no secret key. Returns ciphertexts.

    Takes the device view. Passing the authority's record works too and is what a careless caller
    would do, so the session object below never offers it.
    """
    out = {k: [] for k in _OPS}
    for job in challenge["work"]:
        s1 = dev.add(job["ct_a"], job["ct_b"])
        out["add"].append(s1)
        out["pmul"].append(dev.rescale(dev.mul_plain(s1, job["pt_w"])))
        out["cmul"].append(dev.rescale(dev.mul(s1, job["ct_c"])))
    return {"blind": True, "rounds": challenge["rounds"], "outputs": out}


def judge_blind(auth, challenge: dict, response: dict) -> dict:
    """AUTHORITY side. Decrypt with the key that was never shared and score.

    Precision is measured exactly as the attested path measures it, so the numbers are comparable,
    but every quantity that decides the verdict now originates with the authority.
    """
    if int(response.get("rounds", -1)) != int(challenge["rounds"]):
        raise ValueError("response round count does not match the issued challenge")
    n, D = challenge["n_report"], challenge["scale"]
    got = response["outputs"]
    for op in _OPS:
        if not isinstance(got.get(op), list) or len(got[op]) != challenge["rounds"]:
            raise ValueError(f"response does not answer every issued round for '{op}'")

    meas: dict[str, float] = {}
    for op in _OPS:
        worst = None
        for r in range(challenge["rounds"]):
            exp = np.asarray(challenge["_expect"][r][op], dtype=float)[:n]
            dec = np.asarray(auth.decode_ct(got[op][r], D), dtype=float)[:n]
            denom = max(float(np.max(np.abs(exp))), 1e-30)
            rel = float(np.max(np.abs(dec - exp))) / denom
            bits = 60.0 if rel <= 0 else min(60.0, -np.log2(rel))
            worst = bits if worst is None else min(worst, bits)
        meas[{"add": "add_homomorphism", "pmul": "plainmul_distributive",
              "cmul": "ctmul_distributive"}[op]] = round(float(worst), 2)
    return meas


class BlindSession:
    """One certification session: one ephemeral key, one challenge, one scoring, then no key.

    WHY THE STATE MACHINE IS PART OF THE PROTOCOL

    Moving the observable into the ciphertext domain puts the AUTHORITY in the position of
    decrypting ciphertexts a respondent produced, and then publishing a function of those
    decryptions as a score. That is a decryption interface, and a decryption interface that a
    respondent can query repeatedly under one key, with feedback between queries, is the setting in
    which approximate decryption has been shown to leak (Li and Micciancio, EUROCRYPT 2021). The
    device never decrypting is not on its own an answer, because the authority does.

    So the session is single-shot by construction rather than by convention:

      * the key is EPHEMERAL, generated with the session and used for nothing else;
      * the whole challenge is issued BEFORE any feedback, so the respondent commits to every round
        without seeing a score for any of them;
      * exactly one response is scored, and a second is refused rather than scored again, so there
        is no adaptive query loop;
      * the secret is DESTROYED when the session closes, so a later response cannot be scored under
        the same key even by the authority itself;
      * the score is the only thing published, and the probes are never published, so a respondent
        cannot learn which plaintexts a decryption belonged to.

    What this does NOT do. It bounds the adversary to one non-adaptive query per key, which is
    weaker than proving the interface safe. We have not carried out a reduction to IND-CPA-D or to
    any other standard notion, and a protocol that certifies the same device repeatedly issues one
    such query per session. Whether many single-query sessions compose is open, and noise flooding,
    the standard remedy, would widen the very band the verdict depends on.
    """

    def __init__(self, backend_factory, seed: int, rounds: int = 4, n_report: int = 64):
        self.auth = backend_factory()          # fresh context AND fresh keypair, per session
        self._record = issue_blind_challenge(self.auth, seed=seed, rounds=rounds,
                                             n_report=n_report)
        self._issued = False
        self._scored = False
        self._closed = False

    @property
    def digest(self) -> str:
        return challenge_digest(self._record)

    def challenge(self) -> dict:
        """The transmittable challenge. Issued once, in full, before any feedback."""
        if self._closed:
            raise PermissionError("session is closed; its key no longer exists")
        if self._issued:
            raise PermissionError(
                "challenge already issued. Re-issuing would let a respondent see a score, ask for "
                "the same probes again and answer differently, which is the adaptive loop this "
                "session exists to prevent.")
        self._issued = True
        return device_view(self._record)

    def judge(self, response: dict) -> dict:
        """Score exactly one response, then close the session and destroy the key."""
        if self._closed:
            raise PermissionError(
                "session is closed. Its secret key has been destroyed, so this response cannot be "
                "scored under the key its ciphertexts were produced for.")
        if self._scored:
            raise PermissionError("a response has already been scored in this session")
        if not self._issued:
            raise PermissionError("no challenge was issued in this session")
        self._scored = True
        try:
            _validate_response_shape(self._record, response)
            return judge_blind(self.auth, self._record, response)
        finally:
            self.close()

    def close(self) -> None:
        """Destroy the secret. Idempotent."""
        if not self._closed:
            self.auth.s = None
            self._record["_expect"] = None
            self._closed = True

    def selftest(self) -> dict:
        """Controls for the three properties the docstring claims, checked rather than asserted."""
        return {"device_view_hides_answers": selftest_device_view(self._record),
                "challenge_issued_once": self._issued,
                "key_destroyed_after_scoring": self._closed and getattr(self.auth, "s", 1) is None}


def _validate_response_shape(record: dict, response: dict) -> None:
    """Refuse a malformed response before decrypting anything in it.

    A respondent supplies the objects the authority is about to put through its own secret key, so
    the authority should know what it is decrypting. This checks the structure. It cannot check that
    a ciphertext is well formed inside, which is a property of the backend, and a backend whose
    decryption of a malformed ciphertext is undefined is a hazard this interface inherits.
    """
    if not isinstance(response, dict):
        raise ValueError("response must be a mapping")
    if int(response.get("rounds", -1)) != int(record["rounds"]):
        raise ValueError("response round count does not match the issued challenge")
    outs = response.get("outputs")
    if not isinstance(outs, dict):
        raise ValueError("response carries no outputs mapping")
    for op in _OPS:
        v = outs.get(op)
        if not isinstance(v, list) or len(v) != int(record["rounds"]):
            raise ValueError(f"response does not answer every issued round for '{op}'")
        if any(x is None for x in v):
            raise ValueError(f"response contains an empty answer for '{op}'")


def challenge_digest(challenge: dict) -> str:
    """Bind a certificate to THIS challenge without revealing the probes."""
    h = hashlib.sha256()
    h.update(f"{challenge['seed']}:{challenge['rounds']}:{challenge['n_report']}".encode())
    return h.hexdigest()
