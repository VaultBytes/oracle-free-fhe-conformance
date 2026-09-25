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
    path creates -- an IND-CPA-D exposure on the device's own key -- simply does not arise;
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

    Returns the ciphertexts the device is to evaluate, the plaintext weight it multiplies by
    (a plaintext operand is public by construction), and the bookkeeping the authority keeps to
    itself for scoring.
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


def blind_evaluate(dev, challenge: dict) -> dict:
    """DEVICE side. Evaluate foreign ciphertexts with no secret key. Returns ciphertexts."""
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


def challenge_digest(challenge: dict) -> str:
    """Bind a certificate to THIS challenge without revealing the probes."""
    h = hashlib.sha256()
    h.update(f"{challenge['seed']}:{challenge['rounds']}:{challenge['n_report']}".encode())
    return h.hexdigest()
