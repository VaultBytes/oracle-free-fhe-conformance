#!/usr/bin/env python3
"""The same delegate, against both protocols. It beats one and cannot attempt the other.

Run it:  python3 blind_demo.py

`attack_delegate.py` shows a respondent that performs no homomorphic operation receiving signed
PASS certificates from the attested known-answer protocol. This script shows why, and what fixes
it.

The attested protocol sends the device a SEED. The device derives the probe plaintexts from it,
encrypts under its own key, evaluates, decrypts, and returns decoded numbers. Everything the
delegate needs is therefore handed to it in the clear, and the cheapest correct answer is a
plaintext computation over the examined slots rather than a homomorphic one. The delegate is not
merely indistinguishable from an honest engine, it is cheaper, which is why no threshold, timing
bound or probe-volume argument recovers soundness: raising the cost stresses the honest party.

The blind protocol inverts who holds the key. The AUTHORITY builds the context, encrypts the
probes itself, and sends ciphertexts plus the public evaluation key. The device evaluates without
ever holding the secret and returns ciphertexts. The authority decrypts with a key it never
shared.

A delegate can no longer read the probe values, so it cannot compute the answers cheaply in the
clear. It is left guessing at ciphertexts that must decrypt correctly under a key it does not
hold, which is the assumption CKKS security already rests on.

The authority is now the one decrypting ciphertexts a respondent produced, and publishing a score
derived from those decryptions. That is a decryption interface, so the session is single-shot: an
ephemeral key, the whole challenge issued before any feedback, exactly one response scored, and the
key destroyed afterwards. The controls for all three run below.

This does not bind the computation to a particular device. A delegate may forward the ciphertexts
to a real CKKS library elsewhere and return what comes back, and the certificate then says that
correct CKKS evaluation happened somewhere under the authority's parameters. Custody or a hardware
root of trust is what narrows "somewhere" to "here", and no protocol over this interface supplies
it.
"""
from __future__ import annotations

import sys

import numpy as np

from vbfhe_backend_ckks import SoftwareCKKS
from vbfhe_blind import (BlindSession, blind_evaluate, blind_evaluator, selftest_blindness)
from vbfhe_conformance import derive_floors

SEED = 20260926


def main() -> int:
    print(__doc__.split("Run it:")[0].strip())
    floors, _, _ = derive_floors(SoftwareCKKS(), 9.0)

    _own = SoftwareCKKS()          # the delegate's own context: no access to the authority's key

    def session(i: int) -> BlindSession:
        """A fresh session per respondent. The key is ephemeral, so no two share one."""
        return BlindSession(SoftwareCKKS, seed=SEED + i, rounds=4, n_report=64)

    s0 = session(0)
    dev = blind_evaluator(s0.auth)
    print("\nCONTROLS  each of these must hold, or everything below is decorative")
    print(f"  evaluator holds no secret and refuses to decrypt : {selftest_blindness(dev)}")
    view_ok = s0.selftest()["device_view_hides_answers"]
    print(f"  transmitted challenge carries no expected answers: {view_ok}")
    if not (selftest_blindness(dev) and view_ok):
        print("  CONTROL FAILED, aborting, the comparison below would mean nothing")
        return 1

    def verdict(meas):
        return "PASS" if all(meas[k] >= floors[k] for k in meas) else "FAIL"

    print("\nHONEST device, evaluating blind")
    ch = s0.challenge()
    m = s0.judge(blind_evaluate(dev, ch))
    for k, v in m.items():
        print(f"  [{'PASS' if v >= floors[k] else 'FAIL'}] {k:<24} {v:6.2f}b (floor {floors[k]})")

    print("\nDELEGATE, holding the ciphertexts and no key")
    for i, (label, build) in enumerate((
            ("return an input unchanged",
             lambda c, a: {"blind": True, "rounds": c["rounds"],
                           "outputs": {op: [j["ct_a"] for j in c["work"]]
                                       for op in ("add", "pmul", "cmul")}}),
            ("encrypt a guess of zero",
             # NOT `a`. Handing the delegate the authority's backend hands it the authority's
             # secret, and a delegate that holds the key is not the adversary this protocol is
             # about. It encrypts under its OWN context, which is what a respondent can actually
             # do, and the authority then cannot decrypt the result at all.
             lambda c, a: {"blind": True, "rounds": c["rounds"],
                           "outputs": {op: [_own.encrypt(_own.encode(np.zeros(_own.slots)))
                                            for _ in c["work"]]
                                       for op in ("add", "pmul", "cmul")}}))):
        s = session(i + 1)
        c = s.challenge()
        try:
            mm = s.judge(build(c, s.auth))
            bits = ", ".join(f"{k.split('_')[0]} {v}" for k, v in mm.items())
            print(f"  {label:<28} {verdict(mm):<5} [{bits}]")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {label:<28} REFUSED  {str(exc)[:46]}")
    print(f"  {'answer in float64 + paint noise':<28} UNAVAILABLE")
    print("      the authority wants ciphertexts, and the probe values cannot be read out of")
    print("      the ones it sent, so there is nothing to compute the answer from")

    print("\nSTATE MACHINE, checked rather than described")
    st = session(99)
    st.challenge()
    for label, call in (("re-issue the same challenge", lambda: st.challenge()),
                        ("score a second response", None)):
        if call is None:
            st.judge(blind_evaluate(blind_evaluator(st.auth), st._record))
            call = lambda: st.judge({"blind": True, "rounds": 4, "outputs": {}})
        try:
            call()
            print(f"  {label:<30} ALLOWED   <- the session is not single-shot")
        except PermissionError as exc:
            print(f"  {label:<30} REFUSED   {str(exc)[:44]}")
    print(f"  {'secret destroyed after scoring':<30} "
          f"{'YES' if st.selftest()['key_destroyed_after_scoring'] else 'NO'}")

    print("\nWHAT CHANGED")
    print("  attested : delegate answers in plaintext over the examined slots and passes; forging")
    print("             is CHEAPER than honest evaluation, so cost and timing defences run")
    print("             backwards")
    print("  blind    : forging requires ciphertexts correct under a key the forger lacks, so its")
    print("             cost is honest evaluation's cost, and probe volume becomes usable again")
    print("\n  Not fixed: forwarding the ciphertexts to a real CKKS library elsewhere. That is")
    print("  delegation to something that genuinely does CKKS, and custody is what narrows it.")
    print("  Not proven: that the interface is safe. One non-adaptive query per ephemeral key is")
    print("  a bound on the adversary, not a reduction to a standard notion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
