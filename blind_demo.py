#!/usr/bin/env python3
"""The same delegate, against both protocols. It beats one and cannot attempt the other.

Run it:  python3 blind_demo.py

`attack_delegate.py` shows a respondent that performs no homomorphic operation receiving signed
PASS certificates from the attested known-answer protocol. This script shows why, and what fixes
it.

The attested protocol sends the device a SEED. The device derives the probe plaintexts from it,
encrypts under its own key, evaluates, decrypts, and returns decoded numbers. Everything the
delegate needs is therefore handed to it in the clear, and the cheapest correct answer is an O(n)
plaintext computation rather than an O(ell * N log N) homomorphic one. The delegate is not merely
indistinguishable from an honest engine, it is faster, which is why no threshold, timing bound or
probe-volume argument recovers soundness: raising the cost stresses the honest party.

The blind protocol inverts who holds the key. The AUTHORITY builds the context, encrypts the
probes itself, and sends ciphertexts plus the public evaluation key. The device evaluates without
ever holding the secret and returns ciphertexts. The authority decrypts with a key it never
shared.

A delegate can no longer read the probe values, so it cannot compute the answers cheaply in the
clear. It is left guessing at ciphertexts that must decrypt correctly under a key it does not
hold, which is the assumption CKKS security already rests on.

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
from vbfhe_blind import (blind_evaluate, blind_evaluator, issue_blind_challenge, judge_blind,
                         selftest_blindness)
from vbfhe_conformance import derive_floors

SEED = 20260926


def main() -> int:
    print(__doc__.split("Run it:")[0].strip())
    auth = SoftwareCKKS()
    floors, _, _ = derive_floors(auth, 9.0)
    ch = issue_blind_challenge(auth, seed=SEED, rounds=4, n_report=64)

    dev = blind_evaluator(auth)
    print("\nCONTROL  the evaluator must be unable to decrypt, or the rest is decorative")
    print(f"  secret removed and decrypt refuses : {selftest_blindness(dev)}")
    if not selftest_blindness(dev):
        print("  CONTROL FAILED -- aborting, the comparison below would mean nothing")
        return 1

    def verdict(meas):
        return "PASS" if all(meas[k] >= floors[k] for k in meas) else "FAIL"

    print("\nHONEST device, evaluating blind")
    m = judge_blind(auth, ch, blind_evaluate(dev, ch))
    for k, v in m.items():
        print(f"  [{'PASS' if v >= floors[k] else 'FAIL'}] {k:<24} {v:6.2f}b (floor {floors[k]})")

    print("\nDELEGATE, holding the ciphertexts and no key")
    zeros = auth.encode(np.zeros(auth.slots))
    strategies = {
        "return an input unchanged": lambda: {
            "blind": True, "rounds": ch["rounds"],
            "outputs": {op: [j["ct_a"] for j in ch["work"]] for op in ("add", "pmul", "cmul")}},
        "encrypt a guess of zero": lambda: {
            "blind": True, "rounds": ch["rounds"],
            "outputs": {op: [auth.encrypt(zeros) for _ in ch["work"]]
                        for op in ("add", "pmul", "cmul")}},
    }
    for label, build in strategies.items():
        try:
            mm = judge_blind(auth, ch, build())
            bits = ", ".join(f"{k.split('_')[0]} {v}" for k, v in mm.items())
            print(f"  {label:<28} {verdict(mm):<5} [{bits}]")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {label:<28} REFUSED  {str(exc)[:46]}")
    print(f"  {'answer in float64 + paint noise':<28} UNAVAILABLE")
    print("      the authority wants ciphertexts, and the probe values cannot be read out of")
    print("      the ones it sent, so there is nothing to compute the answer from")

    print("\nWHAT CHANGED")
    print("  attested : delegate answers in O(n) plaintext flops and passes; forging is CHEAPER")
    print("             than honest evaluation, so cost and timing defences run backwards")
    print("  blind    : forging requires ciphertexts correct under a key the forger lacks, so its")
    print("             cost is honest evaluation's cost, and probe volume becomes usable again")
    print("\n  Not fixed: forwarding the ciphertexts to a real CKKS library elsewhere. That is")
    print("  delegation to something that genuinely does CKKS, and custody is what narrows it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
