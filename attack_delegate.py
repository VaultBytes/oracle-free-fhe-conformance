#!/usr/bin/env python3
"""A respondent that performs NO homomorphic encryption and passes this conformance suite.

Run it:  python3 attack_delegate.py

This is shipped deliberately. The suite's own claim is bounded by what this script demonstrates,
and a reader should not have to take that on trust. It is the evidence for the limit stated in
README.md, not a defect report against some other version.

WHAT THE DELEGATE DOES
  * receives the authority's fresh single-use seed;
  * re-derives the challenge probes with numpy, because the derivation is public code in this
    very repository;
  * computes the answers in float64, a + b, (a+b)*w, (a+b)*c, and a cyclic roll for the
    rotation, with no ring, no ciphertext, no key, and no CKKS anywhere;
  * adds Gaussian noise scaled to land at whatever precision it likes;
  * returns the decoded values.

WHAT THE AUTHORITY DOES
  * re-derives the same probes, compares against answers it computed independently, applies
    noise-theoretic floors, and issues a signed PASS.

WHY THE OBVIOUS DEFENCE DOES NOT CLOSE IT
  CKKS is approximate, so an honest score should lie below an upper bound as well as above a
  floor, and a respondent answering too accurately should be refused. The suite now derives that
  upper bound rather than hand-setting it: the encoding quantises at the scale, so absolute error
  below 2^-(scale_bits+1) is not producible, and because the score is relative to max|expected| the
  bound carries a log2 of that same dynamic range. The authority holds the probes, so it computes
  the bound per round and refuses any round above it.

  That band is real and it is why the first three rows below fail. It is also the strongest form
  of this defence, and it does not close the hole. The band has to admit every correct
  implementation, the accepted interval at these parameters is about 25 bits wide, and the
  delegate needs one point inside it. Lowering the target by a few bits costs nothing.

  Simulating CKKS noise convincingly is not hard, because CKKS decoded error is a sum of many
  independent rounding terms pushed through the canonical embedding, so it is close to Gaussian
  by construction.

  One transcript-level statistic looked like it would separate a painter, and does not survive
  contact with a production library. The authority evaluates pmul on the very ciphertext that
  produced add, so an honest engine's pmul error necessarily carries w * err_add, while a
  fabricator that answers each probe independently does not. The certificate reports that
  correlation as `error_coupling`. Measured:

      honest SoftwareCKKS   r = 0.958     (our own reference implementation)
      honest OpenFHE        r = 0.124     (a production library)
      this painter @ 47b    r = 0.136    , ABOVE honest OpenFHE
      this painter @ 31b    r = 0.004

  So the separation is clean against the reference and absent against OpenFHE, where an honest
  engine scores below a delegate. It is reported as a diagnostic and deliberately not enforced.
  A delegate that wanted to defeat it anyway would carry one error vector per probe ciphertext
  and push it through the plaintext circuit, which is still O(n), error algebra rather than
  error magnitude, and more than three lines, but not more than a morning.

THE POINT
  Every correct CKKS implementation returns the same decoded values, to within the precision its
  parameters permit. A protocol that observes only decoded values therefore cannot tell which
  implementation produced them. Worse, producing those values by other means is CHEAPER than
  evaluating the circuit homomorphically, so no cost-based or throughput-based defence works
  either: the delegate is faster than the honest engine, not slower.

  Soundness against this adversary requires moving the transcript into the ciphertext domain 
  verifiable FHE, a hardware root of trust, or custody of the device by the auditor. It cannot be
  recovered by tightening a precision band.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from vbfhe_conformance import _ks_step, _probe_vectors, _slot_window
from vbfhe_server import ConformanceServer

N, SCALE_BITS = 8192, 40


def delegate(server: ConformanceServer, target_bits: float | None) -> dict:
    """Answer the authority's challenge without doing any FHE at all."""
    ch = server.issue_challenge(profile="workload", rounds=4)
    seed, rounds, n_report = ch["seed"], ch["rounds"], ch["n_report"]

    probes = np.random.default_rng(seed)        # mirrors the authority's own derivation
    paint = np.random.default_rng(0xBADC0FFEE)  # separate stream, so probes stay in step
    slots = N // 2
    out: dict[str, list] = {"add": [], "pmul": [], "cmul": [], "ks": []}

    for r in range(rounds):
        a, b, c, w = _probe_vectors(probes, slots, "workload")
        # The authority now chooses WHICH slots it examines, and issues the count. Both are derived
        # from the seed by public code, so the delegate simply derives them too. Binding the window
        # stopped an engine that was correct on a fixed prefix; it does not touch delegation.
        idx = _slot_window(seed, r, slots, n_report)
        answers = {
            "add":  a + b,
            "pmul": (a + b) * w,
            "cmul": (a + b) * c,
            "ks":   np.roll(a, -_ks_step(seed, r)),
        }
        for name, vec in answers.items():
            v = np.asarray(vec, dtype=float)[idx]
            if target_bits is not None:
                e = paint.normal(0.0, 1.0, v.shape)
                e = e / (np.max(np.abs(e)) or 1.0)
                v = v + e * (np.max(np.abs(v)) * 2.0 ** (-target_bits))
            out[name].append([float(x) for x in v])

    request = {
        "challenge_id": ch["challenge_id"], "N": N, "scale_bits": SCALE_BITS,
        "backend_class": "NotAnFHEEngineAtAll", "slots": slots,
        "response": {"attested": True, "rounds": rounds, "profile": "workload",
                     "n_report": n_report, "rotation_supported": True, "outputs": out},
    }
    return json.loads(server.judge_attested(request))


CHALLENGES = 40


def main() -> int:
    print(__doc__.split("WHAT THE DELEGATE DOES")[0].strip())
    print(f"\nring N={N}, scale 2^{SCALE_BITS}, {CHALLENGES} independent challenges per target\n")
    trials = (("exact float64, no noise", None),
              ("painted noise @ 47.9 bits", 47.9),
              ("painted noise @ 44 bits", 44.0),
              ("painted noise @ 42 bits", 42.0),
              ("painted noise @ 40 bits", 40.0),
              ("painted noise @ 31 bits", 31.0),
              ("painted noise @ 24 bits", 24.0),
              ("painted noise @ 20 bits", 20.0))
    print(f"  {'respondent':<28}{'signed PASS':>12}   reason when refused")
    for label, target in trials:
        ok, why = 0, ""
        for _ in range(CHALLENGES):
            try:
                cert = delegate(ConformanceServer(), target)
                ok += cert["verdict"] == "PASS"
            except Exception as exc:                              # noqa: BLE001
                why = why or str(exc).split(":")[0]
        rate = f"{ok}/{CHALLENGES}"
        print(f"  {label:<28}{rate:>12}   {why}")

    print("\n  None of these performed a single homomorphic operation. Honest OpenFHE scores")
    print("  between 21.6 and 25.9 bits at these parameters across observed runs, so the delegate")
    print("  accepted every time at 40 bits reports fourteen bits more precision than the best the")
    print("  production library it imitates managed, and more than any run of it we recorded.")
    print("\n  The refusals are the upper acceptance threshold doing its job. It is derived from")
    print("  the scale, applied per round against that round's own dynamic range, and it refuses a")
    print("  score no correct engine has been observed to reach. The 42-bit row sits on the edge,")
    print("  which is why it depends on the amplitude the authority happened to draw.")
    print("\n  What the threshold rules out is a delegate that aims too high. Aiming lower costs one")
    print("  constant. No threshold helps here whatever its value, because the accepted region is")
    print("  published and the delegate picks a point inside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
