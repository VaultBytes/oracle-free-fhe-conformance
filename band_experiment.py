#!/usr/bin/env python3
"""Two production libraries under identical parameters, and how wide the accepted band has to be.

Run it:  python3 band_experiment.py        (needs openfhe and tenseal)

Two questions, one experiment.

First, do independent CKKS implementations agree when nothing differs between them but the code?
The cross-vendor demo runs each engine at its own comfortable settings, which is the right test for
"can the suite score a foreign engine" and the wrong test for "do correct engines agree". Here
OpenFHE and TenSEAL both run at N=8192, scale 2^40, and the same 60-40-40-60 modulus chain, against
the same challenge seed, scored by the same rule.

Second, can a precision band be drawn tight enough to exclude a respondent that never evaluates?
The band has to admit every correct implementation, so its width is at least the disagreement
measured in the first question. The delegate needs one point inside it. This prints both numbers so
a reader can compare them rather than take the argument on trust.
"""
from __future__ import annotations

import sys

import numpy as np

from vbfhe_conformance import (attested_measurements, attested_response, derive_floors,
                               relative_upper_bits)

N, SCALE_BITS, DEPTH, FIRST_MOD = 8192, 40, 2, 60
ROUNDS, N_REPORT, REPS = 4, 64, 25
LAWS = ("add_homomorphism", "plainmul_distributive", "ctmul_distributive")
BASE_SEED = 20260926


def score(be, seed: int) -> dict:
    resp = attested_response(be, seed=seed, rounds=ROUNDS, profile="workload", n_report=N_REPORT)
    return attested_measurements(resp, N=be.N, seed=seed, profile="workload",
                                 scale_bits=be.scale_bits)


def main() -> int:
    try:
        from vbfhe_backend_openfhe import OpenFHEBackend
        from vbfhe_backend_tenseal import TenSEALBackend
    except Exception as exc:                                       # noqa: BLE001
        print(f"this experiment needs both openfhe and tenseal ({type(exc).__name__}: {exc}).")
        print("Neither engine is scored without the other, because the question is whether the two")
        print("agree, and one engine cannot answer it.")
        return 1

    print(__doc__.split("Run it:")[0].strip())
    a = OpenFHEBackend(scale_bits=SCALE_BITS, depth=DEPTH, ring_dim=N, first_mod_size=FIRST_MOD)
    b = TenSEALBackend(ring_dim=N, scale_bits=SCALE_BITS, depth=DEPTH, first_mod_size=FIRST_MOD)
    assert (a.N, a.scale_bits, a.slots) == (b.N, b.scale_bits, b.slots)
    floors, worst, _ = derive_floors(None, 9.0, N=N, scale_bits=SCALE_BITS)
    print(f"\nOpenFHE and TenSEAL, both N={N}, scale 2^{SCALE_BITS}, chain "
          f"{FIRST_MOD}-{'-'.join([str(SCALE_BITS)]*DEPTH)}-{FIRST_MOD}, {a.slots} slots, "
          f"{ROUNDS} rounds, {N_REPORT} examined slots, {REPS} seeds")

    all_laws = LAWS + ("keyswitch_rotation",)
    got = {"openfhe": {k: [] for k in all_laws}, "tenseal": {k: [] for k in all_laws}}
    tops = {k: [] for k in all_laws}
    for i in range(REPS):
        seed = BASE_SEED + i
        ma, mb = score(a, seed), score(b, seed)
        for k in all_laws:
            if k in ma:
                got["openfhe"][k].append(ma[k])
                tops[k].extend(ma["precision_band_top"][k])
            if k in mb:
                got["tenseal"][k].append(mb[k])

    def cell(v):
        if not v:
            return f"{'n/a':>22}"
        v = np.asarray(v)
        return f"{np.median(v):>8.2f} [{np.percentile(v,5):5.2f},{np.percentile(v,95):6.2f}]"

    print(f"\n  {'law':<24}{'OpenFHE':>23}{'TenSEAL':>23}{'|diff|':>8}{'floor':>7}"
          f"{'band top':>10}{'width':>7}")
    widths, diffs = [], []
    for k in all_laws:
        oa, ob = got["openfhe"][k], got["tenseal"][k]
        top = float(np.median(tops[k])) if tops[k] else float("nan")
        width = top - floors[k]
        line = f"  {k:<24}{cell(oa)}{cell(ob)}"
        if oa and ob:
            d = float(np.median(np.abs(np.asarray(oa) - np.asarray(ob))))
            diffs.append(d); line += f"{d:>8.2f}"
        else:
            line += f"{'-':>8}"
        widths.append(width)
        print(line + f"{floors[k]:>7.1f}{top:>10.2f}{width:>7.2f}")

    print(f"\n  worst-case guaranteed precision (a lower bound, not a ceiling): "
          f"{worst['add_homomorphism']:.1f} bits")
    print(f"  the two libraries disagree by a median of {max(diffs):.2f} bits at the same "
          f"parameters,")
    print(f"  and the accepted band is {min(widths):.1f} to {max(widths):.1f} bits wide. A band that "
          f"excluded a")
    print("  respondent choosing its own score would have to be narrower than the disagreement")
    print("  between two correct implementations, and it would then refuse one of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
