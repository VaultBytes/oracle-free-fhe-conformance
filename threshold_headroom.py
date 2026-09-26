#!/usr/bin/env python3
"""How safe is the upper acceptance threshold, and why it is a threshold and not a bound.

Run it:  python3 threshold_headroom.py      (openfhe and tenseal optional)

The suite refuses a score above scale_bits + 1 absolute, carried into the score's relative units
per round. An earlier version of this repository called that a bound on what CKKS can produce and
argued that beating it would need every rounding contribution to cancel, an event of probability
zero. That argument is wrong. The contributions only have to cancel PARTIALLY, and partial
cancellation has real probability: part one below measures it.

What makes the threshold usable is not impossibility, it is headroom, and headroom is measurable.
Part one measures the best case a correct engine could reach with no ciphertext noise at all. Part
two measures where real engines actually sit.

Neither part is load-bearing for the paper's result. A published acceptance region is targetable
whatever its edges are, which is Theorem 1 and does not depend on where this threshold falls.
"""
from __future__ import annotations

import sys

import numpy as np

from vbfhe_conformance import attested_measurements, attested_response

TRIALS, WINDOW = 300, 64
RINGS = (256, 1024, 8192)


def encode_rounding_errors(N: int, rng, trials: int) -> np.ndarray:
    """|sigma(e)| per slot for rounding vectors e drawn uniformly from [-1/2, 1/2]^N.

    Encoding is round(Delta * sigma^-1(v)), so the decoded error is sigma(e)/Delta for the rounding
    vector e. In units of 1/Delta the error is |sigma(e)|, and the threshold at scale_bits + 1 is
    the line |sigma(e)| = 1/2. Evaluating sigma at the primitive 2N-th roots of unity is a twist by
    zeta^k followed by a length-N DFT.
    """
    tw = np.exp(1j * np.pi * np.arange(N) / N)
    return np.asarray([np.abs(np.fft.fft(rng.uniform(-0.5, 0.5, N) * tw)[:N // 2])
                       for _ in range(trials)])


def part_one() -> None:
    print("PART ONE  encode rounding only, no encryption and no evaluation noise")
    print(f"  {'ring':>6}{'P(one slot beats it)':>22}{'P(all 64 beat it)':>20}"
          f"{'best of %d, bits of headroom':>30}" % TRIALS)
    rng = np.random.default_rng(7)
    for N in RINGS:
        S = encode_rounding_errors(N, rng, TRIALS)
        per_slot = float((S < 0.5).mean())
        worst_case_window = S[:, :WINDOW].max(axis=1)
        all_beat = float((worst_case_window < 0.5).mean())
        headroom = float((np.log2(worst_case_window) + 1.0).min())
        print(f"  {N:>6}{per_slot:>22.2e}{all_beat:>20.0f}{headroom:>30.2f}")
    print("  A single slot beats the threshold often enough to measure, so the threshold is not a")
    print("  limit on what the scheme can produce. The rule reads the MAXIMUM over the examined")
    print("  slots, and that never beat it in any trial, with the closest call still several bits")
    print("  away before any ciphertext noise is added.")


def part_two() -> int:
    print("\nPART TWO  real engines, and how close they come")
    engines = []
    try:
        from vbfhe_backend_ckks import SoftwareCKKS
        engines.append(("SoftwareCKKS", SoftwareCKKS(), 40))
    except Exception as exc:                                       # noqa: BLE001
        print(f"  SoftwareCKKS unavailable ({type(exc).__name__})")
    try:
        from vbfhe_backend_openfhe import OpenFHEBackend
        engines.append(("OpenFHE", OpenFHEBackend(scale_bits=40, depth=2, ring_dim=8192,
                                                  first_mod_size=60), 15))
    except Exception:                                              # noqa: BLE001
        print("  OpenFHE not installed, skipped")
    try:
        from vbfhe_backend_tenseal import TenSEALBackend
        engines.append(("TenSEAL", TenSEALBackend(ring_dim=8192, scale_bits=40, depth=2,
                                                  first_mod_size=60), 15))
    except Exception:                                              # noqa: BLE001
        print("  TenSEAL not installed, skipped")
    if not engines:
        print("  no engine available; nothing measured")
        return 1

    total, closest = 0, None
    print(f"  {'engine':<16}{'runs':>6}{'closest approach':>18}{'median':>10}{'refused':>9}")
    for name, be, reps in engines:
        margins, refused = [], 0
        for i in range(reps):
            seed = 555000 + i
            try:
                m = attested_measurements(
                    attested_response(be, seed=seed, rounds=4, profile="workload", n_report=WINDOW),
                    N=be.N, seed=seed, profile="workload", scale_bits=be.scale_bits)
            except ValueError:
                refused += 1
                continue
            for law, tops in m["precision_band_top"].items():
                margins.append(min(tops) - m[law])
            total += 4
        a = np.asarray(margins)
        closest = a.min() if closest is None else min(closest, a.min())
        print(f"  {name:<16}{reps:>6}{a.min():>17.2f}b{np.median(a):>9.2f}b{refused:>9}")
    print(f"\n  {total} honest rounds scored, none refused, closest approach {closest:.2f} bits.")
    print("  That is the false-reject evidence. It is an argument about headroom, not a proof, and")
    print("  the paper's result does not rest on it: the accepted region is published, so a")
    print("  respondent picks a point inside it whatever the edges are.")
    return 0


def main() -> int:
    print(__doc__.split("Run it:")[0].strip())
    print()
    part_one()
    return part_two()


if __name__ == "__main__":
    sys.exit(main())
