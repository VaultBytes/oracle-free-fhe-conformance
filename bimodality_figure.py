#!/usr/bin/env python3
"""Regenerate the figure: the score is bimodal, the margin below the band mostly is not.

Run it:  python3 bimodality_figure.py [out.pdf]        (needs matplotlib)

The top row is the achieved precision over many runs of one honest engine. It is bimodal in every
law, and the separation is largest for ciphertext multiplication. That is not the engine changing
its mind. The probe amplitude is redrawn each round from {0.5, 2.0, 6.0} and the score is relative
to max|expected|, so the score carries a log2 of the authority's own draw.

The bottom row is the same runs, plotted as the distance from the score to the top of the derived
band for that round. The band carries the same log2 term, so the term cancels and what is left is
closer to a property of the engine. The spread falls by a factor of two to four.

This is the figure for the units fix, not decoration: a conformance metric whose spread is governed
by the authority's draw is not measuring only the engine, and a band that does not carry the same
term is not in the units of the thing it judges.
"""
from __future__ import annotations

import sys

import numpy as np

from vbfhe_backend_ckks import SoftwareCKKS
from vbfhe_conformance import attested_measurements, attested_response, derive_floors

LAWS = (("add_homomorphism", "add homomorphism"),
        ("plainmul_distributive", "plainmul distributive"),
        ("ctmul_distributive", "ctmul distributive"))
RUNS = 300
BASE_SEED = 900000


def collect(be, runs: int):
    score = {k: [] for k, _ in LAWS}
    margin = {k: [] for k, _ in LAWS}
    for i in range(runs):
        seed = BASE_SEED + i
        resp = attested_response(be, seed=seed, rounds=4, profile="workload", n_report=64)
        m = attested_measurements(resp, N=be.N, seed=seed, profile="workload",
                                  scale_bits=be.scale_bits)
        for k, _ in LAWS:
            score[k].append(m[k])
            margin[k].append(min(m["precision_band_top"][k]) - m[k])
    return score, margin


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                       # noqa: BLE001
        print(f"matplotlib is not installed ({type(exc).__name__}); no figure written.")
        return 1

    out = sys.argv[1] if len(sys.argv) > 1 else "bimodal.pdf"
    be = SoftwareCKKS()
    floors, _, _ = derive_floors(be, 9.0)
    print(__doc__.split("Run it:")[0].strip())
    print(f"\nSoftwareCKKS, N={be.N}, scale 2^{be.scale_bits}, {RUNS} runs")
    score, margin = collect(be, RUNS)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for col, (k, title) in enumerate(LAWS):
        s = np.asarray(score[k]); m = np.asarray(margin[k])
        ax = axes[0][col]
        ax.hist(s, bins=40, color="0.35")
        ax.axvline(floors[k], color="k", lw=1.2, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("achieved bits", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"runs (of {RUNS})", fontsize=9)
            ax.annotate(f"floor {floors[k]}", (floors[k], ax.get_ylim()[1] * 0.92),
                        fontsize=8, ha="left")
        ax = axes[1][col]
        ax.hist(m, bins=40, color="0.55")
        ax.set_xlabel("bits below the derived band top", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"runs (of {RUNS})", fontsize=9)
        print(f"  {k:<24} score sd {s.std():5.2f}   margin sd {m.std():5.2f}   "
              f"tighter by {s.std()/m.std():4.1f}x")
    fig.tight_layout()
    fig.savefig(out)
    print(f"\n  figure written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
