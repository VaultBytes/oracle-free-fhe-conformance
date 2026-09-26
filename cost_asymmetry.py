#!/usr/bin/env python3
"""How much cheaper is the delegate than the honest engine, and does that depend on the exam size?

Run it:  python3 cost_asymmetry.py         (needs openfhe)

The suite's soundness argument turns on a cost comparison, so the comparison should be measured
across the parameter that a sceptic would suspect of carrying it. That parameter is the number of
slots the authority examines. The authority reports 64 of 4096 by default, and a delegate only has
to produce the slots that are read, so a single ratio quoted at 64 slots could be an artefact of
sparse reporting rather than a property of the schemes.

So this measures both sides across the whole range, from 64 slots to the full 4096.

What is timed. HONEST is one probe round on OpenFHE: three encryptions, an addition, a plaintext
multiply with its rescale, a ciphertext multiply with its rescale, one rotation, and the four
decryptions the protocol asks for. EVAL ONLY is the same round without the encryptions and
decryptions, because a respondent that already holds ciphertexts pays only that part. DELEGATE is
what the attack actually does: derive the probes from the public seed with numpy, compute the four
answers in float64 over the examined slots, and paint Gaussian noise onto them.

Key generation and context setup are excluded from both. They happen once per session and would
favour the honest side of the comparison if included, so leaving them out is the conservative
choice.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from vbfhe_conformance import _ks_step, _probe_vectors, _slot_window

N, SCALE_BITS, DEPTH, FIRST_MOD = 8192, 40, 2, 60
REPS = 20
SEED = 20260926


def _median_ms(fn, reps: int) -> tuple[float, float, float]:
    """Median, 5th and 95th percentile wall time in milliseconds, after one warm-up call."""
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(t)
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))


def delegate_round(seed: int, r: int, S: int, n: int, target_bits: float) -> None:
    """Exactly the work attack_delegate.py does for one round, and nothing else."""
    probes = np.random.default_rng(seed)
    paint = np.random.default_rng(0xBADC0FFEE)
    a, b, c, w = _probe_vectors(probes, S, "workload")
    idx = _slot_window(seed, r, S, n)
    # Restrict to the examined slots BEFORE doing the arithmetic. Deriving the probe vectors is
    # O(S) and unavoidable, since the random stream is defined over the whole slot count, but the
    # answers are only needed where the authority looks. Computing all S and then slicing would
    # overstate the delegate's cost, and the comparison is the point of this script.
    k = _ks_step(seed, r)
    ai, bi, ci = a[idx], b[idx], c[idx]
    wi, roti = w[idx], a[(idx + k) % S]
    for vec in (ai + bi, (ai + bi) * wi, (ai + bi) * ci, roti):
        v = np.asarray(vec, dtype=float)
        e = paint.normal(0.0, 1.0, v.shape)
        e = e / (np.max(np.abs(e)) or 1.0)
        v = v + e * (np.max(np.abs(v)) * 2.0 ** (-target_bits))
        [float(x) for x in v]


def main() -> int:
    try:
        from vbfhe_backend_openfhe import OpenFHEBackend
    except Exception as exc:                                       # noqa: BLE001
        print(f"openfhe is not installed ({type(exc).__name__}), so the honest side cannot be")
        print("measured here. This script states no ratio without both sides of it.")
        return 1

    print(__doc__.split("Run it:")[0].strip())
    be = OpenFHEBackend(scale_bits=SCALE_BITS, depth=DEPTH, ring_dim=N, first_mod_size=FIRST_MOD)
    S = be.slots
    print(f"\nOpenFHE, N={be.N}, scale 2^{be.scale_bits}, depth {DEPTH}, {S} slots, "
          f"{REPS} repetitions, numpy {np.__version__}, python {sys.version.split()[0]}")

    rng = np.random.default_rng(SEED)
    a, b, c, w = _probe_vectors(rng, S, "workload")
    ea, eb, ec = be.encrypt(be.encode(a)), be.encrypt(be.encode(b)), be.encrypt(be.encode(c))
    pw = be.encode(w)

    def honest_full():
        x, y, z = be.encrypt(be.encode(a)), be.encrypt(be.encode(b)), be.encrypt(be.encode(c))
        s1 = be.add(x, y)
        be.decode_ct(s1, be.delta)
        be.decode_ct(be.rescale(be.mul_plain(s1, pw)), be.delta)
        be.decode_ct(be.rescale(be.mul(s1, z)), be.delta)
        be.decode_ct(be.rotate(x, 3), be.delta)

    def honest_eval():
        s1 = be.add(ea, eb)
        be.rescale(be.mul_plain(s1, pw))
        be.rescale(be.mul(s1, ec))
        be.rotate(ea, 3)

    hf = _median_ms(honest_full, REPS)
    he = _median_ms(honest_eval, REPS)
    print(f"\n  honest, one round, encrypt+evaluate+decrypt : {hf[0]:8.2f} ms  [{hf[1]:.2f}, {hf[2]:.2f}]")
    print(f"  honest, one round, evaluate only           : {he[0]:8.2f} ms  [{he[1]:.2f}, {he[2]:.2f}]")
    print("  neither depends on how many slots the authority reads")

    print(f"\n  {'examined slots':>14}  {'delegate (ms)':>14}  {'vs full':>9}  {'vs eval only':>13}")
    rows = []
    for n in (64, 256, 1024, S):
        d = _median_ms(lambda n=n: delegate_round(SEED, 0, S, n, 24.0), REPS)
        rows.append((n, d[0], hf[0] / d[0], he[0] / d[0]))
        print(f"  {n:>14}  {d[0]:>14.4f}  {hf[0]/d[0]:>8.0f}x  {he[0]/d[0]:>12.0f}x")

    worst = min(r[3] for r in rows)
    print(f"\n  The delegate stays cheaper across the whole range. At full packing, where it "
          f"computes\n  every slot the ring holds, it is still {worst:.0f}x cheaper than evaluation "
          f"alone.")
    print("  The honest cost is flat in the exam size because a ciphertext operation touches the")
    print("  whole ring whatever the authority intends to read, so widening the exam raises the")
    print("  delegate's cost and leaves the honest engine's where it was. That is the direction")
    print("  that makes a probe-volume requirement bind the honest party first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
