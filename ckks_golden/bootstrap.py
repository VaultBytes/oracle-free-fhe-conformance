"""CKKS bootstrap precision oracle (OFC primitive P6, Class B / approximate).

The conformance suite needs a *definitional reference* for CKKS bootstrap, which the
repo previously lacked (only a cost model existed). Bootstrap is the FHE hotspot and
the canonical Class-B op: it is approximate by construction, so its conformance is an
ε-envelope (a precision floor), NOT a bit-exact match. This module defines that floor.

CKKS bootstrap = ModRaise -> CoeffToSlot (CtS) -> EvalMod -> SlotToCoeff (StC). CtS/StC
are *exact* homomorphic linear maps (DFT matrices) -- they belong to Class A and reduce
to keyswitch+matvec already covered by the suite. The one genuinely approximate step,
and thus the one that sets bootstrap precision, is **EvalMod**: removing the q*I term
that ModRaise leaves behind by evaluating a polynomial approximation of the (centered)
mod function. This oracle implements EvalMod faithfully and measures its precision in
bits as a function of (approximation degree, modraise range K) -- exactly the data the
OFC profile table needs for ε_{P,bootstrap}.

Approach (minimax / direct-mod, the modern accurate path -- Lee-Lee-Kim-No style rather
than the older sine-only approximation): after ModRaise a slot holds t = m + q*I with
|m| <= r*q (message small) and I an unknown integer in [-K, K]. We want m = t - q*I, i.e.
in normalized coords u = t/q the target is reduce(u) = u - round(u) = m/q, evaluated only
on the union of small intervals [I-r, I+r] around each integer I in [-K, K] (a polynomial
can only be evaluated, so we fit one polynomial that equals the centered-mod target on
exactly those intervals). Degree buys precision; larger K (more levels consumed before
bootstrap) makes it harder. This reproduces real bootstrap behaviour and is provably the
target function (we fit reduce(u) itself), so it is a sound oracle, not a heuristic.
"""
from __future__ import annotations

try:
    import numpy as np
except Exception as e:  # pragma: no cover - numpy is expected in the surrogate env
    raise SystemExit("bootstrap.py needs numpy (same env as ops/research surrogate): " + str(e))


def _eval_points(K: int, r: float, per_interval: int):
    """Sample points u in the union of [I-r, I+r] for I in [-K..K], with target
    values reduce(u) = u - round(u). These are the (input, target) pairs EvalMod
    must satisfy; everything between the intervals is don't-care (the message is
    guaranteed small after a correct ModRaise)."""
    us, ts = [], []
    for I in range(-K, K + 1):
        # avoid the exact half-integer cusps; sample (I-r, I+r)
        local = np.linspace(-r, r, per_interval)
        for d in local:
            us.append(I + d)
            ts.append(d)  # reduce(I+d) = d  (the centered fractional part)
    return np.array(us), np.array(ts)


def fit_evalmod(K: int, degree: int, r: float = 1.0 / 16, per_interval: int = 64):
    """Least-squares Chebyshev fit of the centered-mod target over the modraise
    range. Returns a numpy Chebyshev object P with P(u) ~= reduce(u) on the message
    intervals. (Least-squares over dense samples approximates the minimax fit; good
    enough to *define* the precision floor a conformant chip must meet or beat.)"""
    import warnings
    us, ts = _eval_points(K, r, per_interval)
    dom = [-(K + r) - 0.5, (K + r) + 0.5]
    with warnings.catch_warnings():
        # high-degree Chebyshev lstsq is poorly conditioned; benign for an oracle that
        # only needs to *define* the precision floor, not be the runtime evaluator.
        warnings.simplefilter("ignore", np.polynomial.polyutils.RankWarning)
        P = np.polynomial.chebyshev.Chebyshev.fit(us, ts, deg=degree, domain=dom)
    return P


def evalmod(P, t, q):
    """Apply the fitted EvalMod polynomial: recover m ~= reduce(t/q)*q.
    `t` may be a scalar or array of post-ModRaise slot values; `q` the base modulus."""
    u = np.asarray(t, dtype=float) / q
    return P(u) * q


def precision_bits(K: int, degree: int, r: float = 1.0 / 16,
                   n_test: int = 4000, seed: int = 12345) -> float:
    """Measured EvalMod precision in bits over the message range:
        bits = -log2( max|m_recovered - m_true| / q ).
    This is the candidate ε_{P,bootstrap} the OFC profile table publishes; a
    conformant accelerator must achieve AT LEAST this many bits (one-sided floor)."""
    P = fit_evalmod(K, degree, r)
    rng = np.random.RandomState(seed)
    Is = rng.randint(-K, K + 1, size=n_test)
    fr = rng.uniform(-r, r, size=n_test)          # m/q in [-r, r]
    q = (1 << 50)                                  # representative scale
    t = (Is + fr) * q                              # post-ModRaise slot values
    m_true = fr * q
    m_rec = evalmod(P, t, q)
    max_abs = float(np.max(np.abs(m_rec - m_true)))
    if max_abs == 0.0:
        return float("inf")
    return -np.log2(max_abs / q)


def bootstrap_oracle(coeffs_in, q, K: int, degree: int, r: float = 1.0 / 16):
    """Definitional bootstrap reference on a vector of (already-small) message
    values `coeffs_in` (treated as the post-StC slot messages, |m| <= r*q). Models
    the full pipeline's *observable* effect: ModRaise adds a random q*I per slot
    (the level-0 wraparound bootstrap must undo), EvalMod removes it, and we return
    the recovered messages + the realized error. CtS/StC are exact linear maps
    (Class A) and are identity on the value vector here by construction, so this
    isolates the Class-B precision the standard certifies.

    Returns (recovered, max_abs_err). A conformant device's recovered output must lie
    within the published envelope of `coeffs_in`."""
    P = fit_evalmod(K, degree, r)
    rng = np.random.RandomState(0xB007)
    m = np.asarray(coeffs_in, dtype=float)
    assert np.max(np.abs(m)) <= r * q + 1e-9, "message exceeds modraise interval r*q"
    Is = rng.randint(-K, K + 1, size=m.shape[0])
    t = m + Is * q                       # ModRaise: the q*I bootstrap must strip
    rec = evalmod(P, t, q)               # EvalMod
    return rec, float(np.max(np.abs(rec - m)))


if __name__ == "__main__":
    # Precision floor table: degree x K -> achieved bits. This is the data that
    # populates ε_{P,bootstrap} in the OFC profile table (OFC_SPEC_v0.1 §4).
    print("EvalMod precision floor (bits) -- rows=degree, cols=K (modraise range)")
    Ks = [4, 8, 16]
    print("deg \\ K   " + "   ".join(f"K={k:<3}" for k in Ks))
    for deg in (15, 31, 63, 127):
        row = [precision_bits(K=k, degree=deg) for k in Ks]
        print(f"  {deg:<5}   " + "   ".join(f"{b:5.1f}" for b in row))
    # End-to-end sanity: bootstrap a small message vector and report realized error.
    q = 1 << 50
    msg = [(-(q >> 6)) + (i * (q >> 7)) % (q >> 5) for i in range(8)]
    rec, err = bootstrap_oracle(msg, q, K=8, degree=63)
    print(f"\nbootstrap_oracle: 8 slots, K=8 deg=63 -> max_abs_err/q = {err/q:.2e} "
          f"(~{-np.log2(err/q):.1f} bits)")
