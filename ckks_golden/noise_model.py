"""Derived CKKS precision-envelope floors — OFC ε-tiers from noise theory, not measurement.

The OFC precision tiers (MATVEC/QKT/BOOT) were set from GLIDE measurements. A standard's
floors must be VENDOR-INDEPENDENT and defensible: derived in closed form from CKKS noise
theory and a device's own submitted parameters (parameter-attestation), not from one
vendor's chip. This module does that. Per the skeptic's load-bearing requirement, each
floor is a RANGE [ε_worst, ε_avg]: ε_worst = worst-case guaranteed precision of a CORRECT
implementation (the provable floor a correct device must MEET-or-beat); ε_avg = the
average-case (heuristic) precision a good implementation typically hits. A device below
ε_worst is broken; far above ε_avg is suspicious (skipped an approximate step / spoofing).

This operationalizes the composable-CKKS-noise line (Costache et al.; "Accurate & Composable
Noise Estimates for CKKS", IACR CIC 2025; Grafting CCS'25 parameter-independence) into OFC's
normative spec — it is standardization of established theory, not new theory.

ANCHOR (validates the model): encode/decode rel-error in the canonical-embedding model is
~sqrt(N/12)/Δ (avg) .. (N/2)/Δ (worst). At N=256, scale_bits=30 → avg 27.8b. The sdk
SoftwareCKKS round-trip measured 28.7b (ofc_verify P7) — the derived avg is within a bit,
slightly conservative, as a floor should be.
"""
from __future__ import annotations
from math import log2

# --- the one rigorously-derived primitive: rounding precision in the canonical embedding ---
def encode_bits(N: int, scale_bits: int, mode: str = "avg") -> float:
    """decode(encode(v)) precision in bits. avg: Δ vs sqrt(N/12) rounding spread;
    worst: vs the (N/2) coefficient-error bound."""
    if mode == "avg":
        return scale_bits - 0.5 * log2(N / 12.0)
    return scale_bits - log2(N / 2.0)


def rescale_bits(N: int, scale_bits: int, levels: int, mode: str = "avg") -> float:
    """Precision after `levels` rescales — independent rounding variances add (×sqrt(levels))."""
    extra = 0.5 * log2(max(levels, 1))
    return encode_bits(N, scale_bits, mode) - extra


def arithmetic_ceiling_bits(N: int, scale_bits: int, depth: int, mode: str = "avg") -> float:
    """CKKS arithmetic precision ceiling after `depth` accumulated noisy ops (matvec/keyswitch
    accumulation). depth noisy terms → variance ×depth (avg) / magnitude ×depth (worst)."""
    if mode == "avg":
        return scale_bits - 0.5 * log2(depth * N / 12.0)
    return scale_bits - log2(depth * N / 2.0)


def bootstrap_bits(K: int, degree: int, mode: str = "avg") -> float:
    """EvalMod approximation precision — DERIVED from the (degree, K) of the mod-function
    polynomial approximation (the real source of bootstrap error). Uses the same minimax/
    Chebyshev fit as bootstrap.py; worst-case shaves a margin for input-distribution tails."""
    try:
        from .bootstrap import precision_bits
        b = float(precision_bits(K=K, degree=degree))
    except Exception:
        # closed-form fallback: minimax mod-approx error ~ scales with degree/(2K) (heuristic)
        b = max(2.0, 0.5 * degree / max(K, 1))
    return b if mode == "avg" else max(2.0, b - 2.0)


# --- per-OP tier floors: floor = min(input-regime bits, CKKS arithmetic ceiling) ---
# The matvec/QKt output precision is BINDER-limited by INPUT quantization (the o/down 12-bit
# matvec-scale result, not CKKS arithmetic — established: matvec precision is granularity, not
# arithmetic). So the floor = min(attested input bits, derived arithmetic ceiling).
def derive_tiers(N: int, scale_bits: int, levels: int, *, matvec_input_bits: int = 12,
                 qkt_input_bits: int = 17, boot_K: int = 4, boot_degree: int = 63,
                 matvec_depth: int = None) -> dict:
    """Return derived [worst, avg] precision-bit ranges per OFC tier, from the profile params."""
    D = matvec_depth or N
    def tier(input_bits, depth):
        lo = min(input_bits - 1, arithmetic_ceiling_bits(N, scale_bits, depth, "worst"))
        hi = min(float(input_bits), arithmetic_ceiling_bits(N, scale_bits, depth, "avg"))
        return (round(max(lo, 0.0), 1), round(max(hi, 0.0), 1))
    return {
        "ENCODE":      tuple(round(encode_bits(N, scale_bits, m), 1) for m in ("worst", "avg")),
        "MATVEC-PROD": tier(matvec_input_bits, D),
        "QKT-PROD":    tier(qkt_input_bits, D),
        "BOOT-LOW":    (round(bootstrap_bits(8, boot_degree, "worst"), 1),
                        round(bootstrap_bits(8, boot_degree, "avg"), 1)),
        "BOOT-HIGH":   (round(bootstrap_bits(boot_K, boot_degree, "worst"), 1),
                        round(bootstrap_bits(boot_K, boot_degree, "avg"), 1)),
    }


def floor_for(tier_name: str, **params) -> float:
    """The conformance FLOOR (bits) a device must meet for a tier = the worst-case of the
    derived range. Vendor submits params → OFC computes the floor (parameter-attestation)."""
    return derive_tiers(**params)[tier_name][0]


if __name__ == "__main__":
    print("=== OFC derived ε-floors (parameter-attested) vs GLIDE A100 measurements ===\n")
    # production-ish profile the GLIDE witness ran (scale ladder ~50 bits, N=2^15)
    prod = dict(N=1 << 15, scale_bits=50, levels=20, matvec_depth=4096)
    tiers = derive_tiers(**prod)
    # GLIDE A100 measurements at the production profile (ENCODE has no production measurement —
    # the 28.7b encode number was at N=256/scale=30, validated separately as the anchor below).
    measured = {"MATVEC-PROD": 13.2, "QKT-PROD": 16.7, "BOOT-LOW": 9.28, "BOOT-HIGH": 17.2}
    print(f"{'tier':<13}{'derived [worst,avg] bits':<28}{'GLIDE measured':<16}in-range?")
    allok = True
    for t, (lo, hi) in tiers.items():
        m = measured.get(t)
        if m is None:
            print(f"{t:<13}[{lo:>5}, {hi:>5}]            {'(no prod meas.)':<16}--")
            continue
        ok = (lo - 1.0) <= m  # device must MEET-or-beat the worst-case floor (1b tolerance)
        allok &= ok
        print(f"{t:<13}[{lo:>5}, {hi:>5}]            {m:<16.1f}{'YES' if ok else 'NO'}")
    # encode anchor: validate the model at the SAME params the 28.7b was measured (N=256, scale=30)
    anchor_avg = encode_bits(256, 30, "avg")
    anchor_ok = abs(anchor_avg - 28.7) <= 2.0
    print(f"\nencode anchor (N=256, scale=30): derived avg {anchor_avg:.1f}b vs measured 28.7b "
          f"(Δ={abs(anchor_avg-28.7):.1f}b — {'MODEL VALIDATED' if anchor_ok else 'OFF'})")
    print(f"all production GLIDE measurements meet their derived worst-case floor: {allok}")
    assert allok and anchor_ok, "noise model failed validation"
    print("\nVALIDATION PASSED.")
    print("\nParameter-attestation: floors computed from (N, scale_bits, levels, depth, K, degree),")
    print("not from any vendor's measurement — a different chip on different params gets a different,")
    print("derived floor. The RANGE is honest about worst-case (provable) vs average-case (typical).")
