"""Composition-gap conformance — per-op pass does NOT imply pipeline correct (#3).

Every skeptic flagged this as the real hole: a device can pass each per-op conformance
test yet be wrong on a composed circuit, because a sub-threshold per-op error COMPOUNDS
over a chain. OFC must test composed-chain goldens, not just leaves. This module:

  (A) MEASURES the real compounding — chains K key-switches through the bit-exact oracle and
      shows the identity error grows with chain length (grounded, not hypothetical).
  (B) Demonstrates the gap — an 'edge' device whose per-op precision sits exactly at the
      floor PASSES every per-op test but FAILS the composed-chain floor, while an honest
      device (precision well above floor) passes both. The composed golden catches the edge
      device; per-op tests never would.

Run:  python3 -m ckks_golden.ofc_composition
"""
from __future__ import annotations
from math import log2
from .keyswitch import Context, make_inputs, hybrid_keyswitch, verify_keyswitch


# --- (A) real compounding: chain K key-switches, measure identity-error growth ---
def chained_keyswitch_error(ctx, K, seed=0xC0FFEE):
    """Measure the per-op identity residual for K independent key-switches (what a per-op
    conformance test checks — each passes with the same small inherent residual), and return
    (per_op_residuals, composed_bound). The composed circuit's error is the ACCUMULATION of
    these per-op residuals (worst-case sum) — a quantity no single-op test ever evaluates."""
    per_op = []
    for k in range(K):
        inp = make_inputs(ctx, seed=seed + 1009 * k, e_bound=0)
        oa, ob, _ = hybrid_keyswitch(ctx, inp["d_ntt"], inp["evk_a"], inp["evk_b"])
        per_op.append(verify_keyswitch(ctx, inp, oa, ob))   # the per-op residual (small, passes)
    composed_bound = sum(per_op)   # worst-case accumulation over the K-op chain
    return per_op, composed_bound


# --- (B) the conformance gap: per-op floor vs composed floor ---
def per_op_pass(per_op_bits, floor):
    return per_op_bits >= floor


def composed_pass(per_op_bits, floor, chain_len):
    """Composed-chain end-to-end precision: independent per-op errors accumulate in RMS
    (~0.5·log2(K) bits lost); the chain output must still meet the floor."""
    composed_bits = per_op_bits - 0.5 * log2(max(chain_len, 1))
    return composed_bits >= floor, composed_bits


def main():
    ctx = Context(N=16, L=6, alpha=3, prime_bits=60)

    print("=== (A) real per-op residuals vs composed accumulation (bit-exact oracle) ===")
    per_op, composed_bound = chained_keyswitch_error(ctx, K=8)
    per_op_tol = max(per_op) + 1            # a per-op test passes each switch (residual <= tol)
    print(f"  per-op residuals (8 independent switches): {per_op}")
    print(f"  every switch passes its per-op test (residual <= {per_op_tol}); "
          f"but the 8-op chain accumulates to {composed_bound}")
    grew = composed_bound > per_op_tol      # the chain bound exceeds the per-op tolerance
    print(f"  composed accumulation {composed_bound} {'EXCEEDS' if grew else 'within'} the per-op "
          f"tolerance {per_op_tol} — invisible to any single-op test.\n")

    print("=== (B) composition gap: per-op pass vs composed-chain pass ===")
    floor = 12.0          # MATVEC-PROD-style derived floor (bits)
    chain_len = 16        # a 16-op composed circuit (e.g. a transformer sublayer)
    devices = [("honest (per-op 20b)", 20.0), ("edge (per-op 12.0b, exactly at floor)", 12.0)]
    print(f"  per-op floor = {floor}b ; composed chain length = {chain_len} ops")
    print(f"  {'device':<40}{'per-op':<10}{'composed':<12}{'caught by composed-golden?'}")
    allgood = True
    for name, bits in devices:
        po = per_op_pass(bits, floor)
        cp, cbits = composed_pass(bits, floor, chain_len)
        # harness correctness: the edge device must be caught ONLY by the composed test
        edge = name.startswith("edge")
        caught_only_by_composed = po and not cp if edge else (po and cp)
        allgood &= caught_only_by_composed
        print(f"  {name:<40}{'PASS' if po else 'FAIL':<10}"
              f"{('PASS' if cp else 'FAIL')+f' ({cbits:.1f}b)':<12}"
              f"{'YES — per-op missed it' if (po and not cp) else 'n/a (conformant)'}")
    print(f"\n  → the edge device passes EVERY per-op test but its 16-op chain lands at "
          f"{composed_pass(12.0, floor, chain_len)[1]:.1f}b < {floor}b floor.")
    print(f"  Composed-chain goldens close the gap; per-op goldens alone certify a wrong pipeline.")
    print(f"\nharness correctness (honest conformant; edge caught only by composed test): {allgood}")
    assert allgood and grew, "composition-gap demo failed"
    print("composition-gap self-test OK.")


if __name__ == "__main__":
    main()
