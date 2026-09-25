"""FHE-ACVP: adversarial conformance — make the OFC test attack-resistant (#2).

A naive golden-vector suite is not an authoritative standard: a malicious/sandbagging
accelerator can PASS it without computing correctly. This module adds the NIST-ACVP-style
defenses (server-side blinded challenges withheld until test time) PLUS the OFC-specific
per-stage-manifest unforgeability, and demonstrates that three concrete cheating-chip
classes — which PASS the fixed-seed test — are CAUGHT:

  1. golden-cache  — caches the published goldens keyed by the public seed; returns them
     regardless of input. Caught by a BLINDED CHALLENGE: the certifier issues a fresh random
     seed at test time (not the catalog seed); the cache returns the stale answer → mismatch.
  2. replay        — knows only final outputs (replays a prior cert); cannot produce the
     intermediate per-stage digests. Caught by PER-STAGE MANIFEST completeness (a valid
     multi-stage sha256 chain is unforgeable without executing each stage).
  3. precision-spoof — self-reports a high Class-B precision it cannot achieve for the params.
     Caught by an UPPER-PLAUSIBILITY bound from the derived noise model (claimed bits beyond
     the average-case ceiling ⇒ it skipped the approximate step / is lying).
  4. sandbag (NEW) — the economically-rational attack a real vendor would actually deploy: a
     chip that computes correctly AT FULL PRECISION whenever it detects a test-shaped input,
     but undercomputes (cheaper low-degree EvalMod ⇒ fewer multiplies, fewer levels, lower
     power) on real high-entropy workloads. It PASSES all three checks above — it answers the
     blinded NTT challenge honestly, emits every stage digest, and *reports* a plausible
     precision — because it does the full work when it thinks it is being watched. Caught by
     two combined defenses: (i) INDISTINGUISHABLE challenges drawn from the real ciphertext
     distribution (full ModRaise band), so the chip's "am-I-being-tested?" detector cannot
     separate audit from production; and (ii) a MEASURED-precision-floor audit — the certifier
     measures the device's *achieved* bits on those workload-shaped challenges and compares to
     the noise model's WORST-CASE floor, not the device's self-report. A sandbagger's achieved
     precision collapses below the floor on the workload band ⇒ caught. This fuses axis #2
     (adversarial) with axis #4 (principled tiers): the noise envelope becomes a live audit,
     not a static ceiling. The demo also shows a NAIVE (narrow-band) audit MISSES the sandbag —
     the challenge distribution is what makes the defense sound.

Honest device = the reference (recomputes for ANY seed, emits all stages, real precision) →
conformant on all axes. Run:  python3 -m ckks_golden.ofc_adversarial
"""
from __future__ import annotations
import hashlib
from math import log2
import numpy as np
from .keyswitch import Context, make_inputs, hybrid_keyswitch, _det_poly
from .noise_model import bootstrap_bits
from .bootstrap import precision_bits, bootstrap_oracle


def _h(limbs):
    m = hashlib.sha256()
    for limb in limbs:
        for w in limb:
            m.update(int(w).to_bytes(8, "little"))
    return m.hexdigest()


# --- honest device: recomputes correctly for any seed, emits every stage, real precision ---
class HonestDevice:
    name = "honest-reference"

    def ntt_for_seed(self, ctx, seed, qi):
        return ctx.fwd(_det_poly(seed, ctx.N, qi), qi)

    def keyswitch_stage_digests(self, ctx, seed):
        inp = make_inputs(ctx, seed=seed, e_bound=0)
        snaps = {}
        hybrid_keyswitch(ctx, inp["d_ntt"], inp["evk_a"], inp["evk_b"],
                         emit=lambda name, limbs: snaps.setdefault(name, _h(limbs)))
        return snaps  # all stages present

    def bootstrap_reported_bits(self, K, degree):
        return float(precision_bits(K=K, degree=degree))  # the real, measured precision

    def bootstrap_achieved_bits(self, messages, q, K, degree):
        """ACTUAL precision the device delivers on `messages` — what the certifier
        MEASURES (vs the self-report above). The honest device runs the full-degree
        EvalMod on every input, so its achieved bits hold across any input band."""
        _, err = bootstrap_oracle(messages, q, K, degree)
        return float("inf") if err == 0.0 else -log2(err / q)


# --- cheat 1: caches the public-seed goldens; ignores the actual input seed ---
class GoldenCacheDevice(HonestDevice):
    name = "cheat:golden-cache"

    def __init__(self, ctx, public_seed):
        self._cache = {qi: ctx.fwd(_det_poly(public_seed, ctx.N, qi), qi) for qi in ctx.q}

    def ntt_for_seed(self, ctx, seed, qi):
        return self._cache[qi]  # returns the CACHED public-seed answer for any seed


# --- cheat 2: only has final keyswitch outputs (replayed); no intermediate stages ---
class ReplayDevice(HonestDevice):
    name = "cheat:replay"

    def keyswitch_stage_digests(self, ctx, seed):
        full = super().keyswitch_stage_digests(ctx, seed)
        # a replay device that stored only the final ciphertext can't emit the intermediates
        return {k: v for k, v in full.items() if k.startswith("s5_")}


# --- cheat 3: self-reports an impossible precision for the params ---
class PrecisionSpoofDevice(HonestDevice):
    name = "cheat:precision-spoof"

    def bootstrap_reported_bits(self, K, degree):
        return 30.0  # claims 30b when deg=63/K=4 can only give ~17b


# --- cheat 4: sandbags — full precision iff it thinks it is being tested, else cheap ---
class SandbagDevice(HonestDevice):
    """Computes at full degree on TEST-shaped inputs (the historical narrow-band goldens),
    but at a cheap low degree on real WORKLOAD-shaped inputs (the full ModRaise band). Reports
    a plausible precision and answers the blinded NTT challenge honestly, so it passes axes a–d.
    Its detector = the band the input occupies: legacy goldens crowd |m| <= r*q/8; production
    ciphertexts fill the whole interval."""
    name = "cheat:sandbag"
    cheap_degree = 15            # ~3.7b — saves ~75% of the EvalMod multiplies vs degree 63
    TEST_BAND_FRAC = 1.0 / 8.0   # what it believes "a test input" looks like

    def __init__(self, r=1.0 / 16):
        self._r = r

    def _looks_like_test(self, messages, q):
        return float(np.max(np.abs(messages))) <= self._r * q * self.TEST_BAND_FRAC * 1.05

    def bootstrap_achieved_bits(self, messages, q, K, degree):
        deg = degree if self._looks_like_test(messages, q) else self.cheap_degree
        _, err = bootstrap_oracle(messages, q, K, deg)
        return float("inf") if err == 0.0 else -log2(err / q)


def _challenge_messages(seed, n, q, lo_frac, hi_frac, r=1.0 / 16):
    """Deterministic bootstrap challenge messages in the band [lo_frac, hi_frac]·r·q.
    lo_frac=0 → narrow legacy band (a NAIVE audit); lo_frac near hi_frac=1 → full
    workload band (indistinguishable from production ciphertexts)."""
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    mag = rng.uniform(lo_frac, hi_frac, size=n) * r * q
    sign = rng.choice((-1.0, 1.0), size=n)
    return mag * sign


# --- the adversarial certifier ---
def certify(device, ctx, public_seed, challenge_seed, boot_K=4, boot_degree=63,
            audit_band=(0.6, 1.0), n_audit=2000, q_boot=1 << 50):
    """Run the five attack-resistant checks. Returns dict of axis -> pass/fail.

    audit_band = the [lo,hi]·r·q fraction the MEASURED-precision audit (axis e) draws from.
    Default (0.6, 1.0) = the full workload band (indistinguishable from production); pass
    (0.0, 0.125) to model a NAIVE narrow-band audit a sandbagger evades."""
    q0 = ctx.q[0]
    # (a) fixed public-seed NTT (the naive test every device passes)
    ref_pub = ctx.fwd(_det_poly(public_seed, ctx.N, q0), q0)
    pub_ok = device.ntt_for_seed(ctx, public_seed, q0) == ref_pub
    # (b) BLINDED challenge-seed NTT (fresh seed withheld until test time)
    ref_chal = ctx.fwd(_det_poly(challenge_seed, ctx.N, q0), q0)
    chal_ok = device.ntt_for_seed(ctx, challenge_seed, q0) == ref_chal
    # (c) per-stage manifest completeness (unforgeable multi-stage chain)
    honest_stages = HonestDevice().keyswitch_stage_digests(ctx, challenge_seed)
    dev_stages = device.keyswitch_stage_digests(ctx, challenge_seed)
    stages_ok = all(dev_stages.get(k) == v for k, v in honest_stages.items())
    # (d) precision plausibility: claimed Class-B bits must not exceed the derived avg ceiling
    avg_ceiling = bootstrap_bits(boot_K, boot_degree, "avg")
    claimed = device.bootstrap_reported_bits(boot_K, boot_degree)
    prec_ok = claimed <= avg_ceiling + 1.0   # > avg+1b for these params is implausible
    # (e) MEASURED precision floor on workload-distribution challenges (catches sandbagging):
    #     the certifier measures ACHIEVED bits — not the self-report — on inputs drawn from the
    #     real ciphertext band, and requires them at/above the noise model's WORST-CASE floor.
    worst_floor = bootstrap_bits(boot_K, boot_degree, "worst")
    msgs = _challenge_messages(challenge_seed, n_audit, q_boot, audit_band[0], audit_band[1])
    achieved = device.bootstrap_achieved_bits(msgs, q_boot, boot_K, boot_degree)
    audit_ok = achieved >= worst_floor - 1.0
    return {"public-seed": pub_ok, "blinded-challenge": chal_ok,
            "stage-manifest": stages_ok, "precision-plausible": prec_ok,
            "measured-floor": audit_ok,
            "_claimed_bits": claimed, "_avg_ceiling": round(avg_ceiling, 1),
            "_achieved_bits": round(achieved, 1), "_worst_floor": round(worst_floor, 1)}


def main():
    ctx = Context(N=16, L=6, alpha=3, prime_bits=60)
    public_seed, challenge_seed = 0xC0FFEE, 0xA5A5_1234  # challenge withheld until test time
    devices = [HonestDevice(), GoldenCacheDevice(ctx, public_seed),
               ReplayDevice(), PrecisionSpoofDevice(), SandbagDevice()]
    axes = ["public-seed", "blinded-challenge", "stage-manifest",
            "precision-plausible", "measured-floor"]
    widths = [9, 10, 9, 7, 8]
    print("=== FHE-ACVP adversarial conformance "
          "(blinded challenge + stage-manifest + plausibility + measured-floor audit) ===\n")
    hdr = "".join(f"{h:<{w}}" for h, w in
                  zip(["public", "blinded", "stages", "prec", "meas-fl"], widths))
    print(f"{'device':<24}{hdr}VERDICT")
    allgood = True
    for d in devices:
        r = certify(d, ctx, public_seed, challenge_seed)
        conformant = all(r[a] for a in axes)
        honest = d.name.startswith("honest")
        # correctness of the HARNESS: honest must be conformant; every cheat must be caught
        harness_ok = (conformant == honest)
        allgood &= harness_ok
        cells = "".join(f"{'ok' if r[a] else 'FAIL':<{w}}" for a, w in zip(axes, widths))
        verdict = "CONFORMANT" if conformant else "CAUGHT"
        print(f"{d.name:<24}{cells}{verdict}")

    sp = certify(PrecisionSpoofDevice(), ctx, public_seed, challenge_seed)
    print(f"\nprecision-plausible detail: spoof claimed 30.0b vs derived avg ceiling "
          f"{sp['_avg_ceiling']}b")

    # The challenge DISTRIBUTION is what makes axis (e) sound: a naive narrow-band audit
    # (what a legacy golden suite would draw) MISSES the sandbag; the workload-band audit
    # catches it. Same device, same self-report — only the input band differs.
    sb = SandbagDevice()
    naive = certify(sb, ctx, public_seed, challenge_seed, audit_band=(0.0, 0.125))
    work = certify(sb, ctx, public_seed, challenge_seed, audit_band=(0.6, 1.0))
    print("\nmeasured-floor audit — why the challenge band matters (sandbag device):")
    print(f"  worst-case noise floor                 = {work['_worst_floor']}b")
    print(f"  NAIVE narrow-band audit  achieved      = {naive['_achieved_bits']}b  -> "
          f"{'PASS (sandbag EVADES)' if naive['measured-floor'] else 'caught'}")
    print(f"  WORKLOAD-band audit      achieved      = {work['_achieved_bits']}b  -> "
          f"{'pass' if work['measured-floor'] else 'FAIL (sandbag CAUGHT)'}")
    # the gap is the whole point: naive audit must miss it, workload audit must catch it
    assert naive["measured-floor"] and not work["measured-floor"], \
        "demo invariant broken: the band choice must flip the sandbag verdict"

    print(f"\nharness correctness (honest passes, every cheat caught): {allgood}")
    assert allgood, "adversarial harness failed: a cheat passed or honest was rejected"
    print("FHE-ACVP self-test OK — the standard is attack-resistant (now incl. sandbagging).")


if __name__ == "__main__":
    main()
