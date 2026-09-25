# Oracle-Free FHE Conformance (OFC)

A conformance suite that checks whether a CKKS implementation computes correctly, using the
scheme's own algebraic laws, without decrypting user data and without shipping a stored golden
output vector. It emits a signed, independently verifiable certificate.

Read the next section before you rely on it for anything.

## What this establishes, and what it does not

**A respondent that performs no homomorphic encryption at all can obtain a signed PASS.** We ship
the attack so you can check:

```
python3 attack_delegate.py
```

It takes the authority's fresh seed, re-derives the challenge probes with numpy because the
derivation is public code in this repository, computes the answers in float64, adds noise scaled to
whatever precision it likes, and returns them. No ring, no ciphertext, no key. It passes.

That is not a bug we intend to fix. Every correct CKKS implementation returns the same decoded
values, to within the precision its parameters permit, so a protocol observing only decoded values
cannot tell which implementation produced them. Producing those values by other means is also
*cheaper* than evaluating the circuit homomorphically, so throughput and latency defences fail in
the same direction: the delegate is faster than the honest engine, not slower.

So the suite is sound against implementations that are **wrong**, and unsound against respondents
that are **dishonest**. Those are different threat models and only the first one is ours.

| it catches | it does not catch |
|---|---|
| a consistently-wrong engine (wrong key, wrong scale) | a respondent computing the answers elsewhere |
| a broken rotation or key-switch | a respondent computing them in plaintext and adding noise |
| arithmetic that misses its noise-theoretic floor | which of several correct engines answered |
| exact arithmetic reporting impossible precision | an engine that routes conformance probes to a slow, correct path |

Use it as a test for error. Do not use it as a test for fraud. If you need the second, you need
verifiable FHE, a hardware root of trust, or physical custody of the device — and note that custody
of the box is not custody of the computation, since the host driver can answer in float64 without
ever touching the accelerator.

## Install and run

```
pip install -r requirements.txt
python3 certify_attested_demo.py        # honest engine passes, injected fault is caught
python3 certify_crossvendor_demo.py     # one authority, two implementations
python3 attack_delegate.py              # the attack above
```

`numpy` is pinned below 2.0 on purpose, see below. `cryptography` is required. `openfhe` is
optional and genuinely skipped when absent: the cross-vendor demo reports the OpenFHE vendor as
SKIPPED and says plainly that the cross-vendor claim is not established by that run. OpenFHE has no
universal wheel, so reproducing its column means building it for your platform.

## What you should see

The honest engine passes and the injected fault is caught:

```
HONEST engine                             verdict: PASS
  [PASS] add_homomorphism       15.17b (floor 6.0)
  [PASS] plainmul_distributive  14.90b (floor 6.0)
  [PASS] ctmul_distributive     10.85b (floor 6.0)

CONSISTENTLY-WRONG engine (+0.5 offset)   verdict: FAIL
  [FAIL] add_homomorphism        3.44b (floor 6.0)
  [FAIL] plainmul_distributive   4.78b (floor 6.0)
  [FAIL] ctmul_distributive      4.68b (floor 6.0)
```

Precision figures move by several bits between runs, because the probes are random and there is no
seed flag to pin them. The reproducible claim is the verdict, not the bit count. The faulted engine
has been observed as close as one bit from its floor, so do not read the gap above as typical.

## Known limitations in this implementation

We would rather you find these here than in a certificate.

**The noise model covers encoding rounding only.** It has no fresh-encryption error, no
relinearization noise, no rescale rounding, and no key-switching noise: `keyswitch_rotation` derives
its ceiling from `encode_bits`, treating key-switching as noiseless. Consequently the published
floors carry 9 bits of hand-set slack (`CONFORMANCE_TOLERANCE_BITS` 6.0 plus
`ATTESTED_KNOWN_ANSWER_MARGIN` 3.0) which absorbs the model error. Real engines have been measured
several bits below the model's nominal worst-case ceiling, so that ceiling should not be used as an
upper acceptance edge until the model includes the missing terms.

**Probe keys are not separate from production keys.** The attested path runs probes on the same
backend instance, hence the same secret key, that encrypted the user's data, and publishes decoded
results to the authority. That is an approximate-decryption oracle in the IND-CPA-D sense. Use a
throwaway context for certification until this is fixed.

**Two certificate sections describe the reference, not your device.** `primitive_conformance` and
`adversarial_audit` are labelled `subject: reference-RNS (NOT the device under test)`. They attest
that the primitive laws hold for a correct implementation. They say nothing about the engine being
certified, and they do not gate its verdict — though a section that *errors* now does, so a broken
dependency can no longer produce a silent PASS.

**Coverage is four algebraic laws at depth one.** Add, plaintext multiply, ciphertext multiply and
one rotation, on fresh ciphertexts, scored over 64 slots, minimum across four rounds. Untested:
deeper levels, rescaling chains, conjugation, relinearization as a separate law, bootstrapping, and
edge cases. There is no fault-detection-probability argument.

**The device declares the parameters that set its own floor.** `N` and `scale_bits` come from the
request with only a range check.

## Fixed in the second revision

The authority now issues the slot COUNT, the slot WINDOW and the rotation requirement with its
challenge. Previously the examined slots were always `[0:64]`, a fixed public window, so an engine
correct on 64 of 4096 slots and arbitrary on the other 4032 was certified; and `has_ks = "ks" in
outputs` meant a wrong-rotation engine could delete one JSON key and watch the invariant vanish from
a signed PASS. A reference attestation that FAILS now also fails the verdict, where before only one
that ERRORED did.

`attack_delegate.py` was updated to speak the new protocol and still passes. That is the point: the
hardening closes protocol evasions and does not touch delegation.

## Fixed in the first revision

For anyone comparing against the first published commit: responses claiming more precision than the declared scale can carry are now refused rather than
silently clamped to a passing value, though that edge is a hand-set sanity bound and not the
noise-theoretic ceiling, which honest engines exceed and which therefore cannot be enforced; a response that misstates its round count, slot
count or round vectors against the issued challenge is now rejected; an errored attestation section
now fails the verdict instead of being waved through; a missing noise model now refuses instead of
substituting a laxer floor; all five sites that put the clone's parent directory ahead of the package on the import path
now append instead of inserting, so the reference package can no longer be shadowed; numpy is pinned below 2.0; the OpenFHE import is guarded; and six references to files
that were never published have been removed.

## Repository contents

The conformance suite, the attested protocol, the authority, signing and verification, an
append-only transparency log with revocation, two backends, and the CKKS reference oracles in
`ckks_golden/` that the suite scores against.

Not included: application cartridges, the threshold and multiparty layer, the hardware backend, and
the chip lowering path. None is needed to reproduce anything here.

## Licence

Apache-2.0. See `LICENSE`.
