# Oracle-Free FHE Conformance (OFC)

A conformance suite that checks whether a CKKS implementation computes correctly, using the
scheme's own algebraic laws, without decrypting user data and without shipping a stored golden
output vector. It emits a signed, independently verifiable certificate.

This is the reference implementation and the artifact for the accompanying paper. Everything
reported in the paper can be reproduced from this repository.

## What this establishes, and what it does not

We want to be exact about this, because a conformance test that overstates its guarantee is worse
than none.

The suite certifies **algebraic correctness against authority-chosen probes at a measured
precision**. The authority issues a fresh single-use seed, the device derives throwaway probes from
it, evaluates the operations, and returns the results. The authority knows the expected answers
independently, so a device cannot pass by precomputing, and it cannot pass while consistently wrong,
because a global key, scale or offset error no longer cancels against a known reference.

It does **not** establish which implementation produced the answers. A respondent that holds the
seed can obtain correct answers by any means, including running a second CKKS library on an ordinary
host. No protocol that observes only challenge and response can separate that case from honest
execution, because every correct CKKS implementation returns the same decoded values by
construction. This is a property of black-box conformance testing, not a defect in this design.

Two deployment modes follow from that, and a certificate should name the one it was issued under.

| mode | what it binds |
|---|---|
| custody | the auditor drives the interface directly, so there is no channel along which the computation can be delegated |
| remote | the certificate binds to the interface: correct CKKS behaviour is reachable behind that endpoint, which detects an incorrect implementation and does not identify which implementation answered |

One case is separable. CKKS is approximate, so an honest engine's precision falls inside a band the
parameters determine. A respondent answering in plaintext, or in any arithmetic materially more
accurate than the declared parameters permit, returns results that are too good. Precision above the
average-case ceiling is inconsistent with having performed the approximate computation at all. The
present implementation enforces only the lower edge of that band; the two-sided form is future work,
and the upper edge is what would carry the anti-delegation property.

The suite also does not prove end-to-end application correctness on a specific model,
security-parameter adequacy, or side-channel resistance. See `SECURITY.md`.

## Install and run

```
pip install -r requirements.txt
python3 certify_attested_demo.py
python3 certify_crossvendor_demo.py
```

`numpy` and `cryptography` are required. `openfhe` is optional: without it the OpenFHE backend is
skipped and the `SoftwareCKKS` reference implementation still runs, so the suite is usable on a
machine with no FHE library installed.

`certify_attested_demo.py` runs the attested known-answer profile against an honest engine and
against a deliberately broken one. `certify_crossvendor_demo.py` runs one authority against two
independent CKKS implementations and writes signed certificates to `examples/`.

## What you should see

The honest engine passes and the injected fault is caught:

```
HONEST engine                        verdict: PASS
  [PASS] add_homomorphism       15.32b (floor 6.0)
  [PASS] plainmul_distributive  14.83b (floor 6.0)
  [PASS] ctmul_distributive     10.40b (floor 6.0)

CONSISTENTLY-WRONG engine (+0.5 offset)   verdict: FAIL
  [FAIL] add_homomorphism        1.76b (floor 6.0)
  [FAIL] plainmul_distributive   0.53b (floor 6.0)
  [FAIL] ctmul_distributive      0.95b (floor 6.0)
```

The constant offset is invisible to a self-consistency check, because the same error appears on both
sides of every identity and cancels. It is caught here because the authority knows the answer.

The precision figures move by a bit or two between runs, since the probes are random. The
reproducible claim is the verdict and the margin against the floor, not the exact bit count.

## Repository contents

The conformance suite, the attested protocol, the authority, the signing and verification path, the
append-only transparency log with revocation, two backends, and the CKKS reference oracles the suite
scores against.

`ckks_golden/` holds the definitional references: modular arithmetic and NTT prime selection, the
negacyclic NTT, the canonical-embedding encoder, the Galois automorphism, key-switching, the
bootstrap precision oracle, and the invariant, adversarial, end-to-end and composition suites built
on them.

Not included: application-layer cartridges, the threshold and multiparty layer, the hardware
backend, and the chip lowering path. None of them is needed to reproduce the paper.

## A note on the dependency structure

`vbfhe_conformance.py` imports several `ckks_golden` modules lazily, inside functions, and catches
failures. If those modules are absent the suite currently degrades rather than refusing: precision
floors fall back to defaults and some invariant families report `conformant: null` while a
certificate is still issued. We found this while assembling this repository, by staging the files in
isolation and running them, which is the only way it becomes visible. Both packages ship together
here so the behaviour does not arise, and making an absent dependency a hard refusal is the correct
fix rather than shipping them together.

## Licence

Apache-2.0. See `LICENSE`.
