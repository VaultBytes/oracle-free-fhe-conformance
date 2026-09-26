# Oracle-Free FHE Conformance (OFC)

A conformance suite that checks whether a CKKS implementation computes correctly. It uses the
scheme's own algebraic laws, it does not decrypt user data, and it ships no stored golden output
vector. On success it emits a signed certificate a third party can verify.

Read the next two sections before you rely on it for anything.

## A respondent that does no FHE can pass the attested protocol

We ship the attack so you can check it yourself.

```
python3 attack_delegate.py
```

It takes the authority's fresh seed, re-derives the challenge probes with numpy, computes the
answers in float64, adds noise scaled to whatever precision it likes, and returns them. It holds no
ring, no ciphertext and no key. It passes.

The reason is not a loose threshold. Producing CKKS's decoded output costs far less than evaluating
CKKS, so a dishonest respondent is faster than an honest one. Raising the probe volume or tightening
a timing bound therefore punishes the honest party. Every cost-based defence runs backwards.

So the attested protocol is sound against implementations that are wrong. It is not sound against
respondents that are dishonest. Those are different threat models and only the first is ours.

| it catches | it does not catch |
|---|---|
| a consistently wrong engine (wrong key, wrong scale) | a respondent computing the answers elsewhere |
| a broken rotation or key-switch | a respondent computing them in plaintext and adding noise |
| arithmetic that misses its noise-theoretic floor | which of several correct engines answered |
| exact arithmetic reporting impossible precision | an engine that routes probes to a slow, correct path |

## The fix is to let the authority hold the key

```
python3 blind_demo.py
```

The attested protocol sends a seed. The device derives the probes in the clear, encrypts under its
own key, evaluates, decrypts, and returns numbers. Everything the delegate needs is handed to it.

The blind protocol inverts that. The authority builds the context, encrypts the probes itself, and
sends ciphertexts with the public evaluation key. The device evaluates without ever holding the
secret and returns ciphertexts. The authority decrypts with a key it never shared.

```
honest blind device            PASS   16.31 / 16.67 / 12.22 bits, floor 6.0
delegate, returns an input     FAIL    0.14 / -0.29 / -0.50
delegate, encrypts a guess     FAIL   -0.00 /  0.00 / -0.00
delegate, float64 plus noise   UNAVAILABLE
```

That last line is the point. The attack is not defeated, it is unavailable. A delegate cannot read
the probe values out of the ciphertexts it receives, so it has nothing to compute an answer from.

Three other problems go with the same change. Forging now costs about what honest evaluation costs,
so probe volume becomes a usable binding on hardware class. The device never decrypts, so the
approximate-decryption oracle that the attested path creates on its own key does not arise. And the
floor derives from parameters the authority chose, so a respondent can no longer declare the bar it
will be judged against.

What it does not fix: a delegate may forward the ciphertexts to a real CKKS library elsewhere. The
certificate then says that correct CKKS evaluation happened somewhere under the authority's
parameters. Narrowing somewhere to here is what custody or a hardware root of trust is for.

## Install and run

```
pip install -r requirements.txt
python3 certify_attested_demo.py     # honest engine passes, injected fault is caught
python3 certify_crossvendor_demo.py  # one authority, three implementations
python3 attack_delegate.py           # the attack above
python3 blind_demo.py                # the same delegate against the blind protocol
```

We pin numpy below 2.0 on purpose, for the reason given in the limitations. `cryptography` is
required. `openfhe` is optional and genuinely skipped when absent, in which case the cross-vendor
demo reports that vendor as skipped and states plainly that the cross-vendor claim is not
established by that run. OpenFHE has no universal wheel, so reproducing its column means building it
for your platform.

## What you should see

The honest engine passes and the injected fault is caught.

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

Precision moves by several bits between runs, because the probes are random and there is no flag to
pin them. The reproducible claim is the verdict, not the bit count. Over 100 runs the faulted engine
fails every time, with medians of 1.77, 1.19 and 1.25 bits, and it has been observed within one bit
of its floor. Do not read the gap above as typical.

## Known limitations

We would rather you found these here than in a certificate.

**The noise model covers encoding rounding only.** It has no fresh-encryption error, no
relinearization noise, no rescale rounding, and no key-switching noise. `keyswitch_rotation` derives
its ceiling from `encode_bits`, which treats key-switching as noiseless. The published floors
therefore carry nine bits of hand-set slack, being `CONFORMANCE_TOLERANCE_BITS` 6.0 plus
`ATTESTED_KNOWN_ANSWER_MARGIN` 3.0, and that slack absorbs the model error. Real engines measure
several bits below the model's nominal worst case, so do not use that ceiling as an upper acceptance
edge until the missing terms are added.

**numpy is pinned below 2.0.** `ckks_golden/bootstrap.py` uses
`np.polynomial.polyutils.RankWarning`, which NumPy 2.0 removed. Under numpy 2 the adversarial audit
raised, the handler swallowed the error, and the authority issued a signed PASS with that
attestation silently absent.

**Probe keys are not separate from production keys in the attested path.** It runs probes on the
same backend instance, and therefore the same secret key, that encrypted the user's data, then
publishes decoded results to the authority. That is an approximate-decryption oracle. Use the blind
protocol, or a throwaway context.

**Two certificate sections describe the reference, not your device.** `primitive_conformance` and
`adversarial_audit` are labelled `subject: reference-RNS (NOT the device under test)`. They attest
that the primitive laws hold for a correct implementation. They say nothing about the engine being
certified and they do not gate its verdict, though a section that errors or fails now does.

**Coverage is four algebraic laws at depth one.** Add, plaintext multiply, ciphertext multiply and
one rotation, on fresh ciphertexts, scored over 64 slots, minimum across four rounds. Deeper levels,
rescaling chains, conjugation, relinearization as a separate law, bootstrapping and edge cases are
untested. There is no fault-detection-probability argument.

**In the attested path the device declares the parameters that set its own floor.** `N` and
`scale_bits` come from the request with only a range check. The blind protocol resolves this,
because the authority chooses them.

## Hardening

This suite was put through four rounds of adversarial review, and the commit history records what
each round changed and why. The short version: the authority now issues the slot count, the slot
window and the rotation requirement with its challenge, an attestation section that errors or fails
no longer passes the verdict, an absent reference package refuses rather than substituting a laxer
threshold, a certificate carrying anything the signature does not cover is refused, the log status
is checked against a signed head and against the largest size previously seen, and the primitive
probes derive from the authority's fresh challenge.

## Repository contents

The conformance suite, both protocols, the authority, signing and verification, an append-only
transparency log with revocation, three backends, and the CKKS reference oracles in `ckks_golden/`
that the suite scores against.

The three backends are deliberately not all ours. OpenFHE and TenSEAL, which wraps Microsoft SEAL,
were written by other people. Our own `SoftwareCKKS` is scored by our own reference oracles, so it
is the weakest of the three as evidence. TenSEAL also rescales automatically where the other two
require an explicit call, which is the kind of legitimate variation a conformance suite must not
mistake for an error. It exposes no rotation in its Python surface, so the suite reports no
keyswitch invariant for it rather than scoring one zero.

Not included: application cartridges, the threshold and multiparty layer, the hardware backend, and
the chip lowering path. None is needed to reproduce anything here.

## Licence

Apache-2.0. See `LICENSE`.
