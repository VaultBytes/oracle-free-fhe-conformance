# Security & soundness — what this system does and does NOT prove

This client is a **reference implementation**. It is deliberately honest about its limits. Read this
before quoting the certificate's guarantees to a regulator, acquirer, or customer.

## What a certificate ACTUALLY proves
A signed certificate attests that, on server-issued precision floors derived from CKKS noise theory,
the measured operations of a CKKS backend are **internally algebraically consistent** (add
commutativity; additive homomorphism; plaintext-mul and ct×ct distributivity) **at the measured
precision**, and that the result-ciphertext digest matches. That is a real, novel, useful signal.

## The delegation limit (read this first)

A respondent performing no homomorphic encryption at all obtains a signed PASS from the attested
path. `attack_delegate.py` demonstrates it: re-derive the public probes, answer in float64, paint
Gaussian noise to any target precision, return. No ring, no ciphertext, no key.

This is structural, not a bug we intend to fix. Producing CKKS's decoded output is cheaper than
evaluating CKKS, so the dishonest respondent is also CHEAPER than the honest one, which inverts every
cost, latency and throughput defence. Measured in `cost_asymmetry.py`: 240x cheaper at the default
64 examined slots, 52x at full packing, 6x at full packing against honest evaluation alone. The
suite is sound against implementations that are WRONG and unsound against respondents that are
DISHONEST. Use it as a test for error, not a test for fraud.

The precision band has been corrected and it does not close this. The upper bound is now derived
(`attainable_bits`) and applied per round against that round's own dynamic range, rather than
hand-set at `scale_bits + 8`. It refuses exact arithmetic and it refuses a delegate aiming above the
band. The accepted interval is still about 25 bits wide at N=8192, scale 2^40, and narrowing it
below the 4.2-bit disagreement between OpenFHE and TenSEAL at identical parameters would refuse one
of them. See `band_experiment.py`.

One transcript statistic looked promising and does not survive a production library: the certificate
reports `error_coupling`, the correlation between the reported pmul error and `w * add error`, which
an honest engine carries structurally. Measured r = 0.958 on our own reference implementation but
only 0.124 on OpenFHE, where a delegate painting at 47 bits scores 0.136 -- above the honest engine.
It is reported and deliberately not enforced.

## A fail-open dependency range (fixed)

`requirements.txt` previously allowed numpy >= 2.0, in which `np.polynomial.polyutils.RankWarning`
no longer exists. The adversarial audit then raised, the handler swallowed it, and the authority
issued a signed PASS with that attestation silently absent. numpy is now pinned below 2.0 and an
attestation section that errors now fails the verdict.

## What it does NOT prove (do not overclaim)
1. **Not "computed correctly against the intended result."** The invariants are *self-consistency*
   checks (they compare two of the device's own outputs), so they are blind to a *consistently wrong*
   device. Precisely: any affine decoder error `y -> s*y + d` cancels wherever both sides of an
   identity apply the decoder the same number of times. `plainmul_distributive` and
   `ctmul_distributive` decode once per side, so both the scale error `s` and the offset `d` vanish
   there; only `add_homomorphism` is unbalanced (one decode against two), and it therefore sees `d`
   and still not `s`. Measured at N=256: a x1.01 scale error scores 50.91 / 18.17 / 15.06 against an
   honest 50.90 / 18.17 / 15.06, indistinguishable on every law. A +0.5 offset scores 3.66 / 18.13 /
   15.10 — caught by add-homomorphism alone, and invisible to the other two.

   An earlier version of this document said the additive offset passes because "the error cancels on
   both sides." That is wrong: it leaves a residual `d` in add-homomorphism. The multiplicative error
   is the one that genuinely cancels everywhere, and it is the dangerous one, because a device wrong
   in every answer it will ever give passes every self-consistency law cleanly.

   Correctness against an external semantics (this ciphertext decrypts to the intended plaintext) is
   **out of scope** for the self-consistency mode. The attested known-answer mode addresses it, and
   is itself bounded by `attack_delegate.py` — see README.
2. **The remote path is not attested.** In the client→server split the device **self-reports** its
   achieved bits; nothing binds those numbers to a real execution. The server bounds them to a
   physically plausible range (`judge_measurements`), which stops the trivial `inf`/`9999` forgery, but
   a determined adversary can still submit plausible fabricated numbers. **Binding measurements to the
   server's challenge (challenge-response / commitment / proof-of-work) is required future work.** The
   co-located `LocalOracleFreeService` path runs the probes in-process and is not fabricatable this way
   (but Finding 1 above still applies).
3. **The `adversarial_audit` section is about the REFERENCE family, not the device under test.** It runs
   `ofc_adversarial` against a known-honest reference implementation using the server's challenge seed.
   It is recorded for transparency, labelled `"subject": "reference-RNS (NOT the device under test)"`,
   and is **not folded into the verdict**. It does not attest the certified device is sandbag-resistant.
   Same for `primitive_conformance`.
4. **`application_conformance` is an INDICATIVE proxy.** It maps measured ct×ct bits to an
   argmax-agreement number via a *plaintext quantization* simulation — not an encrypted LLM run. It
   saturates above ~14 bits, so it only catches badly-degraded devices. The A100/GLIDE certificate's
   "argmax 1.0" is this Python simulation, not a GPU LLM inference.
5. **"Oracle-free" means no *stored golden vector*,** not "no reference computation anywhere": one
   underlying check (NTT↔convolution) compares against a schoolbook convolution computed on the
   certifier's own hardware. This avoids golden-cache gaming but is not literally reference-free.
6. **Derived floors are calibrated.** `CONFORMANCE_TOLERANCE_BITS` (6, or 4 in strict mode) is the
   worst-case-to-average gap; it is a reasonable but not independently-derived bar. Prefer the
   `[worst, avg]` range approach for a normative standard.

## The ATTESTED path (closes Findings 2 & 3)
`AttestedRemoteConformanceService` + `ConformanceServer.issue_challenge`/`judge_attested` implement a
**known-answer challenge**: the server issues a fresh blinded seed, the device runs the server's seeded
throwaway probes (encrypt→compute→decrypt — no user data), and the server re-derives the probes (it
**knows** the answers) and scores the device's decoded outputs against them. A device **cannot
fabricate** (it can't know a fresh seed's probes) and **cannot be consistently-wrong** (a wrong
key/scale/offset makes the decoded output disagree with the server's known plaintext — verified: a
constant-offset engine is caught at ~2 bits and FAILs). Certificates record `attested: true`. The
legacy self-report path (`RemoteConformanceService`) remains for co-located use and is marked
`attested: false`; **prefer the attested path for any untrusted remote device.**

## Production controls — now implemented (reference-grade)
| Control | Now |
|---|---|
| Signing key | **ephemeral random by default**; the demo key is used ONLY if `VBFHE_DEMO_KEY` is set; load a real key from `VBFHE_SIGNING_KEY` (hex seed or file — KMS/HSM export). Never ships a usable prod key. |
| Endpoint auth | `ConformanceServer(api_token=…)` → `/v0/certify`, `/v0/attest`, `/v0/challenge`, `/v0/revoke` require `Authorization: Bearer <token>` |
| TLS | `ConformanceHTTPService(ssl_context=…)` serves HTTPS; `http_transport(tls_context=…)` on the client |
| Authority key pinning | `vbfhe-verify --authority-key <hex>` (and `verify_cert(authority_key=…)`) — a signature alone only proves internal consistency |
| Signed ledger head | `GET /v0/log` and `ConformanceServer.signed_head()` return an authority-**signed** head (MITM-resistant) |
| Persistent ledger | `ConformanceLedger(path=…)` appends to JSONL and verifies the chain on load |
| Input bounds | body cap (256 KB), N/scale/op_trace bounds, implausible-measurement rejection, single-use challenges |

## Still not production-hardened (deployment/ops, not code)
Real HSM/KMS integration (vs. a file/env key), a managed reverse proxy with **rate limiting**, an
accredited-principal registry + cert-chain, externally-published (e.g. object-store) signed ledger
heads, and formal FTO/threat-model sign-off. The **attested** path also currently covers the CKKS
compute invariants; extending known-answer coverage to bootstrap is future work; keyswitch is now covered by the attested path.

## Fixes already applied (this hardening pass)
Fail-closed verdict (an errored optional check no longer passes); `rounds>=1` guard;
implausible-measurement rejection; `poly_eval` constant handled; `mkstemp` (no TOCTOU); per-instance
backend RNG; thread-safe ledger + revoke-only-known-ids; forward-compatible `Certificate.from_json`;
`allow_nan=False` canonical JSON; verifier authority-key pinning; body-size limit + generic errors.

**Bottom line:** the accurate one-line description is *"a signed, algebraically-grounded
self-consistency certificate for a CKKS backend, with server-set precision floors,"* — not "proves any
FHE chip computed your data correctly." Use that framing.
