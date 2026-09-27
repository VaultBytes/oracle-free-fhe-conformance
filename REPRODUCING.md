# Reproducing the paper's tables and figures

Every number in *Decoded-Output Conformance Need Not Establish CKKS Execution* was measured at tag
`eprint-v6`, not from a development checkout. Check that tag out before running anything:

```
git clone https://github.com/VaultBytes/oracle-free-fhe-conformance
cd oracle-free-fhe-conformance && git checkout eprint-v6
```

## Which script produces what

| Script | What it produces |
|---|---|
| `attack_delegate.py` | Table 2, the delegate's signed PASS rate at each target precision |
| `cost_asymmetry.py` | Table 1, the delegate's cost against honest evaluation at each exam size |
| `threshold_headroom.py` | the partial-cancellation rates and the false-reject evidence behind the upper threshold |
| `blind_demo.py` | Table 3, the blind protocol, with its controls |
| `band_experiment.py` | the two external columns of Table 4, the interval widths and the cross-library differences |
| `bimodality_figure.py` | Figure 1 |
| `certify_crossvendor_demo.py` | single runs of all three columns, each engine at its own settings |
| `certify_attested_demo.py` | the honest and consistently-wrong engines under the known-answer protocol |

## Dependencies

`attack_delegate.py` and `blind_demo.py` need only `numpy` and `cryptography`, so Tables 2 and 3
reproduce from a clean clone with no optional dependency. `cost_asymmetry.py` needs OpenFHE.
`band_experiment.py` needs OpenFHE and TenSEAL. `bimodality_figure.py` needs `matplotlib`. Neither
FHE library has a universal wheel, and a vendor that is absent is reported as skipped rather than
scored, so a run with both absent still reproduces the attack and the blind protocol and states
that it does not establish the cross-vendor claim.

Pin `numpy>=1.24,<2`. The reason is in `requirements.txt`.

## What is not reproducible to the decimal, and why

Three results move between runs, and the paper says so in each case. They are not noise in a
measurement; they are quantities that depend on randomness the protocol draws fresh.

**Table 3, the blind protocol.** The authority's key is drawn with `secrets.randbits(63)` per
session, which is the property the protocol depends on, so the scores move. The paper reports
medians over 40 sessions with the observed range.

**Table 2's 42-bit row.** The authority draws a fresh challenge seed per session, and that row
straddles the acceptance threshold, so its pass rate varies between runs. The rows above and below
it do not: exact, 47.9 and 44 bits are refused every time, and 40 bits and below are accepted every
time.

**The consistently-wrong engine in `certify_attested_demo.py`.** The script runs the engine once
and is not seeded. The verdict and the distance from the floor reproduce; the exact triple does not.

## A defect worth knowing about

An earlier revision reached into the `ckks_golden` reference package through imports placed inside
functions, each wrapped in a handler that returned a null result on failure. With that package
absent the suite did not refuse. Precision floors fell back silently to a laxer set, 6.0/5.0/5.0
against the noise-theoretic 6.0/6.0/6.0, so the same code certified against one threshold in the
development tree and a different one when staged alone, while every printed component still read
PASS. Importing every module succeeded in both cases, so an import-level check reported success.

An absent reference package now refuses rather than substituting a threshold. The lesson generalises:
a self-containment claim is vacuous unless the intended files are staged in isolation and executed,
because the development tree silently supplies whatever the package list omits.
