#!/usr/bin/env python3
"""
certify_crossvendor_demo — the persuasion artifact: ONE oracle-free conformance suite certifying
TWO INDEPENDENT CKKS implementations, including the KEYSWITCH primitive every FHE accelerator advertises.

Both engines run the same ATTESTED known-answer protocol: the authority issues a blinded challenge, the
engine computes the authority's seeded throwaway probes (add / plaintext-mul / ct×ct / KEYSWITCH-rotation)
and returns decoded outputs, and the authority — which knows the answers it chose — scores them and signs.
No user data is involved. A device can neither fabricate nor be consistently-wrong.

  Vendor A: OpenFHE (production CKKS library, CPU) — includes attested KEYSWITCH (rotation) coverage.
  Vendor B: SoftwareCKKS (independent reference implementation).
  Vendor C: GLIDE on an NVIDIA A100 (GPU) — separately demonstrated; see the GLIDE A100 run (not included in this distribution).

Run (from the repository root):  python3 certify_crossvendor_demo.py
"""
import json
import os

import numpy as np

from vbfhe_sdk import Session
try:
    from vbfhe_backend_tenseal import TenSEALBackend
    _HAVE_TENSEAL = True
except ImportError as _te:
    TenSEALBackend = None
    _HAVE_TENSEAL = False
    _TENSEAL_WHY = str(_te)

try:
    from vbfhe_backend_openfhe import OpenFHEBackend
    _HAVE_OPENFHE = True
except ImportError as _e:                      # openfhe is a heavy optional native dep
    OpenFHEBackend = None
    _HAVE_OPENFHE = False
    _OPENFHE_WHY = str(_e)
from vbfhe_backend_ckks import SoftwareCKKS
from vbfhe_conformance import AttestedRemoteConformanceService
from vbfhe_server import ConformanceServer, in_process_transport

EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def certify(label, backend, server):
    s = Session(max_batch=8, backend=backend)
    s.matvec(s.encode_plain_matrix([[.5, -.2, .3], [.1, .4, -.6]]),
             [s.encrypt_vector(c) for c in np.random.default_rng(1).normal(size=(3, 6))])
    cert = s.certify(service=AttestedRemoteConformanceService(transport=in_process_transport(server)), rounds=4)
    print(f"\n{label}  (backend={cert.backend['backend_class']}, N={cert.backend['N']})")
    print(f"  attested known-answer · verdict {cert.verdict}")
    for i in cert.invariants:
        if i["blind"]:
            continue
        tag = "KEYSWITCH" if i["name"] == "keyswitch_rotation" else i["name"]
        print(f"    [{'PASS' if i['passed'] else 'FAIL'}] {tag:<22} {i['achieved_bits']}b (floor {i['floor_bits']})")
    print(f"  signed: {cert.verify()}  (authority key {cert.public_key[:12]}…)")
    return cert


if __name__ == "__main__":
    server = ConformanceServer()                              # one authority certifies both vendors
    os.makedirs(EX, exist_ok=True)

    a = None
    if _HAVE_OPENFHE:
        a = certify("Vendor A — OpenFHE (production CKKS lib)",
                    OpenFHEBackend(ring_dim=8192, scale_bits=40, depth=4), server)
        open(os.path.join(EX, "crossvendor_openfhe_cert.json"), "w").write(a.to_json())
    else:
        print("\nVendor A — OpenFHE: SKIPPED, the `openfhe` package is not importable here.")
        print(f"    ({_OPENFHE_WHY})")
        print("    OpenFHE ships a native extension and has no universal wheel; install it to")
        print("    reproduce the OpenFHE column, including the attested keyswitch invariant.")
        print("    Vendor B below needs nothing beyond numpy and cryptography.")

    b = certify("Vendor B — SoftwareCKKS (our own reference)", "software", server)
    open(os.path.join(EX, "crossvendor_software_cert.json"), "w").write(b.to_json())

    t = None
    if _HAVE_TENSEAL:
        t = certify("Vendor C — TenSEAL (Microsoft SEAL)", TenSEALBackend(), server)
        open(os.path.join(EX, "crossvendor_tenseal_cert.json"), "w").write(t.to_json())
    else:
        print(f"\nVendor C — TenSEAL: SKIPPED, `tenseal` is not importable ({_TENSEAL_WHY}).")

    assert b.is_pass and b.verify()
    if t is not None:
        assert t.is_pass and t.verify()
    if a is not None:
        assert a.is_pass and a.verify()
        assert any(i["name"] == "keyswitch_rotation" and i["passed"] for i in a.invariants)

    print("\n---------------------------------------------")
    if a is not None:
        n = 2 + (1 if t is not None else 0)
        print(f"OK: {n} CKKS implementations certified by ONE oracle-free suite, including attested")
        print("    KEYSWITCH on the production library. All PASS, all signed. Two of the three were")
        print("    written by other people, and TenSEAL rescales automatically where the other two")
        print("    need an explicit call, so the suite is not testing its own conventions.")
        print("    Certs saved to examples/crossvendor_*.json.")
    else:
        print("OK: SoftwareCKKS certified by the oracle-free suite and signed. The cross-vendor")
        print("    claim needs both vendors, so it is NOT established by this run — install openfhe")
        print("    and re-run. Cert saved to examples/crossvendor_software_cert.json.")
