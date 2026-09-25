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
  Vendor C: GLIDE on an NVIDIA A100 (GPU) — separately demonstrated; see GLIDE_A100_CONFORMANCE_RESULT.md.

Run (from runtime/vbfhe/sdk/):  python3 certify_crossvendor_demo.py
"""
import json
import os

import numpy as np

from vbfhe_sdk import Session
from vbfhe_backend_openfhe import OpenFHEBackend
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
    a = certify("Vendor A — OpenFHE (production CKKS lib)",
                OpenFHEBackend(ring_dim=8192, scale_bits=40, depth=4), server)
    b = certify("Vendor B — SoftwareCKKS (independent impl)", "software", server)

    os.makedirs(EX, exist_ok=True)
    open(os.path.join(EX, "crossvendor_openfhe_cert.json"), "w").write(a.to_json())
    open(os.path.join(EX, "crossvendor_software_cert.json"), "w").write(b.to_json())

    assert a.is_pass and b.is_pass and a.verify() and b.verify()
    assert any(i["name"] == "keyswitch_rotation" and i["passed"] for i in a.invariants)
    print("\n---------------------------------------------")
    print("OK: two INDEPENDENT CKKS implementations certified by ONE oracle-free suite — including")
    print("    attested KEYSWITCH on the production library — both PASS, both signed. A third engine")
    print("    (GLIDE on an A100 GPU) was certified separately. This is the cross-vendor correctness")
    print("    result no single-vendor benchmark can make. Certs saved to examples/crossvendor_*.json.")
