#!/usr/bin/env python3
"""
certify_attested_demo — the SOUND known-answer conformance path.

The old self-report/self-consistency path could be fabricated (a device sends fake bits) and could not
catch a *consistently-wrong* engine (wrong key/scale/offset — the error cancels in the checks). The
ATTESTED path fixes both: the server issues a blinded challenge, the device runs the server's seeded
throwaway probes and returns the decoded outputs, and the server — which KNOWS the answers it seeded —
scores them. This demo shows an honest engine PASS and a consistently-wrong engine CAUGHT.

Run (from runtime/vbfhe/sdk/):  python3 certify_attested_demo.py
"""
import numpy as np

from vbfhe_sdk import Session
from vbfhe_backend_ckks import SoftwareCKKS
from vbfhe_conformance import AttestedRemoteConformanceService
from vbfhe_server import ConformanceServer, in_process_transport


class ConstOffsetCKKS(SoftwareCKKS):
    """A consistently-WRONG engine: every decryption is shifted by +0.5. It satisfies add/mul
    self-consistency (the offset cancels), so the OLD checks passed it — but its answers are wrong."""
    def decode(self, poly, scale, qm):
        return super().decode(poly, scale, qm) + 0.5


def run(label, backend, server, transport):
    s = Session(max_batch=8, precision_bits=22, backend=backend)
    s.matvec(s.encode_plain_matrix([[.5, -.2, .3], [.1, .4, -.6]]),
             [s.encrypt_vector(c) for c in np.random.default_rng(1).normal(size=(3, 6))])
    cert = s.certify(service=AttestedRemoteConformanceService(transport=transport), rounds=4)
    print(f"\n{label}")
    print(f"  attested   : {cert.attested}   verdict: {cert.verdict}")
    for i in cert.invariants:
        b = "blind" if i["blind"] else f"{i['achieved_bits']}b (floor {i['floor_bits']})"
        print(f"    [{'PASS' if i['passed'] else 'FAIL'}] {i['name']:<22} {b}")
    print(f"  verifies vs published key: {cert.verify() and cert.public_key == server.public_key_hex}")
    return cert


if __name__ == "__main__":
    server = ConformanceServer()
    transport = in_process_transport(server)
    honest = run("HONEST engine", "software", server, transport)
    wrong = run("CONSISTENTLY-WRONG engine (+0.5 offset)",
                ConstOffsetCKKS(logN=8, scale_bits=22, depth=4, q0_bits=34), server, transport)

    assert honest.is_pass and honest.attested
    assert not wrong.is_pass and wrong.attested
    print("\nOK: the known-answer (attested) path PASSED the honest engine and CAUGHT the")
    print("    consistently-wrong one — the class the old self-consistency checks missed. The server")
    print("    chose the probes, so it knew the answers; the device could neither fabricate nor cancel.")
