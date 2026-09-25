#!/usr/bin/env python3
"""
vbfhe-verify — standalone conformance-certificate verifier (the portable trust artifact).

A regulator / auditor / counterparty runs THIS to check a certificate — with only the certificate
file (Ed25519 public key is embedded) and, optionally, the result ciphertexts to confirm binding.
No VaultBytes secret, no access to the data, no network. Exit code 0 iff the certificate is valid.

Usage:
    python3 vbfhe_verify.py CERT.json                 # verify signature + integrity, print verdict
    python3 vbfhe_verify.py CERT.json --result R.json # also confirm it binds those result items
    python3 vbfhe_verify.py CERT.json --require-pass  # exit non-zero unless verdict == PASS
"""
import argparse
import json
import sys

from vbfhe_conformance import Certificate, ct_digest


def verify_cert(cert_json: str, result_items=None, require_pass=False, ledger_status=None,
                authority_key=None):
    cert = Certificate.from_json(cert_json)
    checks = {}
    # A signature verifies against the key EMBEDDED in the cert — so it only proves internal
    # consistency. To prove the cert came from the real authority, pin the authority's public key
    # (distributed out-of-band). Without pinning, anyone can sign their own "valid" certificate.
    if authority_key is not None:
        checks["authority_key"] = (cert.public_key == authority_key)
    checks["signature"] = cert.verify()                      # Ed25519 (embedded pubkey) + manifest hash
    if result_items is not None:
        checks["result_binding"] = cert.binds_result(result_items)
    if require_pass:
        checks["verdict_pass"] = cert.is_pass
    if ledger_status is not None:
        from vbfhe_ledger import check_cert_status            # logged in transparency ledger + not revoked
        checks["ledger"] = check_cert_status(cert, ledger_status)
    ok = all(checks.values())
    return ok, cert, checks


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vbfhe-verify", description="Verify a vbfhe conformance certificate.")
    ap.add_argument("cert", help="path to the certificate JSON")
    ap.add_argument("--result", help="path to a JSON list of result items (bytes hex or limb triples) to check binding")
    ap.add_argument("--require-pass", action="store_true", help="fail unless verdict == PASS")
    ap.add_argument("--ledger", help="authority base URL to check the cert is logged and not revoked")
    ap.add_argument("--authority-key", help="pinned authority Ed25519 public key (hex) to require; "
                                            "without it, a signature only proves internal consistency")
    args = ap.parse_args(argv)

    with open(args.cert) as fh:
        cert_json = fh.read()
    result_items = None
    if args.result:
        with open(args.result) as fh:
            raw = json.load(fh)
        # accept hex strings (serialized ciphertexts) or raw limb triples
        result_items = [bytes.fromhex(x) if isinstance(x, str) else x for x in raw]

    ledger_status = None
    if args.ledger:
        from vbfhe_http_server import fetch_status
        ledger_status = fetch_status(args.ledger, Certificate.from_json(cert_json).manifest_sha256)

    ok, cert, checks = verify_cert(cert_json, result_items=result_items,
                                   require_pass=args.require_pass, ledger_status=ledger_status,
                                   authority_key=args.authority_key)

    print(f"certificate : {cert.version}  method={cert.method}")
    print(f"backend     : {cert.backend.get('backend_class')}  N={cert.backend.get('N')}")
    print(f"verdict     : {cert.verdict}")
    print(f"signer      : {cert.signer}  [{cert.signature_algo}]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("RESULT      :", "VALID" if ok else "INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
