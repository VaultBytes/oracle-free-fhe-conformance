#!/usr/bin/env python3
"""
vbfhe conformance SERVER — reference implementation of the CLOSED judging service.

This is the other end of `RemoteConformanceService`: the VaultBytes conformance authority. It receives
a request (backend descriptor + op-trace + the device's MEASURED bits — NO ciphertexts, NO keys, NO
plaintext), sets the precision FLOORS from CKKS noise theory (its authority, not the device's), judges
pass/fail + application + composition, and returns a certificate signed with the SERVICE key. The
public key is published so anyone verifies certificates with no shared secret.

Separation of concerns (the certification-authority model, like an accredited lab + a standards body):
  * the DEVICE measures its own precision on throwaway probes (client-side `probe_measure`);
  * the SERVER owns the floors + the signature — a device cannot pick an easy bar or forge a cert.
Anti-fabrication of the submitted measurements is the `ckks_golden/ofc_adversarial` stage-manifest
layer (blinded challenges + unforgeable per-stage digests); this reference judges the measurements.

The transport is abstracted: `in_process_transport(server)` runs it in-process (used by the demo/tests);
a real deployment swaps in an HTTP handler that calls `server.judge(request)`.
"""
from __future__ import annotations

import json

from vbfhe_conformance import (
    derive_floors, judge_measurements, assemble_certificate, CONFORMANCE_TOLERANCE_BITS,
)


class ConformanceServer:
    """The VaultBytes conformance authority. Holds the service signing key; publishes the public key."""
    signer_name = "vbfhe-conformance-service/authority"

    def __init__(self, signer=None, with_primitive: bool = True, ledger=None, api_token=None):
        from vbfhe_signing import default_signer
        self._signer = signer or default_signer()      # HSM/KMS-held Ed25519 in production
        self._with_primitive = with_primitive
        if ledger is None:
            from vbfhe_ledger import ConformanceLedger
            ledger = ConformanceLedger()
        self.ledger = ledger                            # append-only transparency log of issued certs
        self.api_token = api_token                      # if set, certify/revoke require this bearer token
        self._pending = {}                              # single-use challenge seeds {id: (seed,profile,rounds)}
        self._chal_counter = 0

    @property
    def public_key_hex(self):
        """Publish this; anyone verifies VaultBytes certificates against it with no secret."""
        return self._signer.public_key_hex

    def authorize(self, token) -> bool:
        """Endpoints that mutate trust (certify, revoke) require the operator token if one is configured."""
        return self.api_token is None or (token is not None and str(token) == str(self.api_token))

    # ---- attested (known-answer) flow: server issues the blinded challenge, then judges the answers ----
    def issue_challenge(self, profile="workload", rounds=6) -> dict:
        import secrets
        rounds = max(1, min(int(rounds), 32))
        cid = f"ch-{self._chal_counter}"; self._chal_counter += 1
        seed = secrets.randbits(63)
        from vbfhe_conformance import _ATTEST_N
        n_report = _ATTEST_N                       # the AUTHORITY sets the exam size, not the examinee
        self._pending[cid] = (seed, profile, rounds, n_report)
        return {"challenge_id": cid, "seed": seed, "profile": profile, "rounds": rounds,
                "n_report": n_report}

    def judge_attested(self, request: dict) -> str:
        """Judge a KNOWN-ANSWER attested response: verify the challenge is one we issued (single use),
        re-derive the probes from OUR seed, score the device's decoded outputs against the answers WE
        know, and sign. Non-fabricatable and catches consistently-wrong engines (see SECURITY.md)."""
        from vbfhe_conformance import attested_measurements
        cid = request.get("challenge_id")
        if cid not in self._pending:
            raise KeyError("unknown or already-used challenge")
        seed, profile, rounds, n_issued = self._pending.pop(cid)   # single-use
        N = int(request["N"]); scale_bits = int(request["scale_bits"])
        if not (256 <= N <= (1 << 20)) or not (10 <= scale_bits <= 120):
            raise ValueError("implausible params")
        # The AUTHORITY set the examination; the response does not get to restate it. Every one of
        # these was previously taken from the device unvalidated, so a response of rounds=1,
        # n_report=1 against a rounds=4, n_report=64 challenge PASSED and the certificate then
        # recorded rounds=4 -- an assertion about a measurement nobody made. Scoring is min-over-
        # rounds and max-over-slots, so both fields are a difficulty dial the examinee was holding.
        resp = request["response"]
        if int(resp.get("rounds", -1)) != int(rounds):
            raise ValueError(
                f"response declares rounds={resp.get('rounds')!r} but challenge {cid} issued "
                f"rounds={rounds}")
        if resp.get("profile") != profile:
            raise ValueError("response profile does not match the issued challenge")
        # n_report was a difficulty dial the examinee held: scoring is max-error-over-REPORTED
        # slots, so shrinking it shrinks the exam. An engine wrong on 63 of 64 slots reported
        # n_report=1 and received a signed PASS. It is now issued by the authority and checked.
        _n = int(resp.get("n_report", 0))
        if _n != int(n_issued):
            raise ValueError(
                f"response reports {_n} slot(s) but challenge {cid} issued n_report={n_issued}")
        # outputs is keyed by operation, each holding one vector per round.
        _outs = resp.get("outputs") or {}
        for _op in ("add", "pmul", "cmul"):          # the three every CKKS engine must answer
            _v = _outs.get(_op)
            if not isinstance(_v, list) or len(_v) != int(rounds):
                raise ValueError(
                    f"response gives {0 if _v is None else len(_v)} round(s) for '{_op}'; "
                    f"challenge {cid} issued {rounds}")
            for _i, _row in enumerate(_v):
                if not isinstance(_row, list) or len(_row) != _n:
                    raise ValueError(
                        f"'{_op}' round {_i} reports {0 if _row is None else len(_row)} slots, "
                        f"not the declared n_report={_n}")
        if "ks" in _outs and len(_outs["ks"]) != int(rounds):
            raise ValueError("keyswitch answered for a different number of rounds than issued")
        from vbfhe_conformance import ATTESTED_KNOWN_ANSWER_MARGIN
        meas = attested_measurements(request["response"], N=N, seed=seed, profile=profile)
        tol = (4.0 if profile == "strict" else CONFORMANCE_TOLERANCE_BITS) + ATTESTED_KNOWN_ANSWER_MARGIN
        floors, ceilings, source = derive_floors(None, tol, N=N, scale_bits=scale_bits)
        invs = judge_measurements(meas, floors, scale_bits=scale_bits)
        _r = meas.get("error_coupling_r")
        coupling = None if _r is None else {
            "pearson_r": _r, "pairs": int(rounds) * int(resp.get("n_report", 0)),
            "enforced": False,
            "meaning": "honest engines derive pmul from the add ciphertext, so this is strongly "
                       "positive; an independent fabricator sits near zero regardless of how well "
                       "it matches error magnitude. DIAGNOSTIC ONLY -- not gated, false-refusal "
                       "rate on real engines is uncharacterised."}
        desc = {"scheme": request.get("scheme", "CKKS"), "backend_class": str(request["backend_class"])[:128],
                "N": N, "slots": int(request.get("slots", N // 2)),
                "scale_bits": scale_bits, "q_bits": int(request.get("q_bits", 0))}
        cert = assemble_certificate(
            descriptor=desc, session_id=str(request.get("session_id", "remote"))[:128],
            op_trace=list(request.get("op_trace", []))[:512], rounds=rounds, profile=profile,
            floors=floors, ceilings=ceilings, floor_source=source, invs=invs,
            result_digest=request.get("result_digest"), with_primitive=self._with_primitive,
            signer=self._signer, signer_name=self.signer_name, challenge_seed=seed, attested=True,
            error_coupling=coupling, n_report=int(n_issued))
        self.ledger.record(cert)
        return cert.to_json()

    def revoke(self, cert_id: str, reason: str):
        """Revoke a previously-issued certificate (append-only in the ledger)."""
        return self.ledger.revoke(cert_id, reason)

    def status(self, cert_id: str) -> dict:
        return self.ledger.status(cert_id)

    def signed_head(self) -> dict:
        """The current ledger head, SIGNED by the authority, so a verifier can authenticate it
        independently of transport security (defends against a MITM spoofing ledger state)."""
        head = self.ledger.head(); size = self.ledger.size()
        sig = self._signer.sign(f"{head}:{size}".encode())
        return {"head": head, "size": size, "chain_ok": self.ledger.verify_chain(),
                "signature": sig, "algo": self._signer.algo, "public_key": self._signer.public_key_hex}

    def judge(self, request: dict) -> str:
        """Judge a conformance request and return a signed certificate (JSON). The SERVER derives the
        floors from the request's declared params (the device does not choose them) AND issues a fresh
        BLINDED challenge seed for the anti-sandbag audit (device cannot precompute or replay it)."""
        import secrets
        profile = request.get("profile", "workload")
        N = int(request["N"]); scale_bits = int(request["scale_bits"])
        if not (256 <= N <= (1 << 20)) or not (10 <= scale_bits <= 120):
            raise ValueError("implausible params (N/scale_bits out of range)")
        op_trace = list(request.get("op_trace", []))[:512]          # bound attacker-controlled trace
        tol = 4.0 if profile == "strict" else CONFORMANCE_TOLERANCE_BITS
        floors, ceilings, source = derive_floors(None, tol, N=N, scale_bits=scale_bits)
        invs = judge_measurements(request["measurements"], floors, scale_bits=scale_bits)
        desc = {"scheme": request.get("scheme", "CKKS"), "backend_class": str(request["backend_class"])[:128],
                "N": N, "slots": int(request.get("slots", N // 2)),
                "scale_bits": scale_bits, "q_bits": int(request.get("q_bits", 0))}
        cert = assemble_certificate(
            descriptor=desc, session_id=str(request.get("session_id", "remote"))[:128],
            op_trace=op_trace, rounds=int(request.get("rounds", 0)), profile=profile,
            floors=floors, ceilings=ceilings, floor_source=source, invs=invs,
            result_digest=request.get("result_digest"), with_primitive=self._with_primitive,
            signer=self._signer, signer_name=self.signer_name, challenge_seed=secrets.randbits(31))
        self.ledger.record(cert)                        # log every issued certificate (transparency)
        return cert.to_json()


def in_process_transport(server: ConformanceServer):
    """A transport that runs the server in-process (serialize the request through JSON to mimic the wire).
    Dispatches the attested two-step (challenge/attest) and the legacy self-report judge. Swap for an
    HTTP client in production."""
    def _transport(request: dict) -> str:
        wire = json.loads(json.dumps(request))          # round-trip: only JSON-able data crosses
        kind = wire.get("kind")
        if kind == "challenge":
            return json.dumps(server.issue_challenge(wire.get("profile", "workload"), wire.get("rounds", 6)))
        if kind == "attest":
            return server.judge_attested(wire)
        return server.judge(wire)                       # legacy self-report path
    return _transport
