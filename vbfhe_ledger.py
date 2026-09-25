#!/usr/bin/env python3
"""
vbfhe conformance LEDGER — an append-only transparency log + revocation for issued certificates.

A conformance authority needs the same trust infrastructure a CA does: a tamper-evident, append-only
record of every certificate it issued, and the ability to revoke one. This is that log.

  * Every issued cert is recorded by its `cert_id` (= the certificate's manifest_sha256, unique per
    cert body) in a hash-CHAINED log: each entry hashes the previous entry, so any edit/removal of
    history changes the head and is detectable. (Certificate-Transparency-style, simplified.)
  * Revocation is itself an append-only entry, so the log stays immutable; `status()` reflects the
    latest state. A verifier checks a cert is LOGGED and NOT REVOKED before trusting it.

In-memory reference; a production authority persists the chain and publishes signed heads.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

GENESIS = "0" * 64


def _entry_hash(prev_hash: str, core: dict) -> str:
    body = prev_hash + json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class LedgerEntry:
    seq: int
    kind: str                 # "issue" | "revoke"
    cert_id: str
    verdict: Optional[str]
    detail: Optional[str]     # revoke reason, or None
    at: float
    prev_hash: str
    entry_hash: str


class ConformanceLedger:
    """Append-only, hash-chained log of issued/revoked certificates."""

    def __init__(self, path=None):
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()          # ThreadingHTTPServer -> concurrent record/revoke
        self._path = path                      # optional append-only persistence (JSONL)
        if path:
            self._load(path)

    def _load(self, path):
        import os
        if not os.path.exists(path):
            return
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._entries.append(LedgerEntry(**json.loads(line)))
        if not self.verify_chain():
            raise ValueError(f"persisted ledger {path!r} failed chain verification (tampered?)")

    # ---- append-only writes ----
    def _append(self, kind: str, cert_id: str, verdict=None, detail=None) -> LedgerEntry:
        with self._lock:                       # read-compute-append must be atomic (else chain breaks)
            prev = self._entries[-1].entry_hash if self._entries else GENESIS
            core = {"seq": len(self._entries), "kind": kind, "cert_id": cert_id,
                    "verdict": verdict, "detail": detail, "at": round(time.time(), 3)}
            e = LedgerEntry(**core, prev_hash=prev, entry_hash=_entry_hash(prev, core))
            self._entries.append(e)
            if self._path:                     # durable append (survives restart)
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(asdict(e), sort_keys=True) + "\n")
            return e

    def record(self, cert) -> LedgerEntry:
        """Record an issued certificate by its manifest hash (its stable id)."""
        return self._append("issue", cert.manifest_sha256, verdict=cert.verdict)

    def revoke(self, cert_id: str, reason: str) -> LedgerEntry:
        """Revoke a previously-ISSUED certificate. Rejects unknown ids so the log can't be polluted
        with revocations of certificates that were never issued."""
        if not any(e.cert_id == cert_id and e.kind == "issue" for e in self._entries):
            raise KeyError(f"cannot revoke unknown cert_id {cert_id!r}")
        return self._append("revoke", cert_id, detail=reason)

    # ---- reads ----
    def status(self, cert_id: str) -> dict:
        logged = any(e.cert_id == cert_id and e.kind == "issue" for e in self._entries)
        rev = next((e for e in self._entries if e.cert_id == cert_id and e.kind == "revoke"), None)
        return {"cert_id": cert_id, "logged": logged, "revoked": rev is not None,
                "reason": (rev.detail if rev else None), "head": self.head()}

    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def size(self) -> int:
        return len(self._entries)

    def verify_chain(self) -> bool:
        """Recompute the hash chain — any tampering with history breaks it."""
        prev = GENESIS
        for e in self._entries:
            core = {"seq": e.seq, "kind": e.kind, "cert_id": e.cert_id,
                    "verdict": e.verdict, "detail": e.detail, "at": e.at}
            if e.prev_hash != prev or e.entry_hash != _entry_hash(prev, core):
                return False
            prev = e.entry_hash
        return True

    def entries(self) -> list:
        return [asdict(e) for e in self._entries]


def check_cert_status(cert, status: dict) -> bool:
    """A verifier's ledger check: the cert must be LOGGED, NOT REVOKED, and the status must be for
    THIS certificate (its manifest hash). Returns True iff trustworthy per the ledger."""
    return bool(status and status.get("cert_id") == cert.manifest_sha256
                and status.get("logged") and not status.get("revoked"))
