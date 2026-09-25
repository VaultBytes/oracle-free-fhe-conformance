#!/usr/bin/env python3
"""
vbfhe conformance — certificate signing.

A conformance certificate is only a *portable trust artifact* if a third party (a regulator, an
auditor, a counterparty) can verify it WITHOUT any shared secret. That means asymmetric signatures:
the VaultBytes conformance service signs with a PRIVATE key; anyone verifies with the PUBLIC key,
which is embedded in the certificate itself.

Primary: Ed25519 (via `cryptography`). Fallback: HMAC-SHA256 (symmetric — verifier needs the key),
so the skeleton still runs where `cryptography` is absent. The certificate records which was used.
"""
from __future__ import annotations

import hashlib
import hmac

ALGO_ED25519 = "ed25519"
ALGO_HMAC = "hmac-sha256"

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAVE_ED = True
except Exception:                        # pragma: no cover
    _HAVE_ED = False


class Signer:
    algo = None
    public_key_hex = None
    def sign(self, body: bytes) -> str:
        raise NotImplementedError


class Ed25519Signer(Signer):
    """Asymmetric signer — certificates verify against the embedded public key, no secret shared."""
    algo = ALGO_ED25519

    def __init__(self, seed32: bytes | None = None):
        if not _HAVE_ED:
            raise RuntimeError("cryptography/Ed25519 unavailable")
        if seed32 is not None:
            self._sk = Ed25519PrivateKey.from_private_bytes(seed32[:32].ljust(32, b"\0"))
        else:
            self._sk = Ed25519PrivateKey.generate()
        pk = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key_hex = pk.hex()

    def sign(self, body: bytes) -> str:
        return self._sk.sign(body).hex()


class HmacSigner(Signer):
    """Fallback symmetric signer — the verifier must hold the same key. Not a portable artifact."""
    algo = ALGO_HMAC

    def __init__(self, key: bytes):
        self._key = key
        self.public_key_hex = None       # symmetric: nothing public to embed

    def sign(self, body: bytes) -> str:
        return hmac.new(self._key, body, hashlib.sha256).hexdigest()


def verify_signature(algo: str, body: bytes, signature_hex: str,
                     public_key_hex: str | None, hmac_key: bytes | None = None) -> bool:
    """Verify a certificate signature. Ed25519 needs only the embedded public key (true third-party
    verification); HMAC needs the shared key passed in."""
    if algo == ALGO_ED25519:
        if not _HAVE_ED or not public_key_hex:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
                bytes.fromhex(signature_hex), body)
            return True
        except Exception:
            return False
    if algo == ALGO_HMAC:
        if hmac_key is None:
            return False
        expect = hmac.new(hmac_key, body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expect, signature_hex)
    return False


# A PUBLIC, deterministic demo key. NEVER use it as a real authority key — anyone can derive it from
# this source. It is used ONLY when VBFHE_DEMO_KEY is set (tests/demos want reproducible signatures).
_DEMO_ED25519_SEED = hashlib.sha256(b"vbfhe-conformance-demo-ed25519-seed").digest()
_DEMO_HMAC_KEY = hashlib.sha256(b"vbfhe-conformance-demo-service-key").digest()


def load_signer_from_seed_hex(seed_hex: str) -> Signer:
    """Load an Ed25519 signer from a 32-byte seed hex (e.g. from a secrets manager / KMS export)."""
    if not _HAVE_ED:
        raise RuntimeError("Ed25519 unavailable")
    return Ed25519Signer(seed32=bytes.fromhex(seed_hex))


def default_signer() -> Signer:
    """Production-safe default. Key selection, in order:
      1. VBFHE_SIGNING_KEY  — hex 32-byte Ed25519 seed, or a path to a file containing it (KMS/HSM export).
      2. VBFHE_DEMO_KEY set — the PUBLIC demo key (reproducible; tests/demos only, NEVER production).
      3. otherwise           — a fresh EPHEMERAL random key (secure default; the demo key is never used
                               implicitly). Publish `signer.public_key_hex` so verifiers can pin it.
    """
    import os
    env = os.environ.get("VBFHE_SIGNING_KEY")
    if env:
        seed_hex = env
        if os.path.exists(env):
            with open(env) as fh:
                seed_hex = fh.read().strip()
        return load_signer_from_seed_hex(seed_hex)
    if os.environ.get("VBFHE_DEMO_KEY"):
        return Ed25519Signer(seed32=_DEMO_ED25519_SEED) if _HAVE_ED else HmacSigner(_DEMO_HMAC_KEY)
    if _HAVE_ED:
        return Ed25519Signer()                       # ephemeral, cryptographically random
    return HmacSigner(hashlib.sha256(os.urandom(32)).digest())
