#!/usr/bin/env python3
"""
vbfhe conformance service over HTTP — the reference network deployment.

Wraps `ConformanceServer` in a stdlib HTTP server so the full ARM-model flow runs over a network, not
just in-process:
    GET  /v0/pubkey    -> {"public_key": <hex>, "algo": "ed25519", "signer": ...}
    POST /v0/certify   -> body = conformance request (descriptor + op-trace + measured bits);
                          returns the signed certificate JSON.

`http_transport(base_url)` is a drop-in for `RemoteConformanceService(transport=...)`. Nothing but
declared params + measured bits crosses the wire — no ciphertexts, no keys, no plaintext.

Reference only: binds to 127.0.0.1, no TLS/auth. Production terminates TLS and authenticates the caller
(the conformance authority is the trust root; its Ed25519 public key is published at /v0/pubkey).
"""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vbfhe_server import ConformanceServer


def _make_handler(server: ConformanceServer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path); path = u.path.rstrip("/")
            if path == "/v0/pubkey":
                self._send(200, {"public_key": server.public_key_hex, "algo": "ed25519",
                                 "signer": server.signer_name})
            elif path == "/v0/status":
                cid = (parse_qs(u.query).get("id") or [""])[0]
                self._send(200, server.status(cid))              # transparency: is this cert logged/revoked?
            elif path == "/v0/log":
                self._send(200, server.signed_head())            # authority-signed ledger head
            else:
                self._send(404, {"error": "not found"})

        MAX_BODY = 256 * 1024                                     # cap request body (DoS guard)

        def _authed(self):
            tok = self.headers.get("Authorization", "")
            if tok.startswith("Bearer "):
                tok = tok[7:]
            return server.authorize(tok or None)

        def do_POST(self):
            path = self.path.rstrip("/")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n < 0 or n > self.MAX_BODY:
                    return self._send(413, {"error": "request too large"})
                body = json.loads(self.rfile.read(n) or b"{}")
                if path == "/v0/challenge":                       # issue a blinded known-answer challenge
                    if not self._authed():
                        return self._send(401, {"error": "unauthorized"})
                    self._send(200, server.issue_challenge(body.get("profile", "workload"),
                                                           body.get("rounds", 6)))
                elif path == "/v0/attest":                        # sound known-answer certification
                    if not self._authed():
                        return self._send(401, {"error": "unauthorized"})
                    self._send(200, server.judge_attested(body))
                elif path == "/v0/certify":                       # legacy self-report path
                    if not self._authed():
                        return self._send(401, {"error": "unauthorized"})
                    self._send(200, server.judge(body))
                elif path == "/v0/revoke":
                    if not self._authed():
                        return self._send(401, {"error": "unauthorized"})
                    e = server.revoke(body["cert_id"], body.get("reason", "unspecified"))
                    self._send(200, {"revoked": body["cert_id"], "seq": e.seq, "head": server.ledger.head()})
                else:
                    self._send(404, {"error": "not found"})
            except Exception:                                    # generic (do not leak schema/internals)
                self._send(400, {"error": "invalid request"})

        def log_message(self, *a):                               # quiet
            pass
    return Handler


class ConformanceHTTPService:
    """Run a ConformanceServer over HTTP(S) on localhost. Use as a context manager; `url` is the base.
    Pass `ssl_context` (an ssl.SSLContext) to serve TLS. If the server has an `api_token`, the mutating
    endpoints require `Authorization: Bearer <token>`."""
    def __init__(self, server: ConformanceServer | None = None, host="127.0.0.1", port=0, ssl_context=None):
        self.server = server or ConformanceServer()
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self.server))
        scheme = "http"
        if ssl_context is not None:
            self._httpd.socket = ssl_context.wrap_socket(self._httpd.socket, server_side=True)
            scheme = "https"
        self.url = f"{scheme}://{host}:{self._httpd.server_address[1]}"
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()


def http_transport(base_url: str, api_token=None, tls_context=None):
    """A transport for RemoteConformanceService / AttestedRemoteConformanceService. Routes by message
    `kind`: challenge -> /v0/challenge, attest -> /v0/attest, else -> /v0/certify. Adds a Bearer token
    if `api_token` is set. Pass `tls_context` (ssl.SSLContext) for HTTPS with a pinned/CA cert."""
    routes = {"challenge": "/v0/challenge", "attest": "/v0/attest"}

    def _transport(request: dict) -> str:
        path = routes.get(request.get("kind"), "/v0/certify")
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        req = urllib.request.Request(base_url.rstrip("/") + path, data=json.dumps(request).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30, context=tls_context) as resp:
            return resp.read().decode()
    return _transport


def fetch_public_key(base_url: str) -> dict:
    """GET the published conformance-authority public key — anyone verifies certificates against it."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/v0/pubkey", timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_status(base_url: str, cert_id: str) -> dict:
    """GET a certificate's transparency status (logged / revoked) from the authority's ledger."""
    from urllib.parse import quote
    url = base_url.rstrip("/") + "/v0/status?id=" + quote(cert_id)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def revoke_cert(base_url: str, cert_id: str, reason: str, api_token=None) -> dict:
    """POST a revocation to the authority. Requires the operator Bearer token if the server sets one."""
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    data = json.dumps({"cert_id": cert_id, "reason": reason}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v0/revoke", data=data,
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())
