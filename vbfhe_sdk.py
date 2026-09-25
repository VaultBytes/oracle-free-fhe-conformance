#!/usr/bin/env python3
"""
vbfhe SDK — minimal encrypted-analytics surface over the software CKKS backend.

Goal: a developer runs encrypted batch scoring in ONE DAY, without touching CKKS params, scale,
rotation, packing, or levels — all hidden behind this surface. Exposes ONLY the passed-kernel
primitives (linear + polynomial). NO SQL, NO top-k, NO attention, NO cross-slot reduction.

PACKING CONVENTION (hidden from the user, stated for the record): FEATURE-MAJOR / batch-in-slots.
One ciphertext holds ONE quantity measured across the whole batch — slot i = record i. So a batch of
B records with D features is D ciphertexts {ct_0..ct_{D-1}}, ct_d carrying feature d of every record.
Consequence: the linear score Σ_d W_d·x_d is a per-slot accumulate of plaintext-weighted ciphertexts
— **NO rotations needed** (each record's score lives in its own slot, independently). Cross-slot
reductions (general matvec, top-k, attention) are intentionally OUT OF SCOPE for this skeleton — they
need rotations/bootstrapping the chip adds later (board=1). board = 0.

The chip backend plugs in by swapping SoftwareCKKS for a hardware-backed CKKS implementing the same
ops (encrypt / add / mul_plain / mul+relin / rescale); the SDK surface above is unchanged.
"""
import numpy as np
from vbfhe_backend_ckks import SoftwareCKKS


class Ct:
    """Opaque encrypted handle. The user never sees scale/level/modulus."""
    def __init__(self, raw, scale, session):
        self._raw = raw; self._scale = scale; self._s = session


class Session:
    """One encrypted-analytics session. Owner holds the secret key (inside the backend)."""

    def __init__(self, max_batch=128, precision_bits=22, backend="software"):
        if backend == "openfhe":
            from vbfhe_backend_openfhe import OpenFHEBackend      # real production CKKS (OpenFHE)
            self.be = OpenFHEBackend(max_batch=max_batch, scale_bits=max(30, precision_bits), depth=4)
        elif backend == "software":
            logN = max(8, (max_batch * 2 - 1).bit_length())       # slots >= max_batch
            self.be = SoftwareCKKS(logN=logN, scale_bits=precision_bits, depth=4, q0_bits=34)
        else:                                                     # a pre-built backend object
            self.be = backend
        self.max_batch = max_batch
        self._trace = []                                    # computation SHAPE (op names + dims), NO values

    @classmethod
    def from_fhetch_trace(cls, trace, backend="software"):
        """Build a certifiable session from a FOREIGN FHE build's FHETCH/OpenFHE trace — the
        interop path behind 'certify any FHE build, including Niobium's'. The trace supplies the
        declared CKKS params + op shape (NO ciphertexts, NO keys). certify() then attests, on a
        reference engine at those params, that the build's parameters + op-shape are conformable to
        FHE algebra. (Certifying the vendor's *actual silicon* uses RemoteConformanceService with
        their backend adapter.)"""
        p = trace.get("params", {}) or {}
        sb = int(p.get("scale_bits") or 22)
        s = cls(max_batch=8, precision_bits=sb, backend=backend)
        ops = [{"op": (o.get("op") or o.get("kind") or "?"), "src": "fhetch"} for o in trace.get("ops", [])]
        s._trace = [{"op": "fhetch_import", "scheme": trace.get("scheme", "CKKS"),
                     "declared_N": int(p.get("N") or p.get("ring_dim") or 0),
                     "declared_scale_bits": sb, "n_ops": len(ops)}] + ops
        return s

    # ---- trace / identity (feeds the conformance certificate; records shape, never data) ----
    def _rec(self, op, **dims):
        self._trace.append({"op": op, **dims})

    @property
    def trace(self):
        return list(self._trace)

    def session_id(self):
        import hashlib, json
        body = json.dumps({"N": int(self.be.N), "scale_bits": int(self.be.scale_bits),
                           "q_bits": int(self.be.q.bit_length()), "trace": self._trace},
                          sort_keys=True).encode()
        return "sess-" + hashlib.sha256(body).hexdigest()[:16]

    def certify(self, results=None, service=None, rounds=8):
        """Certify — WITHOUT decrypting the data — that the engine which produced this session's
        results computes correctly, via FHE's own algebraic laws (oracle-free; PCT filed). Pass the
        output Ct(s) as `results` to BIND the certificate to that exact output. Returns a signed
        Certificate the data owner can hand to a third party — the verb a record-and-trust client
        structurally cannot ship."""
        from vbfhe_conformance import LocalOracleFreeService, ct_digest
        service = service or LocalOracleFreeService()
        digest = None
        if results is not None:
            digest = ct_digest(self.result_items(results))
        return service.certify(self, self._trace, rounds=rounds, result_digest=digest)

    def result_items(self, results):
        """Canonical, backend-correct digest inputs for the output ciphertext(s) — serialized bytes
        (real backends) or limb triples (software). Use with Certificate.binds_result()."""
        from vbfhe_conformance import ct_items
        cts = results if isinstance(results, (list, tuple)) else [results]
        return ct_items(self.be, [c._raw for c in cts])

    def dump_result_items(self, results):
        """JSON-safe form of result_items for the standalone verifier: serialized bytes -> hex string,
        limb triple -> [[int...],[int...],qm]. Round-trips through Certificate.binds_result()."""
        out = []
        for it in self.result_items(results):
            if isinstance(it, (bytes, bytearray)):
                out.append(it.hex())
            else:
                c0, c1, qm = it
                out.append([[int(x) for x in c0], [int(x) for x in c1], int(qm)])
        return out

    # ---- ingest ----
    def encrypt_vector(self, per_record_values):
        """Encrypt one batch-quantity (slot i = record i). Returns an opaque Ct."""
        raw = self.be.encrypt(self.be.encode(np.asarray(per_record_values, float)))
        self._rec("encrypt_vector", n=int(min(len(np.atleast_1d(per_record_values)), self.be.slots)))
        return Ct(raw, self.be.delta, self)

    def encode_plain_matrix(self, weights):
        """Prepare plaintext weights (kept in the clear — they are the model, not the data)."""
        return np.asarray(weights, float)

    # ---- linear primitives (rotation-free, feature-major) ----
    def _ctmod(self, raw):
        return self.be.ct_modulus(raw) if hasattr(self.be, "ct_modulus") else raw[2]

    def _pmul(self, ct, scalar_vec):
        pt = self.be.encode(np.full(self.be.slots, 0.0) + scalar_vec, qm=self._ctmod(ct._raw))
        m = self.be.rescale(self.be.mul_plain(ct._raw, pt))
        return Ct(m, ct._scale, self)

    def inner_product(self, plain_weights, ct_list):
        """Σ_d w_d · ct_d  -> one Ct (per-record inner products). The linear score. No rotations."""
        if not ct_list:
            raise ValueError("inner_product requires a non-empty ct_list")
        w = np.asarray(plain_weights, float)
        acc = None
        for d, ct in enumerate(ct_list):
            term = self._pmul(ct, np.full(self.be.slots, float(w[d])))
            acc = term if acc is None else Ct(self.be.add(acc._raw, term._raw), term._scale, self)
        return acc

    def matvec(self, plain_W, ct_list):
        """Plaintext W [O×D] · encrypted features {ct_d} -> O output Cts (each a per-record value)."""
        W = np.asarray(plain_W, float)
        self._rec("matvec", O=int(W.shape[0]), D=int(W.shape[1]))
        return [self.inner_product(W[o], ct_list) for o in range(W.shape[0])]

    def sum(self, ct_list):
        """Per-record sum across the listed quantities (elementwise add). No rotations."""
        if not ct_list:
            raise ValueError("sum requires a non-empty ct_list")
        acc = ct_list[0]
        for ct in ct_list[1:]:
            acc = Ct(self.be.add(acc._raw, ct._raw), acc._scale, self)
        return acc

    def mean(self, ct_list):
        s = self.sum(ct_list)
        return self._pmul(s, np.full(self.be.slots, 1.0 / len(ct_list)))

    # ---- polynomial primitive (degree-2 calibration; one ct×ct square) ----
    def poly_eval(self, coeffs, ct):
        """Evaluate p(x)=Σ c_k x^k on an encrypted per-record value (degree<=2 supported here).
        c[0] + c[1]·x + c[2]·x²  — one square (ct×ct + relin + rescale), then plaintext-weighted adds."""
        coeffs = list(coeffs)
        assert 1 <= len(coeffs) <= 3, "skeleton supports degree 0..2 (<=1 multiplicative level)"
        self._rec("poly_eval", degree=len(coeffs) - 1)
        if hasattr(self.be, "poly_eval"):          # real backend: native polynomial eval (e.g. OpenFHE)
            return Ct(self.be.poly_eval(ct._raw, coeffs), self.be.delta, self)
        if len(coeffs) == 1 or all(c == 0 for c in coeffs[1:]):   # constant polynomial: c0 (+ 0·x + 0·x²)
            zero = self.be.mul_plain(ct._raw, self.be.encode(np.full(self.be.slots, 0.0), qm=self._ctmod(ct._raw)))
            return Ct(self.be.add_plain(zero, self.be.encode(np.full(self.be.slots, float(coeffs[0])), qm=self._ctmod(zero))), self.be.delta, self)
        # x^2 (rescale back to scale Δ)
        sq = self.be.rescale(self.be.mul(ct._raw, ct._raw))
        # a2*x^2  and  a1*x  at scale Δ, plus a0
        terms = []
        if len(coeffs) == 3 and coeffs[2] != 0:
            terms.append(self.be.rescale(self.be.mul_plain(sq, self.be.encode(np.full(self.be.slots, float(coeffs[2])), qm=sq[2]))))
        if len(coeffs) >= 2 and coeffs[1] != 0:
            terms.append(self.be.rescale(self.be.mul_plain(ct._raw, self.be.encode(np.full(self.be.slots, float(coeffs[1]))))))
        qm = min(t[2] for t in terms)
        acc = terms[0]
        for t in terms[1:]:
            acc = self.be.add([acc[0], acc[1], qm], [t[0], t[1], qm])
        acc = self.be.add_plain(acc, self.be.encode(np.full(self.be.slots, float(coeffs[0])), qm=acc[2]))
        return Ct(acc, self.be.delta, self)

    # ---- egress ----
    def decrypt_to_owner(self, ct, n=None):
        out = self.be.decode_ct(ct._raw, ct._scale)
        return out[:(n or self.max_batch)]
