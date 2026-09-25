"""RNS-CKKS hybrid key-switch reference (bit-exact, deterministic).

Implements, for one key-switch of an input polynomial d (NTT domain, L limbs):

  iNTT  ->  decompose into dnum digits  ->  ModUp (EXACT base conversion to the
  full L+alpha basis, then NTT)  ->  pointwise MAC with the eval key  ->
  ModDown (divide by P: iNTT P-part, base-convert to Q, subtract, x P^{-1})

plus a standalone rescale op (divide by the last prime). Base conversion is the
EXACT variant -- full CRT reconstruction modulo the (small) source product, then
reduce modulo each destination prime (see Context.base_convert). The cheaper
*approximate* fast-base-conversion (SEAL/HEonGPU hybrid-KS convention) is an
explicit open choice that would produce different bytes (see docs). All constants
(g_i, M/src mod dst, P^{-1}, ...) are host-precomputed, public, key-independent.

Stage snapshots are returned so each RTL block can be verified independently.
"""
from __future__ import annotations
from dataclasses import dataclass
from .modarith import find_ntt_primes, find_psi, modinv, modinv_general
from .ntt import psi_powers_bitrev, ntt_forward, ntt_inverse


@dataclass
class Limb:
    q: int
    psi: int
    psi_rev: list[int]
    ipsi_rev: list[int]
    n_inv: int


class Context:
    """All public parameters + precomputed NTT/base-conversion constants."""

    def __init__(self, N: int, L: int, alpha: int, prime_bits: int = 60):
        self.N = N
        self.L = L
        self.alpha = alpha
        self.two_n = 2 * N
        self.dnum = (L + alpha - 1) // alpha
        # Q-primes (L) + P special primes (alpha), all distinct, == 1 mod 2N.
        primes = find_ntt_primes(L + alpha, prime_bits, self.two_n)
        self.q = primes[:L]          # Q chain
        self.p = primes[L:L + alpha]  # special primes
        self.all_primes = self.q + self.p   # extended basis, length L+alpha
        self.limbs: dict[int, Limb] = {}
        for qi in self.all_primes:
            psi = find_psi(qi, self.two_n)
            self.limbs[qi] = Limb(
                q=qi, psi=psi,
                psi_rev=psi_powers_bitrev(psi, N, qi),
                ipsi_rev=psi_powers_bitrev(modinv(psi, qi), N, qi),
                n_inv=modinv(N, qi),
            )
        # digit groups: digit l covers Q-limb indices [l*alpha : min((l+1)*alpha, L))
        self.groups = [list(range(l * alpha, min((l + 1) * alpha, L)))
                       for l in range(self.dnum)]
        # P^{-1} mod q_i  (for ModDown)
        self.P = 1
        for pj in self.p:
            self.P *= pj
        self.Pinv_modq = [modinv(self.P % qi, qi) for qi in self.q]
        # EXACT base conversion (full CRT reconstruct mod M) = the bit-exact RTL
        # oracle default. Set False for the cheaper *approximate* fast-BConv (skip
        # the mod-M reduction) -- a scheme-owner datapath choice that trades
        # precision for cost; precision_elastic uses it to probe that tradeoff.
        self.exact_bconv = True

    # ---- per-limb NTT helpers ------------------------------------------
    def fwd(self, coeffs: list[int], qi: int) -> list[int]:
        L_ = self.limbs[qi]
        return ntt_forward(coeffs, L_.psi_rev, qi)

    def inv(self, evals: list[int], qi: int) -> list[int]:
        L_ = self.limbs[qi]
        return ntt_inverse(evals, L_.ipsi_rev, L_.n_inv, qi)

    # ---- exact base conversion (full CRT reconstruct mod the small source
    #      product M, then reduce mod each destination prime) ----------------
    # The source product M here is always a small group product (Q_l <= 3 primes
    # for ModUp, or P = alpha primes for ModDown), so M is <=~180-bit and the
    # reconstruction is cheap. Exact (vs the cheaper *approximate* HPS variant)
    # so the key-switch identity holds bit-exactly; the approximate variant is a
    # listed scheme-owner choice (see docs).
    def base_convert(self, src_coeffs: list[list[int]], src_primes: list[int],
                     dst_primes: list[int]) -> list[list[int]]:
        """src_coeffs[i] = coeffs mod src_primes[i] (coefficient domain).
        Returns dst_coeffs[j] = coeffs mod dst_primes[j], exactly."""
        M = 1
        for s in src_primes:
            M *= s
        ghat = []          # g_i = (M/s_i)^{-1} mod s_i
        Mover = []         # M/s_i
        for s in src_primes:
            mover = M // s
            ghat.append(modinv(mover % s, s))
            Mover.append(mover)
        N = self.N
        exact = getattr(self, "exact_bconv", True)
        out = [[0] * N for _ in dst_primes]
        for n in range(N):
            V = 0
            for i in range(len(src_primes)):
                V += ((src_coeffs[i][n] * ghat[i]) % src_primes[i]) * Mover[i]
            if exact:
                a = V % M  # exact integer in [0, M)
                for j, d in enumerate(dst_primes):
                    out[j][n] = a % d
            else:
                # approximate fast-BConv: skip the mod-M reduction -> leaves a
                # bounded q-overflow (k*M, k<#src) that grows the key-switch noise.
                for j, d in enumerate(dst_primes):
                    out[j][n] = V % d
        return out

    def gadget_modall(self, group_idx: int) -> list[int]:
        """CRT idempotent g_l for digit `group_idx`: g_l == 1 (mod Q_group) and
        == 0 (mod the other Q groups), reduced mod every extended-basis prime.
        Satisfies  sum_l [d]_{Q_l} * g_l == d (mod Q)."""
        grp = self.groups[group_idx]
        Ql = 1
        for i in grp:
            Ql *= self.q[i]
        Q = 1
        for qi in self.q:
            Q *= qi
        cofac = Q // Ql
        g = cofac * modinv_general(cofac % Ql, Ql)  # == 1 mod Ql, 0 mod Q/Ql, in [0,Q)
        g %= Q
        return [g % t for t in self.all_primes]


def _det_poly(seed: int, N: int, q: int) -> list[int]:
    """Deterministic pseudo-random polynomial (no RNG import; splitmix64)."""
    x = (seed * 0x9E3779B97F4A7C15 + 0x1234567) & ((1 << 64) - 1)
    out = []
    for _ in range(N):
        x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        z = z ^ (z >> 31)
        out.append(z % q)
    return out


def _ternary(seed: int, N: int) -> list[int]:
    """Deterministic ternary {-1,0,1} secret coefficients."""
    raw = _det_poly(seed, N, 3)   # 0,1,2
    return [v - 1 for v in raw]   # -1,0,1


def _small_centered(seed: int, N: int, bound: int) -> list[int]:
    """Deterministic small centered noise poly with coeffs in [-bound, bound]."""
    if bound <= 0:
        return [0] * N
    raw = _det_poly(seed, N, 2 * bound + 1)   # 0 .. 2*bound
    return [v - bound for v in raw]


def make_inputs(ctx: Context, seed: int = 0xC0FFEE, e_bound: int = 0):
    """Deterministic inputs for ONE real hybrid key-switch (switch d from secret
    s2 to s1).  Returns dict with:
      d_ntt   : input polynomial (NTT, L limbs)              [the c1 to switch]
      evk_a/b : eval key, dnum pairs x (L+alpha) limbs, NTT  [b = -a.s1 + P.g_l.s2 + e_l]
      s1/s2_ntt, s1/s2_coeff : secrets (for verification)
    `e_bound` injects a fresh small centered noise e_l (coeffs in [-e_bound,e_bound])
    into each digit's eval key, as a real RLWE eval key carries noise. Default 0 ->
    the key-switch identity is EXACT (the bit-exact RTL oracle path, unchanged)."""
    allp = ctx.all_primes
    # secret keys (ternary), reduced + NTT'd per extended-basis prime
    s1_coeff = _ternary(seed + 1, ctx.N)
    s2_coeff = _ternary(seed + 2, ctx.N)
    s1_ntt = [ctx.fwd([c % t for c in s1_coeff], t) for t in allp]
    s2_ntt = [ctx.fwd([c % t for c in s2_coeff], t) for t in allp]

    # input d (NTT domain, L limbs)
    d_ntt = [ctx.fwd(_det_poly(seed + 101 * i, ctx.N, ctx.q[i]), ctx.q[i])
             for i in range(ctx.L)]

    evk_a, evk_b = [], []
    for l in range(ctx.dnum):
        gmod = ctx.gadget_modall(l)            # g_l mod each extended prime
        e_l = _small_centered(seed + 500000 + l, ctx.N, e_bound)   # fresh key noise
        a_l, b_l = [], []
        for t, pt in enumerate(allp):
            a = _det_poly(seed + 1000 * l + t + 7, ctx.N, pt)   # random a (NTT)
            Pg = (ctx.P % pt) * gmod[t] % pt                    # (P * g_l) mod pt
            e_ntt = ctx.fwd([c % pt for c in e_l], pt)          # noise in NTT domain
            b = [(-(a[n] * s1_ntt[t][n]) + Pg * s2_ntt[t][n] + e_ntt[n]) % pt
                 for n in range(ctx.N)]                         # b = -a.s1 + P.g_l.s2 + e
            a_l.append(a)
            b_l.append(b)
        evk_a.append(a_l)
        evk_b.append(b_l)
    return {
        "d_ntt": d_ntt, "evk_a": evk_a, "evk_b": evk_b,
        "s1_ntt": s1_ntt, "s2_ntt": s2_ntt,
        "s1_coeff": s1_coeff, "s2_coeff": s2_coeff,
    }


def verify_keyswitch(ctx, inp, out_a, out_b):
    """Check the defining identity  out_b + out_a * s1 == d * s2  (mod Q),
    exactly (e=0, exact base conversion). Returns max abs coefficient error
    after centered-CRT reconstruction; 0 == perfect."""
    q = ctx.q
    Q = 1
    for qi in q:
        Q *= qi
    N = ctx.N
    # LHS per limb (NTT): out_b + out_a*s1 ; RHS per limb: d*s2  -> iNTT to coeff
    lhs_coeff, rhs_coeff = [], []
    for i in range(ctx.L):
        qi = q[i]
        s1 = inp["s1_ntt"][i]
        s2 = inp["s2_ntt"][i]
        d = inp["d_ntt"][i]
        lhs = [(out_b[i][n] + out_a[i][n] * s1[n]) % qi for n in range(N)]
        rhs = [(d[n] * s2[n]) % qi for n in range(N)]
        lhs_coeff.append(ctx.inv(lhs, qi))
        rhs_coeff.append(ctx.inv(rhs, qi))
    # CRT-reconstruct each coeff and compare (centered)
    crt = []
    for qi in q:
        Qi = Q // qi
        crt.append(Qi * modinv(Qi % qi, qi))
    max_err = 0
    for n in range(N):
        L_ = sum(lhs_coeff[i][n] * crt[i] for i in range(ctx.L)) % Q
        R_ = sum(rhs_coeff[i][n] * crt[i] for i in range(ctx.L)) % Q
        diff = (L_ - R_) % Q
        if diff > Q // 2:
            diff -= Q
        if abs(diff) > max_err:
            max_err = abs(diff)
    return max_err


def hybrid_keyswitch(ctx: Context, d_ntt, evk_a, evk_b, emit=None):
    """Full hybrid key-switch. Returns (out_a_ntt, out_b_ntt) [L limbs each].

    Stage snapshots are exposed for independent RTL verification: if `emit` is
    given it is called as emit(stage_name, limbs) per stage (per-digit for the
    large ModUp stages) so the caller can stream-write and free memory; the
    returned snap dict is then empty. Without `emit`, all snapshots are returned
    in the dict (small N only)."""
    q, p, allp = ctx.q, ctx.p, ctx.all_primes
    snap = {}

    def put(name, limbs):
        if emit is not None:
            emit(name, limbs)
        else:
            snap[name] = [list(x) for x in limbs]

    put("s0_input_d_ntt", d_ntt)

    # ---- 1. iNTT to coefficient domain ----
    d_coeff = [ctx.inv(d_ntt[i], q[i]) for i in range(ctx.L)]
    put("s1_d_coeff", d_coeff)

    # ---- 2-3. decompose + ModUp (base-convert each digit to full L+alpha basis, then NTT) ----
    digits_modup_ntt = []
    for l, grp in enumerate(ctx.groups):
        src_primes = [q[i] for i in grp]
        src_coeffs = [d_coeff[i] for i in grp]
        up_coeff = ctx.base_convert(src_coeffs, src_primes, allp)  # L+alpha limbs
        up_ntt = [ctx.fwd(up_coeff[t], allp[t]) for t in range(len(allp))]
        put(f"s2_digit{l:02d}_modup_coeff", up_coeff)
        put(f"s3_digit{l:02d}_modup_ntt", up_ntt)
        digits_modup_ntt.append(up_ntt)

    # ---- 4. pointwise MAC with eval key (NTT domain), over L+alpha limbs ----
    Lp = len(allp)
    res_a = [[0] * ctx.N for _ in range(Lp)]
    res_b = [[0] * ctx.N for _ in range(Lp)]
    for l in range(ctx.dnum):
        for t in range(Lp):
            pt = allp[t]
            dg = digits_modup_ntt[l][t]
            ea, eb = evk_a[l][t], evk_b[l][t]
            ra, rb = res_a[t], res_b[t]
            for n in range(ctx.N):
                ra[n] = (ra[n] + dg[n] * ea[n]) % pt
                rb[n] = (rb[n] + dg[n] * eb[n]) % pt
    put("s4_ks_a_ntt", res_a)
    put("s4_ks_b_ntt", res_b)

    # ---- 5. ModDown each result from L+alpha down to L (divide by P) ----
    def moddown(res):
        # P-part = limbs [L : L+alpha]; iNTT to coeff, base-convert to Q basis
        p_coeff = [ctx.inv(res[ctx.L + j], p[j]) for j in range(ctx.alpha)]
        e_coeff = ctx.base_convert(p_coeff, list(p), list(q))   # L limbs, coeff
        out = []
        for i in range(ctx.L):
            qi = q[i]
            e_ntt = ctx.fwd(e_coeff[i], qi)
            Pinv = ctx.Pinv_modq[i]
            out.append([((res[i][n] - e_ntt[n]) * Pinv) % qi for n in range(ctx.N)])
        return out

    out_a = moddown(res_a)
    out_b = moddown(res_b)
    put("s5_out_a_ntt", out_a)
    put("s5_out_b_ntt", out_b)
    return out_a, out_b, snap


def rescale(ctx: Context, ct_ntt):
    """Rescale an L-limb ciphertext component by the last prime q_{L-1}.
    Returns (L-1)-limb result + snapshot."""
    L = ctx.L
    q = ctx.q
    last = q[L - 1]
    t_coeff = ctx.inv(ct_ntt[L - 1], last)  # last limb -> coeff
    snap = {"r0_last_coeff": list(t_coeff)}
    out = []
    for i in range(L - 1):
        qi = q[i]
        e_ntt = ctx.fwd([c % qi for c in t_coeff], qi)
        inv_last = modinv(last % qi, qi)
        out.append([((ct_ntt[i][n] - e_ntt[n]) * inv_last) % qi for n in range(ctx.N)])
    snap["r1_rescaled_ntt"] = [list(x) for x in out]
    return out, snap
