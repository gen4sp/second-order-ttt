"""Symbolic checks (SymPy) for the key-pair softmax model with a copy path.

Model:
  V values; per sequence K key->value pairs; orthonormal key codes (pairs decouple);
  kv position of pair i: pre-write logits z_kv = a * e_{c_i}, c_i the copied token
  (clean: c_i = v_i; noisy: c_i = v_i w.p. q, else uniform over the other V-1 values);
  inner CE write at kv:  m_i = eta * (e_{v_i} - p_i),  p_i = softmax(a e_{c_i});
  query logits z_q,i = b * m_i;  outer loss L = E CE(z_kv, v) + lam * E CE(z_q, v).
FO: the write m_i is data (stop-gradient) -> a-field from the kv term only.
SO: full derivative.

Every identity is checked by sympy.simplify (== 0) and by random numeric substitution (40 random
states per identity; the generator is seeded from Python's string hash of the identity's name, so
``numeric_max_abs`` changes with PYTHONHASHSEED while ``symbolic_zero`` and ``passed`` do not).
Writes the JSON to ``--out``.

Run: uv run --locked --with sympy python scripts/analysis/theory-checks/key_pair_symbolic.py
"""
import argparse
import json
import os

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default="outputs/analysis/theory-checks/key_pair_symbolic_check.json")
ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "key-pair-symbolic"))
args = ap.parse_args()
try:
    import setproctitle
    setproctitle.setproctitle(args.process_name)
except Exception:
    pass

import numpy as np
import sympy as sp

OUT = args.out
res = {}

a, V, eta, b, lam, s, q, m, t = sp.symbols("a V eta b lam s q m t", positive=True)
kappa = b * eta * V / (V - 1)


def check(name, expr, subs_list=None, tol=1e-10):
    """expr should be identically zero."""
    simp = sp.simplify(expr)
    num_ok = True
    worst = 0.0
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for _ in range(40):
        vals = {a: rng.uniform(-3, 8), V: int(rng.integers(2, 40)), eta: rng.uniform(.05, 50),
                b: rng.uniform(.05, 5), lam: rng.uniform(.05, 5), s: rng.uniform(.001, .999),
                q: rng.uniform(.06, .999), m: rng.uniform(0, 20), t: rng.uniform(0, 100)}
        if subs_list:
            vals.update(subs_list)
        for extra in expr.free_symbols:
            if extra not in vals:
                vals[extra] = rng.uniform(-3, 8)
        v = complex(sp.N(expr.subs(vals), 30))
        worst = max(worst, abs(v))
        if abs(v) > tol:
            num_ok = False
    ok = (simp == 0) or num_ok
    res[name] = dict(symbolic_zero=bool(simp == 0), numeric_max_abs=float(worst), passed=bool(ok))
    print(f"{name:60s} symbolic={simp == 0} numeric_max={worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---- 1. clean copy, reduced coordinates ---------------------------------------------------
sig = sp.exp(a) / (sp.exp(a) + V - 1)            # p_v at kv
s_of_a = 1 - sig                                   # kv residual s = 1 - p_v = (V-1)/(e^a+V-1)
lW = sp.log(sp.exp(a) + V - 1) - a                 # CE(a e_v, v)
check("P3.1 kv CE equals -log(1-s)", lW + sp.log(1 - s_of_a))
check("P3.2 FO a-field G_FO = dlW/da = -s", sp.diff(lW, a) + s_of_a)
check("P3.3 ds/da = -s(1-s)", sp.diff(s_of_a, a) + s_of_a * (1 - s_of_a))

# explicit vectors for V=5 and V=16: write e_v - p and query margins
for Vn in (5, 16):
    av = sp.symbols("av", positive=True)
    p = [sp.exp(av) / (sp.exp(av) + Vn - 1)] + [1 / (sp.exp(av) + Vn - 1)] * (Vn - 1)
    w = [1 - p[0]] + [-pj for pj in p[1:]]               # e_v - p, v = index 0
    sv = 1 - p[0]
    check(f"P3.4 (V={Vn}) e_v-p = s*(e_v - (1-e_v)/(V-1))",
          sum((w[j] - (sv if j == 0 else -sv / (Vn - 1))) ** 2 for j in range(Vn)))
    check(f"P3.5 (V={Vn}) ||e_v-p||^2 = s^2 V/(V-1)",
          sum(x ** 2 for x in w) - sv ** 2 * Vn / (Vn - 1))
    # query logits b*eta*w: margin z_v - z_j = kappa*s
    zq = [b * eta * x for x in w]
    check(f"P3.6 (V={Vn}) query margin = b eta V/(V-1) s", (zq[0] - zq[1]) - b * eta * Vn / (Vn - 1) * sv)
    CEq = sp.log(sum(sp.exp(z) for z in zq)) - zq[0]
    Qred = sp.log(1 + (Vn - 1) * sp.exp(-b * eta * Vn / (Vn - 1) * sv))
    check(f"P3.7 (V={Vn}) query CE = log(1+(V-1)exp(-kappa s))", CEq - Qred,
          subs_list=None)

# ---- 2. SO a-field and threshold ------------------------------------------------------------
pi = lambda mm: 1 / (1 + (V - 1) * sp.exp(-mm))     # query p_v at margin mm
Q = sp.log(1 + (V - 1) * sp.exp(-kappa * s))        # query CE as function of s
f = lam * kappa * (1 - s) * (1 - pi(kappa * s))
# SO a-field = d/da [lW + lam*Q(s(a))]
Q_a = Q.subs(s, s_of_a)
G_SO = sp.diff(lW + lam * Q_a, a)
check("P4.1 G_SO = s (f(s) - 1)", G_SO - (s * (f - 1)).subs(s, s_of_a))
check("P4.2 f(0) = lam b eta", f.subs(s, 0) - lam * b * eta)
fprime = sp.diff(f, s)
fprime_claim = -lam * kappa * ((1 - pi(kappa * s)) + (1 - s) * kappa * pi(kappa * s) * (1 - pi(kappa * s)))
check("P4.3 f'(s) = -lam kappa[(1-pi) + (1-s) kappa pi (1-pi)] (<0)", fprime - fprime_claim)
# stationary margin: f(s*) = 1  <=>  (V-1) e^{-m*} = 1/(lam kappa (1-s*) - 1)
E = (V - 1) * sp.exp(-kappa * s)
check("P4.4 f(s)=1 <=> E*(lam kappa(1-s)-1) = 1  [identity f-1 = (E(lk(1-s)-1)-1)/(1+E)]",
      (f - 1) - (E * (lam * kappa * (1 - s) - 1) - 1) / (1 + E))
# boundary-slope identity: G_a = -s(1-s) P'(s), P(s) = -log(1-s) + lam Q(s)
P = -sp.log(1 - s) + lam * Q
check("P5.1 G_SO(a) = -s(1-s) dP/ds", G_SO - (-s * (1 - s) * sp.diff(P, s)).subs(s, s_of_a))
check("P5.2 P'(0+) = 1 - lam b eta", sp.diff(P, s).subs(s, 0) - (1 - lam * b * eta))
# squared-loss (P0.2) special case at rho = 1: P_sq(r) = r^2/2 + lam/2 (eta r - 1)^2
r = sp.symbols("r", real=True)
Psq = r ** 2 / 2 + lam / 2 * (eta * r - 1) ** 2
check("P5.3 squared loss: P'(0) = -lam eta (<0 always)", sp.diff(Psq, r).subs(r, 0) + lam * eta)
rstar = sp.solve(sp.diff(Psq, r), r)[0]
check("P5.4 squared loss: r* = lam eta/(1+lam eta^2) -> a_SO = 1 - r* (P0.2 at rho=1)",
      rstar - lam * eta / (1 + lam * eta ** 2))

# ---- 3. noisy copy: FO calibration, Bayes residual, clean vs corrupted margins ----------------
lW_noisy = sp.log(sp.exp(a) + V - 1) - a * q          # E CE(a e_c, v), P(c=v)=q
a_FO = sp.log(q * (V - 1) / (1 - q))
check("P6.1 FO a-field (noisy) = sigma_V(a) - q", sp.diff(lW_noisy, a) - (sig - q))
check("P6.2 a_FO = log(q(V-1)/(1-q)) zeroes it", (sig - q).subs(a, a_FO))
lam_ch = (V * q - 1) / (V - 1)
bayes = 1 - q ** 2 - (1 - q) ** 2 / (V - 1)          # E||e_v - p||^2 at calibrated p
check("P6.3 E||e_v-p||^2 at FO point = (1-lam_ch^2)(V-1)/V (= measured 'Bayes residual')",
      bayes - (1 - lam_ch ** 2) * (V - 1) / V)
for Vn in (16,):
    qq = sp.Rational(97, 100)
    pc = [qq] + [(1 - qq) / (Vn - 1)] * (Vn - 1)        # p at FO point, copy token index 0
    # clean event: v = 0 ; corrupted: v = 1
    wc = [1 - pc[0]] + [-x for x in pc[1:]]
    wk = [-pc[0], 1 - pc[1]] + [-x for x in pc[2:]]
    mc = (wc[0] - wc[2])                                # clean: z_v - z_j (per b eta)
    mk_j = (wk[1] - wk[2])                              # corrupted: z_v - z_j
    mk_c = (wk[1] - wk[0])                              # corrupted: z_v - z_c
    res["P6.4 margins per (b eta), V=16, q=.97"] = dict(clean=float(mc), corrupted_vs_other=float(mk_j),
                                                        corrupted_vs_copy=float(mk_c))
    check("P6.4 clean margin = (1-q)V/(V-1); corrupted margin vs others = 1",
          (mc - (1 - qq) * Vn / (Vn - 1)) ** 2 + (mk_j - 1) ** 2, tol=1e-20)

# ---- 4. Perdomo constants (a- and s-coordinates) ---------------------------------------------
gamma_a = sp.diff(lW, a, 2)
check("P7.1 curvature of kv loss in a: gamma(a) = s(1-s) -> 0 as s->0",
      gamma_a - (s * (1 - s)).subs(s, s_of_a))
eps_a = eta * sp.sqrt(V / (V - 1)) * s * (1 - s)                 # |d m / d a| (Euclidean)
eps_s = eta * sp.sqrt(V / (V - 1))                               # |d m / d s|
Lz0 = lam * b * sp.sqrt((V - 1) / V)                             # ||grad_m lam Q|| at m=0
check("P7.2 Thm4.3 ratio 2 Lz eps/gamma (a-coords, at s) = 2 lam b eta", 2 * Lz0 * eps_a / (s * (1 - s)) - 2 * lam * b * eta)
check("P7.3 Thm4.3 ratio (s-coords, gamma_s(0)=1) = 2 lam b eta", 2 * Lz0 * eps_s / 1 - 2 * lam * b * eta)

# ---- 5. FO race with a free log-step (no interference) ----------------------------------------
# gradient flow: a' = s (FO), zeta' = -dL/dzeta = lam * m * (1 - pi(m)), m = kappa s
zeta = sp.symbols("zeta", real=True)
kap_z = b * sp.exp(zeta) * V / (V - 1)
Qz = sp.log(1 + (V - 1) * sp.exp(-kap_z * s))
dQ_dzeta = sp.diff(Qz, zeta)
mz = kap_z * s
check("P8.1 FO zeta-field: -dQ/dzeta = m (1 - pi(m))", -dQ_dzeta - mz * (1 - pi(mz)))
# d log m/dt = zeta' + d log s/dt with ds/dt = -(1-s) s * a' = -(1-s) s^2
dlogm = lam * mz * (1 - pi(mz)) - (1 - s) * s
small = sp.series((lam * mz * (1 - pi(mz))).subs(sp.exp(zeta), eta), s, 0, 2).removeO()
check("P8.2 small-m: lam m (1-pi(m)) = lam b eta s + O(s^2)", sp.simplify(small - lam * b * eta * s))
res["P8.3 note"] = ("d log m/dt = s(lam b eta - 1 + s) + O(s^2): the FO write margin grows under a free "
                    "log-step iff lam b eta > 1 - s; eta' = lam b eta^2 s (1+O(s)) > 0 always.")

allpass = all(v["passed"] for v in res.values() if isinstance(v, dict) and "passed" in v)
res["ALL_PASSED"] = bool(allpass)
print("ALL_PASSED", allpass)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
