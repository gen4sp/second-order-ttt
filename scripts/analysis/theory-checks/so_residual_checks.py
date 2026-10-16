"""Symbolic / numeric checks of the reduced model's SO fixed point (exit code 1 on any failure).
S7  R = -D_P/D_FO equals the bracket (16/15) p (1-p_q) eta gamma^2   (virtual copy gain)
S8  conditional copy nullcline: B(1/16) = x/(e^x+15) < 1, B(1) = x, dB/dp > 0, unique root iff x > 1;
    the residual 1 - p* at the root for x = lambda*b*eta in (1.01, 1.1, 2, 10, 40)
S9  Adam: |m_hat|/sqrt(v_hat) <= B_n (Cauchy-Schwarz), sup_n B_n = (1-b1)/sqrt((1-b2)(1-b1^2/b2)) = 1.1650
S10 read model: argmax accuracy under iid Gumbel(0,b) noise on 16 tied-but-one logits = p16(mu/b) (Monte Carlo)
S11 exact corrupted write vs the reduced model's projection at p=.95, x=3 (.58826 vs .61805)
Writes the PASS/FAIL lines to ``--out`` and prints them.

Run: uv run --locked --with sympy python scripts/analysis/theory-checks/so_residual_checks.py
"""
import argparse, math, os, sys
from pathlib import Path
import numpy as np, sympy as sp
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default="outputs/analysis/theory-checks/so_residual_checks.txt")
ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "so-residual-checks"))
args = ap.parse_args()
try:
    import setproctitle; setproctitle.setproctitle(args.process_name)
except Exception: pass
out, ok = [], True
def rec(name, cond, msg):
    global ok
    ok &= bool(cond); out.append(f"{'PASS' if cond else 'FAIL'} {name}: {msg}")

# S7 ------------------------------------------------------------------------
gv, c, eta, g = sp.symbols("gv c eta g", positive=True)
p16 = lambda x: 1 / (1 + 15 * sp.exp(-x))
pk = p16(gv * c)                                  # copy prob with virtual gain on the copy logit
d = sp.Rational(16, 15) * (1 - pk)                # CE write residual (clean, q=1)
mu = eta * g**2 * d
L_copy = -sp.log(pk)
L_q = sp.log(1 + 15 * sp.exp(-mu))
D_FO = sp.diff(L_copy, gv)                        # FO: write detached -> only the outer kv CE sees gv
D_SO = sp.diff(L_copy + L_q, gv)
R = sp.simplify(-(D_SO - D_FO) / D_FO)
pq = p16(mu)
target = sp.Rational(16, 15) * pk * (1 - pq) * eta * g**2
sym0 = sp.simplify((R - target).subs(gv, 1)) == 0
rng = np.random.default_rng(0); errs = []
for _ in range(200):   # 300-digit evaluation: (1-p_q) underflows in float64 when mu is large
    vals = {gv: 1, c: sp.Rational(int(rng.integers(1, 1200)), 100), eta: sp.Rational(int(rng.integers(20, 4000)), 100),
            g: sp.Rational(int(rng.integers(10, 170)), 100)}   # mu <= ~120 (toy range)
    a, b = sp.N(R.subs(vals), 300), sp.N(target.subs(vals), 300)
    errs.append(float(abs(a - b) / abs(b)))
rec("S7 R identity", max(errs) < 1e-40, f"symbolic simplify(R-target)==0: {sym0}; max rel err {max(errs):.1e} over 200 random states (mu<=120) at 300 digits (gv=1)")
rec("S7 signs", all(float(D_FO.subs({gv: 1, c: x, eta: 1, g: 1})) < 0 for x in (.1, 1, 5, 10)), "D_FO < 0 (FO pushes copy up)")

# S8 ------------------------------------------------------------------------
x, p = sp.symbols("x p", positive=True)
A = sp.Rational(16, 15) * x
Bp = A * p * (1 - p16(A * (1 - p)))
rec("S8 B(1/16)", sp.simplify(Bp.subs(p, sp.Rational(1, 16)) - x / (sp.exp(x) + 15)) == 0, "= x/(e^x+15)")
rec("S8 B(1)", sp.simplify(Bp.subs(p, 1) - x) == 0, "= x")
f = sp.lambdify((x, p), Bp, "numpy"); df = sp.lambdify((x, p), sp.diff(Bp, p), "numpy")
P = np.linspace(1/16, 1 - 1e-9, 4001); good_mono = True; roots_ok = True
for xv in np.concatenate([np.linspace(.05, .999, 40), np.linspace(1.001, 60, 120)]):
    good_mono &= bool(np.all(df(xv, P) > 0))
    s = f(xv, P) - 1; nroot = int(np.sum(np.sign(s[1:]) != np.sign(s[:-1])))
    roots_ok &= (nroot == 1) if xv > 1 else (nroot == 0)
rec("S8 dB/dp>0", good_mono, "on p in [1/16,1), x in [.05,60]")
rec("S8 root iff x>1", roots_ok, "exactly one interior root for x>1, none for x<1 (grid)")
res = {}
for xv in (1.01, 1.1, 2, 10, 40):
    s = f(xv, P) - 1; i = int(np.flatnonzero(np.sign(s[1:]) != np.sign(s[:-1]))[0]); lo, hi = P[i], P[i + 1]
    for _ in range(80):
        mid = .5 * (lo + hi); lo, hi = (mid, hi) if f(xv, mid) < 1 else (lo, mid)
    res[xv] = round(float(1 - lo), 5)
rec("S8 residual non-monotone in x", res[2] > res[1.1] and res[10] < res[2], f"1-p* at x: {res} (expected: .00928 .08495 .44969 .41009 .14706)")

# S9 ------------------------------------------------------------------------
b1, b2 = .9, .95; r = b1 * b1 / b2
Bn = lambda n: (1 - b1) / (1 - b1**n) * math.sqrt((1 - b2**n) / (1 - b2)) * math.sqrt((1 - r**n) / (1 - r))
sup = max(Bn(n) for n in range(1, 20001)); lim = (1 - b1) / math.sqrt((1 - b2) * (1 - r))
rec("S9 Adam bound", abs(sup - lim) < 1e-6 and abs(lim - 1.16502) < 1e-4, f"sup_n B_n = {sup:.5f}, limit {lim:.5f}; B_1 = {Bn(1):.3f}")
# worst case attained by g_i proportional to (b1/b2)^(n-i)
n = 3000; gi = (b1 / b2) ** (n - np.arange(1, n + 1))
m = v = 0.0
for k in range(n):
    m = b1 * m + (1 - b1) * gi[k]; v = b2 * v + (1 - b2) * gi[k]**2
ratio = (m / (1 - b1**n)) / math.sqrt(v / (1 - b2**n))
rec("S9 bound attained", abs(ratio - Bn(n)) < 1e-9, f"g_i ~ (b1/b2)^(n-i): ratio {ratio:.5f} = B_n {Bn(n):.5f}")

# S10 -----------------------------------------------------------------------
rg = np.random.default_rng(1); errs = []
for mu_, bb in ((.5, .7), (2., .7), (3.4, .7), (1., 1.), (4.9, 1.), (2., .5)):
    G = rg.gumbel(0, bb, size=(400000, 16)); G[:, 0] += mu_
    mc = float(np.mean(G.argmax(1) == 0)); th = 1 / (1 + 15 * math.exp(-mu_ / bb)); errs.append(abs(mc - th))
rec("S10 Gumbel read", max(errs) < 3e-3, f"max |MC - p16(mu/b)| = {max(errs):.1e} (4e5 draws each)")
# S11 -----------------------------------------------------------------------
pp, xx = .95, 3.0; u = (1 - pp) / 15
exact = 1 / (1 + math.exp(-xx * (1 + pp - u)) + 14 * math.exp(-xx))
proj = 1 / (1 + 15 * math.exp(-xx * 16 / 15 * (1 - u)))
rec("S11 corrupted write", abs(exact - .58826) < 1e-5 and abs(proj - .61805) < 1e-5, f"exact {exact:.5f} vs projection {proj:.5f}")
txt = "\n".join(out) + f"\nALL {'PASS' if ok else 'FAIL'}\n"
Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(txt); print(txt)
sys.exit(0 if ok else 1)
