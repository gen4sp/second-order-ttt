"""Reduced model: a minimal order-parameter model of first-order write starvation in the echo toy.

Pure numpy, no torch, no project code.

State (per training step t of the toy's outer loop):
  s      copy-circuit factor; copy logit margin at the kv position is c = s**2
  gamma  memory gain factor; the memory's logit margin at the query is
         mu = eta_eff * gamma**2 * d, gamma appearing once on the write side and once
         on the read side (same physical weights, FO detaches the write-side copy)
  rho    log(eta_kv / eta_base) for the kv-only step-size arms (Adam, like the toy)

d is the value-discriminative inner residual at the kv write (CE: 16/15 (1 - p_v);
lin and AICE in chunk 0: 1).  The outer loss per key is
  L = w_kv [ CE_copy(c; q) + CE_class(gamma) ] + CE_class(gamma) + CE_query(mu)
with CE_query(mu) = log(1 + 15 exp(-mu)), CE_class(g) = -log(16 e^{beta g}/(16 e^{beta g}+34)),
CE_copy with a 16-value copy channel of fidelity q.  First order (FO) holds the write
(gamma_write, d) fixed when differentiating; second order (SO) differentiates through it.
Updates: preconditioned gradient descent for s and gamma with rates kappa_c, kappa_m times
the toy's LR schedule; Adam (toy hyperparameters) for rho.

Used as a module by ``reduced_model_predictions.py``; run directly it calibrates the two rates
for the given conventions:  uv run --locked python scripts/analysis/theory-checks/reduced_model.py --help
"""
from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import setproctitle

    setproctitle.setproctitle(os.environ.get("PROCESS_NAME", "reduced-model"))
except Exception:  # pragma: no cover
    pass

N_VAL = 16          # values (echo: 16 keys, 16 values)
N_VOCAB = 50        # 2 markers + 16 keys + 16 values + 16 fillers
MU_STAR = math.log(0.9 / 0.1 * (N_VAL - 1))   # p_q(mu*) = 0.9  ->  4.905


# --------------------------------------------------------------------------- schedule
def lr_scale(step: int, warmup: int = 200, total: int = 3000, floor: float = 0.1) -> float:
    """The toy's schedule (``tttmem.exp_ttt.lr_scale``): linear warmup, cosine to floor."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


# --------------------------------------------------------------------------- params
@dataclass(frozen=True)
class Params:
    """Two calibrated rates and three declared conventions."""

    kappa_c: float = 0.02     # copy rate        (calibrated: anchor loss time)
    kappa_m: float = 0.02     # memory rate      (calibrated: anchor acquisition time)
    s0: float = 0.1           # convention: initial copy factor (c0 = 0.01)
    gamma0: float = 0.3       # convention: initial memory gain factor (post-collapse plateau)
    beta: float = 1.0         # convention: class-logit gain of gamma


@dataclass(frozen=True)
class Arm:
    name: str
    order: str = "FO"            # "FO" | "SO"
    rule: str = "CE"             # "CE" | "lin" (lin at kv) | "aice" (== lin at kv in chunk 0)
    kv_outer: bool = True        # False: outer loss masked at kv targets (kvm)
    copy_edge: bool = True       # False: copy edge masked in every layer (a5 / echo_ctrl)
    q: float = 1.0               # copy-channel fidelity (cn99 -> .99, ...)
    noise_onset: int = 0         # step at which copy noise starts (hysteresis test)
    eta_hat: float = 1.0         # fixed shared eta / 10  (eta_lr = 0 regime)
    kv_eta_lr: float | None = None   # Adam lr of rho_kv (kveta .01, kveta10 .1); None: no kv eta
    steps: int = 1500
    warmup: int = 200
    lr_total: int = 3000
    kv_outer_off_until: int = 0  # >0: kv outer loss masked until this step (unused by default)
    lin_from: int | None = None  # step at which the kv inner loss switches CE -> lin (hysteresis)


# --------------------------------------------------------------------------- pieces
def p16(x):
    """Probability of the favoured value among 16 when it has logit margin x over 15 others."""
    return 1.0 / (1.0 + (N_VAL - 1) * np.exp(-x))


def p_class(g, beta):
    """P(next token is a value) with class-logit gap beta*g over a uniform 50-token start."""
    e = N_VAL * np.exp(beta * g)
    return e / (e + (N_VOCAB - N_VAL))


def residual_d(c, cls, rule):
    """Value-discriminative inner residual at the kv write (normalised so d=1 at uniform)."""
    if rule in ("lin", "aice"):
        return 1.0, 0.0
    pc = p16(c)
    dpc = pc * (1 - pc)
    if cls == "clean":       # copy points to v
        return (16 / 15) * (1 - pc), -(16 / 15) * dpc
    # corrupted: copy points to r != v, p_v = (1 - p_c)/15
    return (16 / 15) * (1 - (1 - pc) / 15), (16 / 225) * dpc


@dataclass
class Trace:
    step: np.ndarray
    c: np.ndarray
    gamma: np.ndarray
    eta_eff: np.ndarray
    p_kv: np.ndarray          # p(v) at kv for clean keys
    d_clean: np.ndarray
    mu_clean: np.ndarray
    mu_corr: np.ndarray
    acc: np.ndarray           # accuracy proxy: key-weighted p_q(mu)
    acc_clean: np.ndarray
    acc_corr: np.ndarray

    def bound(self, thr: float = 0.9):
        return self.acc >= thr


def simulate(arm: Arm, P: Params, record_every: int = 1, substeps: int = 1) -> Trace:
    """Euler integration, one update per toy step (``substeps`` > 1 only for the
    discretisation check: the step's LR is split into equal sub-updates; Adam is not split)."""
    if substeps > 1:
        return _simulate_sub(arm, P, substeps)
    s = 0.0 if not arm.copy_edge else P.s0
    g = P.gamma0
    rho, m1, v1 = 0.0, 0.0, 0.0
    b1, b2, eps = 0.9, 0.95, 1e-8
    so = arm.order == "SO"
    rows = []
    for t in range(arm.steps):
        ell = lr_scale(t, arm.warmup, arm.lr_total)
        q_t = arm.q if t >= arm.noise_onset else 1.0
        rule = arm.rule if (arm.lin_from is None or t < arm.lin_from) else "lin"
        w_kv = 1.0 if (arm.kv_outer and t >= arm.kv_outer_off_until) else 0.0
        c = s * s
        eta_eff = arm.eta_hat * math.exp(rho)
        classes = [("clean", q_t)] + ([("corr", 1 - q_t)] if q_t < 1 else [])
        G_g = -(w_kv + 1.0) * P.beta * (1 - p_class(g, P.beta))
        G_c = w_kv * (p16(c) - q_t) if arm.copy_edge else 0.0
        G_rho = 0.0
        mus, accs = {}, {}
        for cls, pi in classes:
            d, dd = residual_d(c if arm.copy_edge else 0.0, cls, rule)
            mu = eta_eff * g * g * d
            dl = -(1 - p16(mu))                      # dCE_query/dmu
            G_g += pi * dl * (2.0 if so else 1.0) * eta_eff * g * d
            if so and arm.copy_edge:
                G_c += pi * dl * eta_eff * g * g * dd
            G_rho += pi * dl * mu
            mus[cls], accs[cls] = mu, p16(mu)
        if t % record_every == 0:
            pc = p16(c) if arm.copy_edge else 1 / N_VAL
            acc = sum(pi * accs[cls] for cls, pi in classes)
            rows.append((t, c, g, eta_eff, pc, residual_d(c if arm.copy_edge else 0.0, "clean", rule)[0],
                         mus["clean"], mus.get("corr", np.nan), acc, accs["clean"],
                         accs.get("corr", np.nan)))
        # updates
        s = max(0.0, s - P.kappa_c * ell * 2 * s * G_c) if arm.copy_edge else 0.0
        g = max(0.0, g - P.kappa_m * ell * G_g)
        if arm.kv_eta_lr is not None:
            n = t + 1
            m1 = b1 * m1 + (1 - b1) * G_rho
            v1 = b2 * v1 + (1 - b2) * G_rho * G_rho
            rho -= arm.kv_eta_lr * ell * (m1 / (1 - b1 ** n)) / (math.sqrt(v1 / (1 - b2 ** n)) + eps)
    a = np.array(rows, dtype=float)
    return Trace(*(a[:, i] for i in range(a.shape[1])))


def _gradients(arm: Arm, P: Params, s: float, g: float, rho: float, t: int):
    """(G_s, G_gamma, G_rho, record) at one state; used by the sub-stepped integrator."""
    q_t = arm.q if t >= arm.noise_onset else 1.0
    rule = arm.rule if (arm.lin_from is None or t < arm.lin_from) else "lin"
    w_kv = 1.0 if (arm.kv_outer and t >= arm.kv_outer_off_until) else 0.0
    so = arm.order == "SO"
    c = s * s
    eta_eff = arm.eta_hat * math.exp(rho)
    classes = [("clean", q_t)] + ([("corr", 1 - q_t)] if q_t < 1 else [])
    G_g = -(w_kv + 1.0) * P.beta * (1 - p_class(g, P.beta))
    G_c = w_kv * (p16(c) - q_t) if arm.copy_edge else 0.0
    G_rho, acc = 0.0, 0.0
    for cls, pi in classes:
        d, dd = residual_d(c if arm.copy_edge else 0.0, cls, rule)
        mu = eta_eff * g * g * d
        dl = -(1 - p16(mu))
        G_g += pi * dl * (2.0 if so else 1.0) * eta_eff * g * d
        if so and arm.copy_edge:
            G_c += pi * dl * eta_eff * g * g * dd
        G_rho += pi * dl * mu
        acc += pi * p16(mu)
    return 2 * s * G_c, G_g, G_rho, acc


def _simulate_sub(arm: Arm, P: Params, substeps: int) -> Trace:
    s = 0.0 if not arm.copy_edge else P.s0
    g, rho, m1, v1 = P.gamma0, 0.0, 0.0, 0.0
    b1, b2, eps = 0.9, 0.95, 1e-8
    steps, accs, cs = [], [], []
    for t in range(arm.steps):
        ell = lr_scale(t, arm.warmup, arm.lr_total) / substeps
        _, _, G_rho, acc = _gradients(arm, P, s, g, rho, t)
        steps.append(t); accs.append(acc); cs.append(s * s)
        for _ in range(substeps):
            G_s, G_g, _, _ = _gradients(arm, P, s, g, rho, t)
            s = max(0.0, s - P.kappa_c * ell * G_s) if arm.copy_edge else 0.0
            g = max(0.0, g - P.kappa_m * ell * G_g)
        if arm.kv_eta_lr is not None:
            n = t + 1
            m1 = b1 * m1 + (1 - b1) * G_rho
            v1 = b2 * v1 + (1 - b2) * G_rho * G_rho
            rho -= arm.kv_eta_lr * ell * substeps * (m1 / (1 - b1 ** n)) / (math.sqrt(v1 / (1 - b2 ** n)) + eps)
    z = np.full(len(steps), np.nan)
    st, acc, c = np.array(steps, float), np.array(accs), np.array(cs)
    return Trace(st, c, z, z, p16(c), z, z, z, acc, z, z)


# --------------------------------------------------------------------------- events
def events(tr: Trace, grid: int = 25, thr: float = 0.9) -> dict:
    """Exact and probe-grid (every 25 steps, like the toy's quick eval) event times."""
    out = {}
    for name, sel in (("exact", np.ones_like(tr.step, bool)),
                      ("grid", (tr.step + 1) % grid == 0)):
        st = tr.step[sel] + (1 if name == "grid" else 0)   # grid: completed steps 25, 50, ...
        b = tr.acc[sel] >= thr
        idx = np.flatnonzero(b)
        first = float(st[idx[0]]) if idx.size else None
        lose = rebind = None
        if idx.size:
            after = np.flatnonzero(~b[idx[0]:]) + idx[0]
            if after.size:
                lose = float(st[after[0]])
                again = np.flatnonzero(b[after[0]:]) + after[0]
                rebind = float(st[again[0]]) if again.size else None
        out[name] = dict(first_bound=first, first_loss=lose, rebind=rebind,
                         final_bound=bool(b[-1]), ever_bound=bool(idx.size))
        if idx.size and lose is not None:
            seg = tr.acc[sel][np.flatnonzero(~b[idx[0]:]) [0] + idx[0]:]
            if rebind is not None:
                seg = tr.acc[sel][after[0]:again[0]]
            out[name]["dip_min_acc"] = float(seg.min())
    out["final_acc"] = float(tr.acc[-1])
    out["final_p_kv"] = float(tr.p_kv[-1])
    out["final_d_clean"] = float(tr.d_clean[-1])
    out["final_eta_eff"] = float(tr.eta_eff[-1])
    out["final_mu_clean"] = float(tr.mu_clean[-1])
    return out


# --------------------------------------------------------------------------- calibration
ANCHOR = Arm("anchor e2e1emb (FO, CE, fixed eta)")
# declared targets: probe-grid medians 212.5 (first bound) and 275 (first loss) of the
# FO-CE fixed-eta anchor runs, minus half a probe interval -> exact crossings 200 and 262.5
T_ACQ, T_LOSE = 200.0, 262.5


def crossing_times(P: Params) -> tuple[float, float]:
    ev = events(simulate(ANCHOR, P))["exact"]
    big = 5000.0
    return (ev["first_bound"] if ev["first_bound"] is not None else big,
            ev["first_loss"] if ev["first_loss"] is not None else big)


def _solve_m(kc: float, s0: float, gamma0: float, beta: float, tol_t: float = 1.5):
    """kappa_m with exact anchor acquisition at T_ACQ for a given kappa_c, or None.

    t_acq(kappa_m) is 'never' below a jump and decreasing above it; a solution exists only
    if the crossing of T_ACQ is continuous (|t_acq - T_ACQ| <= tol_t at the boundary)."""
    lo, hi = math.log(1e-5), math.log(30.0)
    if crossing_times(Params(kc, math.exp(hi), s0, gamma0, beta))[0] > T_ACQ:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if crossing_times(Params(kc, math.exp(mid), s0, gamma0, beta))[0] > T_ACQ:
            lo = mid
        else:
            hi = mid
    P = Params(kc, math.exp(hi), s0, gamma0, beta)
    return P if abs(crossing_times(P)[0] - T_ACQ) <= tol_t else None


def calibrate_info(s0: float, gamma0: float, beta: float, n_grid: int = 36):
    """Grid-bracketed solve of (t_acq, t_lose) = (T_ACQ, T_LOSE) for (kappa_c, kappa_m).

    Scan log kappa_c, solve kappa_m on each grid point, bracket sign changes of
    t_lose - T_LOSE between feasible neighbours and bisect inside each bracket.
    Returns (Params or None, info) where info lists every solution found."""
    grid = np.geomspace(1e-4, 3.0, n_grid)
    feas = []
    for kc in grid:
        P = _solve_m(float(kc), s0, gamma0, beta)
        if P is not None:
            feas.append((math.log(kc), P, crossing_times(P)[1] - T_LOSE))
    sols = []
    for (l0, P0, f0), (l1, P1, f1) in zip(feas, feas[1:]):
        if f0 == 0:
            sols.append(P0)
            continue
        if f0 * f1 >= 0 or abs(l1 - l0) > 1.5 * (math.log(grid[1]) - math.log(grid[0])):
            continue
        a, b, fa = l0, l1, f0
        Pm = None
        for _ in range(30):
            m = 0.5 * (a + b)
            Pm = _solve_m(math.exp(m), s0, gamma0, beta)
            if Pm is None:
                break
            fm = crossing_times(Pm)[1] - T_LOSE
            if fa * fm > 0:
                a, fa = m, fm
            else:
                b = m
        if Pm is not None:
            ta, tl = crossing_times(Pm)
            if abs(ta - T_ACQ) <= 1.5 and abs(tl - T_LOSE) <= 1.5:
                sols.append(Pm)
    info = dict(n_feasible_grid=len(feas), n_solutions=len(sols),
                solutions=[(p.kappa_c, p.kappa_m) for p in sols])
    return (sols[0] if sols else None), info


def calibrate(s0: float, gamma0: float, beta: float) -> Params:
    P, info = calibrate_info(s0, gamma0, beta)
    if P is None:
        raise RuntimeError(f"no calibration for s0={s0}, gamma0={gamma0}, beta={beta}: {info}")
    return P


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s0", type=float, default=0.1)
    ap.add_argument("--gamma0", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    P = calibrate(a.s0, a.gamma0, a.beta)
    print(P, crossing_times(P))
