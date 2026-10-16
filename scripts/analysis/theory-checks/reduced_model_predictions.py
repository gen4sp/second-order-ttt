"""Reduced model: calibration, reproduction table, sensitivity, phase diagram, predictions.

Everything is simulated from reduced_model.py. The only measured inputs are the two declared
calibration targets (anchor first-bound and first-loss probe medians of the FO-CE fixed-eta anchor
runs), which live in reduced_model.T_ACQ / T_LOSE. MEASURED_* constants below are used ONLY to print
the comparison column of the reproduction table; they never enter a simulation.

Writes to ``--out-dir``: repro-arms.csv, discretisation-check.csv, predictions-base.csv,
sensitivity.csv, predictions-sensitivity-summary.csv, phase.npz, phase-clean-anchor.csv,
recovery-collapse.csv, trajectories.json, analysis-metadata.json.

Run:  uv run --locked python scripts/analysis/theory-checks/reduced_model_predictions.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reduced_model import (ANCHOR, MU_STAR, Arm, Params, calibrate, crossing_times, events,  # noqa: E402
                     simulate)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out-dir", default="outputs/analysis/theory-checks/reduced-model")
ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "reduced-model-predictions"))
args = ap.parse_args()
DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)

BASE_CONV = dict(s0=0.1, gamma0=0.3, beta=1.0)   # declared before any arm was simulated
SENS_GRID = dict(s0=[0.03, 0.1, 0.3], gamma0=[0.1, 0.3, 0.6], beta=[0.5, 1.0, 2.0])

# ---------------------------------------------------------------- arms (model regime)
REPRO_ARMS = {
    "anchor (FO, CE)": ANCHOR,
    "SO (e2e2emb)": Arm("SO", order="SO"),
    "kvm (outer kv mask)": Arm("kvm", kv_outer=False),
    "edge (copy edge masked)": Arm("edge", copy_edge=False),
    "linkv": Arm("linkv", rule="lin"),
    "linvW (CE at kv)": Arm("linvW", rule="CE"),
    "aice (== lin at chunk 0)": Arm("aice", rule="aice"),
    "kveta (kv_eta_lr .01)": Arm("kveta", kv_eta_lr=0.01),
    "kveta10 (kv_eta_lr .1)": Arm("kveta10", kv_eta_lr=0.1),
    "cn99": Arm("cn99", q=0.99),
    "cn97": Arm("cn97", q=0.97),
    "cn95": Arm("cn95", q=0.95),
    "cn90": Arm("cn90", q=0.90),
}
# comparison column only (measured toy results)
MEASURED = {
    "anchor (FO, CE)": "acq 10/10 @212.5, lost 10/10 @275, no rebind",
    "SO (e2e2emb)": "retained 10/10; transient dip 9/10, rebind 175-275 (learned eta)",
    "kvm (outer kv mask)": "retained 10/10, first bound 400 (325-475)",
    "edge (copy edge masked)": "retained 10/10, first bound 225 (200-250)",
    "linkv": "retained 10/10, first bound 250",
    "linvW (CE at kv)": "retained 0/10 (lost @275)",
    "aice (== lin at chunk 0)": "retained 10/10, first bound 400-725",
    "kveta (kv_eta_lr .01)": "retained 8/10; lost 10/10 @275-325, rebind 300-675; eta_kv 2243",
    "kveta10 (kv_eta_lr .1)": "retained 10/10; lost 10/10 @250-450, rebind 375-600; eta_kv 7586",
    "cn99": "retained 0/6 (final acc .74-.87)",
    "cn97": "retained 6/6 (lost @225-300, rebind 425-575)",
    "cn95": "retained 6/6 (lost @250-300, rebind 300-400)",
    "cn90": "retained 6/6 (2/6 dip, rebind 300)",
}


def ev_row(name, arm, P):
    tr = simulate(arm, P)
    ev = events(tr)
    g, x = ev["grid"], ev["exact"]
    return dict(arm=name, first_bound_grid=g["first_bound"], first_loss_grid=g["first_loss"],
                rebind_grid=g["rebind"], final_bound=g["final_bound"],
                first_bound_exact=x["first_bound"], first_loss_exact=x["first_loss"],
                rebind_exact=x["rebind"], dip_min_acc=g.get("dip_min_acc"),
                final_acc=ev["final_acc"], final_p_kv=ev["final_p_kv"],
                final_d_clean=ev["final_d_clean"], final_eta_hat=ev["final_eta_eff"],
                final_mu_clean=ev["final_mu_clean"])


def repro_flags(P):
    """F1..F7 reproduction criteria."""
    r = {k: ev_row(k, a, P) for k, a in REPRO_ARMS.items()}
    a = r["anchor (FO, CE)"]
    f = {}
    f["F1"] = bool(a["first_bound_exact"] is not None and a["first_loss_exact"] is not None
                   and a["rebind_exact"] is None and not a["final_bound"])
    f["F2"] = bool(r["SO (e2e2emb)"]["final_bound"])
    f["F3_rescue"] = bool(r["kvm (outer kv mask)"]["final_bound"])
    kfb, afb = r["kvm (outer kv mask)"]["first_bound_exact"], a["first_bound_exact"]
    f["F3_later"] = bool(kfb is not None and afb is not None and kfb > afb)
    f["F4"] = bool(r["edge (copy edge masked)"]["final_bound"])
    f["F5"] = bool(r["linkv"]["final_bound"] and not r["linvW (CE at kv)"]["final_bound"])
    kv = r["kveta (kv_eta_lr .01)"]
    f["F6"] = bool(kv["final_bound"] and kv["final_p_kv"] > 0.99 and kv["final_eta_hat"] > 10)
    f["F6_dip_rebind"] = bool(kv["first_loss_exact"] is not None and kv["rebind_exact"] is not None)
    f["F7_aice"] = bool(r["aice (== lin at chunk 0)"]["final_bound"])
    # measured pattern at learned eta: cn99 lost, cn97/95/90 retained
    pat = (not r["cn99"]["final_bound"]) and all(r[k]["final_bound"] for k in ("cn97", "cn95", "cn90"))
    f["F7_threshold_at_eta1"] = bool(pat)
    return f, r


def threshold_eta_window(P, etas=None):
    """Fixed eta_hat values at which the measured cn pattern (99 lost; 97/95/90 kept) holds."""
    etas = etas if etas is not None else np.round(np.geomspace(1, 32, 21), 3)
    ok = []
    for eh in etas:
        res = {q: events(simulate(Arm("cn", q=q, eta_hat=float(eh)), P))["grid"]["final_bound"]
               for q in (0.99, 0.97, 0.95, 0.90)}
        if (not res[0.99]) and res[0.97] and res[0.95] and res[0.90]:
            ok.append(float(eh))
    return (min(ok), max(ok)) if ok else (None, None)


# ---------------------------------------------------------------- predictions (model side)
PRED_ARMS = {
    # P1: fixed eta = 10 (eta_lr = 0), copy-noise arms
    "P1 cn99 eta10": Arm("x", q=0.99), "P1 cn97 eta10": Arm("x", q=0.97),
    "P1 cn95 eta10": Arm("x", q=0.95), "P1 cn90 eta10": Arm("x", q=0.90),
    "P1 linkv eta10": Arm("x", rule="lin"), "P1 anchor eta10": Arm("x"),
    # P2: fixed eta = 40
    "P2 cn99 eta40": Arm("x", q=0.99, eta_hat=4), "P2 cn97 eta40": Arm("x", q=0.97, eta_hat=4),
    "P2 cn95 eta40": Arm("x", q=0.95, eta_hat=4), "P2 cn90 eta40": Arm("x", q=0.90, eta_hat=4),
    "P2 anchor eta40": Arm("x", eta_hat=4), "P2 linkv eta40": Arm("x", rule="lin", eta_hat=4),
    # P3: clean anchor timing vs fixed eta
    "P3 anchor eta10": Arm("x", eta_hat=1), "P3 anchor eta20": Arm("x", eta_hat=2),
    "P3 anchor eta40": Arm("x", eta_hat=4),
    # P4: no hysteresis: noise switched on after starvation
    "P4 cn95 eta40 onset0": Arm("x", q=0.95, eta_hat=4),
    "P4 cn95 eta40 onset600": Arm("x", q=0.95, eta_hat=4, noise_onset=600),
    # P5: kv inner loss switched CE -> lin after starvation
    "P5 linkv-from-600 eta10": Arm("x", lin_from=600),
    "P4 cn90 eta40 onset0": Arm("x", q=0.90, eta_hat=4),
    "P4 cn90 eta40 onset600": Arm("x", q=0.90, eta_hat=4, noise_onset=600),
    # P6: SO at fixed eta
    "P6 SO eta10": Arm("x", order="SO", eta_hat=1), "P6 SO eta40": Arm("x", order="SO", eta_hat=4),
    "P6 FO anchor eta10": Arm("x"),
    # P7: recovery delay collapses on eta * (1 - q)
    "P7 cn90 eta20": Arm("x", q=0.90, eta_hat=2), "P7 cn95 eta40": Arm("x", q=0.95, eta_hat=4),
    "P7 cn95 eta20": Arm("x", q=0.95, eta_hat=2), "P7 cn97 eta40": Arm("x", q=0.97, eta_hat=4),
    # P8: race threshold: below eta ~ 9 the copy forms before the memory binds
    "P8 anchor eta5": Arm("x", eta_hat=0.5), "P8 edge eta5": Arm("x", copy_edge=False, eta_hat=0.5),
    "P8 linkv eta5": Arm("x", rule="lin", eta_hat=0.5), "P8 anchor eta7": Arm("x", eta_hat=0.7),
}


def pred_rows(P):
    return {k: ev_row(k, a, P) for k, a in PRED_ARMS.items()}


# ---------------------------------------------------------------- phase diagram
def phase(P, n_eta=29, n_r=29):
    etas = np.geomspace(0.5, 32, n_eta)
    rs = np.geomspace(1e-3, 0.3, n_r)
    acc = np.zeros((n_r, n_eta))
    cls = np.zeros((n_r, n_eta), int)   # 0 never bound, 1 lost (permanent), 2 lost+rebound, 3 never lost
    for i, r in enumerate(rs):
        for j, eh in enumerate(etas):
            tr = simulate(Arm("ph", q=float(1 - r), eta_hat=float(eh)), P)
            ev = events(tr)["grid"]
            acc[i, j] = tr.acc[-1]
            if not ev["ever_bound"]:
                cls[i, j] = 0
            elif ev["first_loss"] is None:
                cls[i, j] = 3
            elif ev["final_bound"]:
                cls[i, j] = 2
            else:
                cls[i, j] = 1
    clean = []
    for eh in etas:
        ev = events(simulate(Arm("ph", eta_hat=float(eh)), P))
        clean.append((float(eh), ev["exact"]["first_bound"], ev["exact"]["first_loss"],
                      ev["grid"]["final_bound"], ev["final_acc"]))
    return etas, rs, acc, cls, clean


def recovery_collapse(P):
    """Recovery delay (rebind - first loss) against eta_hat * d*, d* = 16/15 (1 - q)."""
    rows = []
    for eh in (1, 2, 4, 8, 16):
        for q in (0.995, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7):
            ev = events(simulate(Arm("rc", q=q, eta_hat=eh), P))["exact"]
            if ev["first_loss"] is not None and ev["rebind"] is not None:
                rows.append(dict(eta_hat=eh, q=q, x=eh * 16 / 15 * (1 - q),
                                 delay=ev["rebind"] - ev["first_loss"]))
    return rows


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    meta = dict(reduced_model_sha256=sha(HERE / "reduced_model.py"), predictions_script_sha256=sha(HERE / "reduced_model_predictions.py"),
                mu_star=MU_STAR, base_conventions=BASE_CONV)
    # 1. calibration
    P = calibrate(**BASE_CONV)
    ta, tl = crossing_times(P)
    meta["calibrated"] = dict(kappa_c=P.kappa_c, kappa_m=P.kappa_m, t_acq_exact=ta, t_lose_exact=tl)
    print("calibrated", P, ta, tl)

    # 2. reproduction
    flags, rows = repro_flags(P)
    for k in rows:
        rows[k]["measured"] = MEASURED[k]
    write_csv(DATA / "repro-arms.csv", rows.values())
    meta["repro_flags"] = flags
    meta["cn_pattern_eta_window"] = threshold_eta_window(P)
    print("flags", flags, "cn window", meta["cn_pattern_eta_window"])

    # 3. discretisation check
    disc = []
    for k, a in REPRO_ARMS.items():
        e1 = events(simulate(a, P))["grid"]
        e4 = events(simulate(a, P, substeps=4))["grid"]
        disc.append(dict(arm=k, fb1=e1["first_bound"], fb4=e4["first_bound"], fl1=e1["first_loss"],
                         fl4=e4["first_loss"], rb1=e1["rebind"], rb4=e4["rebind"],
                         final1=e1["final_bound"], final4=e4["final_bound"]))
    write_csv(DATA / "discretisation-check.csv", disc)

    # 4. predictions at the base conventions
    pr = pred_rows(P)
    write_csv(DATA / "predictions-base.csv", pr.values())

    # 5. sensitivity over the three conventions (each recalibrated on the anchor only)
    sens = []
    for s0, g0, beta in itertools.product(*SENS_GRID.values()):
        Pk = calibrate(s0, g0, beta)
        ta_k, tl_k = crossing_times(Pk)
        fk, _ = repro_flags(Pk)
        pk = pred_rows(Pk)
        calib_ok = abs(ta_k - 200.0) <= 2 and abs(tl_k - 262.5) <= 2
        row = dict(s0=s0, gamma0=g0, beta=beta, kappa_c=Pk.kappa_c, kappa_m=Pk.kappa_m,
                   t_acq=ta_k, t_lose=tl_k, calib_ok=calib_ok, **fk)
        lo, hi = threshold_eta_window(Pk, etas=np.round(np.geomspace(1, 32, 11), 3))
        row["cn_window_lo"], row["cn_window_hi"] = lo, hi
        for name, v in pk.items():
            key = name.split(" ", 1)[0] + "|" + name.split(" ", 1)[1]
            row[key + "|final"] = v["final_bound"]
            row[key + "|fb"] = v["first_bound_grid"]
            row[key + "|fl"] = v["first_loss_grid"]
            row[key + "|rb"] = v["rebind_grid"]
            row[key + "|pkv"] = round(v["final_p_kv"], 4)
            row[key + "|acc"] = round(v["final_acc"], 4)
        sens.append(row)
        print("sens", s0, g0, beta, "ok" if all(fk[x] for x in ("F1", "F2", "F3_rescue", "F4", "F5", "F6", "F7_aice")) else fk)
    write_csv(DATA / "sensitivity.csv", sens)
    summ = []
    for name in PRED_ARMS:
        key = name.split(" ", 1)[0] + "|" + name.split(" ", 1)[1]
        def col(f):
            return [r[key + "|" + f] for r in sens]
        num = lambda xs: [x for x in xs if x is not None]
        fb, fl, rb, pkv, acc = (num(col(f)) for f in ("fb", "fl", "rb", "pkv", "acc"))
        summ.append(dict(prediction_arm=name, n_conventions=len(sens),
                         frac_retained=float(np.mean(col("final"))),
                         first_bound_min=min(fb) if fb else None, first_bound_max=max(fb) if fb else None,
                         n_lost=len(fl), first_loss_min=min(fl) if fl else None,
                         first_loss_max=max(fl) if fl else None,
                         n_rebind=len(rb), rebind_min=min(rb) if rb else None, rebind_max=max(rb) if rb else None,
                         p_kv_min=min(pkv), p_kv_max=max(pkv), acc_min=min(acc), acc_max=max(acc)))
    write_csv(DATA / "predictions-sensitivity-summary.csv", summ)
    flag_names = [k for k in sens[0] if k.startswith("F")]
    meta["sensitivity_flag_rates"] = {k: float(np.mean([r[k] for r in sens])) for k in flag_names}
    meta["sensitivity_calib_ok"] = float(np.mean([r["calib_ok"] for r in sens]))

    # 6. phase diagram + collapse
    etas, rs, acc, cls, clean = phase(P)
    np.savez(DATA / "phase.npz", etas=etas, rs=rs, acc=acc, cls=cls)
    write_csv(DATA / "phase-clean-anchor.csv",
              [dict(eta_hat=a, t_acq_exact=b, t_lose_exact=c, final_bound=d, final_acc=e)
               for a, b, c, d, e in clean])
    rc = recovery_collapse(P)
    write_csv(DATA / "recovery-collapse.csv", rc)
    xs = np.log([r["x"] for r in rc])
    ys = np.log([r["delay"] for r in rc])
    slope, icpt = np.polyfit(xs, ys, 1)
    meta["recovery_collapse_fit"] = dict(slope=float(slope), intercept=float(icpt), n=len(rc),
                                         resid_sd=float(np.std(ys - (slope * xs + icpt))))
    # trajectories for the figure
    traj = {}
    for k, a in {"FO, CE (anchor)": ANCHOR, "SO, CE": Arm("SO", order="SO"),
                 "FO, lin at kv": Arm("l", rule="lin"), "FO, CE, kv-only learned $\\eta$": Arm("k", kv_eta_lr=0.01),
                 "FO, CE, $q=.95$, $\\eta=40$": Arm("c", q=0.95, eta_hat=4)}.items():
        tr = simulate(a, P)
        traj[k] = dict(step=tr.step[::5].tolist(), acc=tr.acc[::5].tolist(), p_kv=tr.p_kv[::5].tolist())
    with open(DATA / "trajectories.json", "w") as fh:
        json.dump(traj, fh)
    with open(DATA / "analysis-metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    try:
        import setproctitle

        setproctitle.setproctitle(args.process_name)
    except Exception:
        pass
    main()
