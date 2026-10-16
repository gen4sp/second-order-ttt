"""Registered G0c analysis: does the inverse-sigma rule lift the M-L_adapt frontier beyond eta (mini-m)?

Inputs: the G0, G0b and G0c roots, each ``<root>/<preset>/<arm>/s<seed>/g0_read_<name>.npz``.
Every rule was fixed before the data existed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .g0_analysis import FO, MIN_GAP, SO, _iv
from .g0b_analysis import CE_ATOL, CE_RTOL, _points, _read, interp_at

FO_ETA = ("ce", "eta125", "eta150", "eta200", "eta250", "eta300", "eta400")
SO_ETA = ("ce", "eta150", "eta200")                      # the eta >= 1 (Pareto) branch of SO's curve
FO_LIFT = ("isig4_eta150", "isig4_eta250", "isig4_eta300")   # primary lift points
FO_ISIG = ("isig4", "isig4_eta150", "isig4_eta250", "isig4_eta300", "isig4_eta400")
SO_ISIG = ("isig4", "isig4_eta150", "isig4_eta200")
LIFT_LB, LIFT_MIN = .05, .10   # ISIG-LIFT: lb > .05 and mean >= .10 (units of Delta M)


def quad_at(points: list[dict], L_star: float) -> float:
    """Robustness interpolant, parametrised by eta: quadratics L(eta) and M(eta) through the
    bracketing segment of ``interp_at`` and the neighbour nearer to its eta midpoint; solve
    L(eta) = L_star inside the segment. NaN without a bracket, a third point, or a monotone L(eta)."""
    pts = sorted(points, key=lambda p: p["eta"])
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        if b["L"] > a["L"] and a["L"] <= L_star <= b["L"]:
            cand = [j for j in (i - 1, i + 2) if 0 <= j < len(pts)]
            if not cand:
                return math.nan
            mid = (a["eta"] + b["eta"]) / 2
            j = min(cand, key=lambda k: abs(pts[k]["eta"] - mid))
            e = [p["eta"] for p in sorted([a, b, pts[j]], key=lambda p: p["eta"])]
            tri = sorted([a, b, pts[j]], key=lambda p: p["eta"])
            cL = np.polyfit(e, [p["L"] for p in tri], 2)
            cM = np.polyfit(e, [p["M"] for p in tri], 2)
            grid = np.linspace(a["eta"], b["eta"], 20001)
            Lg = np.polyval(cL, grid)
            if not np.all(np.diff(Lg) > 0):
                return math.nan
            return float(np.polyval(cM, np.interp(L_star, Lg, grid)))
    return math.nan


def _lift(read: dict, curve: list[dict], dM: float, f) -> float:
    m_hat = f(curve, read["L"])
    return (read["M"] - m_hat) / dM if dM > MIN_GAP and np.isfinite(m_hat) else math.nan


def _label(lin: dict, quad: dict, n_seeds: int) -> str:
    def one(v):
        if v["n"] < n_seeds:
            return "NE"
        if v["lb"] > LIFT_LB and v["mean"] >= LIFT_MIN:
            return "ISIG-LIFT"
        if v["ub"] < -LIFT_LB:
            return "ISIG-BELOW"
        if v["lb"] > -LIFT_LB and v["ub"] < LIFT_LB:
            return "ISIG-NO-LIFT"
        return "UNRESOLVED"
    a, b = one(lin), one(quad)
    if "NE" in (a, b):
        return "NE" if a == "NE" else "UNRESOLVED"     # quadratic NE alone cannot confirm the linear label
    return a if a == b else "UNRESOLVED"


def analyse(g0, g0b, g0c, preset="mini-m", seeds=(0, 1, 2)) -> dict:
    roots = [Path(g0c), Path(g0b), Path(g0)]
    per_seed, ce_ok = {}, True
    for s in seeds:
        fo = _points(roots, preset, FO, s, FO_ETA + FO_ISIG)
        so = _points(roots, preset, SO, s, SO_ETA + SO_ISIG)
        for arm in (FO, SO):
            a, c = _read(Path(g0), preset, arm, s, "ce"), _read(Path(g0c), preset, arm, s, "ce")
            if a and c:
                ce_ok &= all(abs(a[k] - c[k]) <= CE_ATOL + CE_RTOL * abs(a[k]) for k in ("M", "L"))
            else:
                ce_ok = False
        if "ce" not in fo or "ce" not in so:
            continue
        dM = so["ce"]["M"] - fo["ce"]["M"]
        fo_eta = [fo[n] for n in FO_ETA if n in fo]
        so_eta = [so[n] for n in SO_ETA if n in so]
        fo_isig = [fo[n] for n in FO_ISIG if n in fo]
        so_isig = [so[n] for n in SO_ISIG if n in so]
        lin = {n: _lift(fo[n], fo_eta, dM, interp_at) if n in fo else math.nan for n in FO_ISIG}
        quad = {n: _lift(fo[n], fo_eta, dM, quad_at) if n in fo else math.nan for n in FO_ISIG}
        lin_so = {n: _lift(so[n], so_eta, dM, interp_at) if n in so else math.nan for n in SO_ISIG}
        m_fo_isig_at_so = interp_at(fo_isig, so["ce"]["L"])
        m_fo_isig_at_soisig = interp_at(fo_isig, so["isig4"]["L"]) if "isig4" in so else math.nan
        per_seed[s] = {
            "dM": dM, "lift_lin": lin, "lift_quad": quad, "lift_SO_lin": lin_so,
            "lambda_lin": float(np.mean([lin[n] for n in FO_LIFT])),
            "lambda_quad": float(np.mean([quad[n] for n in FO_LIFT])),
            "lambda_SO_lin": float(np.mean([lin_so[n] for n in ("isig4_eta150", "isig4_eta200")])),
            # SO's advantage at SO's L when FO is moved there with isig4 x eta (G0b's rho_match analogue)
            "rho_match_isig": (so["ce"]["M"] - m_fo_isig_at_so) / dM if dM > MIN_GAP else math.nan,
            # both arms with isig4: SO isig4 at eta x1 vs FO's isig4 curve at that L
            "rho_match_both_isig": ((so["isig4"]["M"] - m_fo_isig_at_soisig) / dM
                                    if dM > MIN_GAP and "isig4" in so else math.nan),
            "points": {"FO": fo, "SO": so},
        }
    lam = _iv([r["lambda_lin"] for r in per_seed.values()])
    lam_q = _iv([r["lambda_quad"] for r in per_seed.values()])
    label = _label(lam, lam_q, len(seeds)) if ce_ok else "NE"
    by_point = {n: _iv([r["lift_lin"][n] for r in per_seed.values()]) for n in FO_ISIG}
    return {"preset": preset, "label": label, "ce_identical": ce_ok,
            "lambda_lin": lam, "lambda_quad": lam_q, "lift_by_point": by_point,
            "lambda_SO_lin": _iv([r["lambda_SO_lin"] for r in per_seed.values()]),
            "rho_match_isig": _iv([r["rho_match_isig"] for r in per_seed.values()]),
            "rho_match_both_isig": _iv([r["rho_match_both_isig"] for r in per_seed.values()]),
            "per_seed": per_seed}


def _f(x, nd=3):
    return "NE" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def markdown(res: dict) -> str:
    iv = lambda v: f"{_f(v['mean'])} [{_f(v['lb'])}, {_f(v['ub'])}]"  # noqa: E731
    lines = [f"# G0c analysis ({res['preset']})", "",
             f"- Frontier lift of inverse sigma (FO): **{res['label']}**; lambda (linear) {iv(res['lambda_lin'])}; "
             f"lambda (quadratic) {iv(res['lambda_quad'])}; ce reads identical to G0: {res['ce_identical']}",
             "- Lift by point (linear): " + "; ".join(f"{n} {iv(v)}" for n, v in res["lift_by_point"].items()),
             f"- SO lift (linear, isig4 x eta 1.5/2): {iv(res['lambda_SO_lin'])}",
             f"- rho_match with FO on its isig4 curve: {iv(res['rho_match_isig'])}; "
             f"both arms with isig4: {iv(res['rho_match_both_isig'])}", "",
             "| seed | arm | read | eta | L_adapt | M |", "|---|---|---|---|---|---|"]
    for s, r in res["per_seed"].items():
        for arm in ("FO", "SO"):
            for n, p in sorted(r["points"][arm].items(), key=lambda kv: (kv[0].startswith("isig"), kv[1]["eta"])):
                lines.append(f"| {s} | {arm} | {n} | {p['eta']:g} | {p['L']:.4f} | {p['M']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("g0")
    ap.add_argument("g0b")
    ap.add_argument("g0c")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analyse(a.g0, a.g0b, a.g0c)
    out = Path(a.out)
    (out / "g0c_analysis.json").write_text(json.dumps(res, indent=2, allow_nan=True, default=str) + "\n")
    (out / "g0c_analysis.md").write_text(markdown(res))
    print(json.dumps({"label": res["label"], "lambda_lin": res["lambda_lin"]["mean"],
                      "lambda_quad": res["lambda_quad"]["mean"]}))


if __name__ == "__main__":
    main()
