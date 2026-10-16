"""Registered G0b analysis: matched-LM frontier and the inverse-sigma write rule (mini-m).

Inputs: the G0 root and the G0b root, each ``<root>/<preset>/<arm>/s<seed>/g0_read_<name>.npz``.
Every rule was fixed before the data existed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .g0_analysis import FO, LM_GUARD, MIN_GAP, SO, _iv

FO_ETAS = ("ce", "eta125", "eta150", "eta200", "eta250", "eta300", "eta400")
SO_ETAS = ("eta050", "eta075", "ce", "eta200")
BAND = .25            # |rho| band of the frontier labels
INV_MIN = .10         # minimum mean kappa of the inverse rule for SELECTIVE
CE_RTOL, CE_ATOL = 1e-5, 1e-6   # the G0b ce read must equal G0's within the V0 tolerances


def _read(root: Path, preset: str, arm: str, seed: int, name: str):
    p = root / preset / arm / f"s{seed}" / f"g0_read_{name}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {"M": float(np.mean(z["M_doc"])), "L": float(z["L_adapt"]), "eta": float(z["eta"])}


def _points(roots, preset, arm, seed, names):
    out = {}
    for name in names:
        for root in roots:
            r = _read(root, preset, arm, seed, name)
            if r is not None:
                out[name] = r
                break
    return out


def interp_at(points: list[dict], L_star: float) -> float:
    """M at L_star on the eta-ordered curve: the lowest-eta segment with L strictly increasing
    that brackets L_star; NaN without a bracket (no extrapolation)."""
    pts = sorted(points, key=lambda p: p["eta"])
    for a, b in zip(pts, pts[1:]):
        if b["L"] > a["L"] and a["L"] <= L_star <= b["L"]:
            w = (L_star - a["L"]) / (b["L"] - a["L"])
            return a["M"] + w * (b["M"] - a["M"])
    return math.nan


def frontier(g0: Path, g0b: Path, preset="mini-m", seeds=(0, 1, 2)) -> dict:
    per_seed, ce_ok = {}, True
    for s in seeds:
        fo = _points([g0b, g0], preset, FO, s, FO_ETAS)
        so = _points([g0b, g0], preset, SO, s, SO_ETAS)
        for arm, pts in ((FO, fo), (SO, so)):
            a, b = _read(g0, preset, arm, s, "ce"), _read(g0b, preset, arm, s, "ce")
            if a and b:
                ce_ok &= all(abs(a[k] - b[k]) <= CE_ATOL + CE_RTOL * abs(a[k]) for k in ("M", "L"))
        if "ce" not in fo or "ce" not in so:
            continue
        dM = so["ce"]["M"] - fo["ce"]["M"]
        m_fo = interp_at(list(fo.values()), so["ce"]["L"])
        m_so = interp_at(list(so.values()), fo["ce"]["L"])
        Ls = sorted(so.values(), key=lambda p: p["eta"])
        per_seed[s] = {
            "dM": dM, "L_FO_ce": fo["ce"]["L"], "L_SO_ce": so["ce"]["L"],
            "M_FO_at_L_SO": m_fo, "M_SO_at_L_FO": m_so,
            "rho_match": (so["ce"]["M"] - m_fo) / dM if dM > MIN_GAP else math.nan,
            "rho_match_rev": (m_so - fo["ce"]["M"]) / dM if dM > MIN_GAP else math.nan,
            "fo_curve": {k: v for k, v in sorted(fo.items(), key=lambda kv: kv[1]["eta"])},
            "so_curve": {k: v for k, v in sorted(so.items(), key=lambda kv: kv[1]["eta"])},
            "so_L_min_eta": Ls[int(np.argmin([p["L"] for p in Ls]))]["eta"],
        }
    rho = _iv([r["rho_match"] for r in per_seed.values()])
    rho_rev = _iv([r["rho_match_rev"] for r in per_seed.values()])
    if not ce_ok:
        label = "NE"
    elif rho["n"] < len(seeds):
        label = "NE"          # a seed without a bracket (or missing) is not extrapolated
    elif rho["lb"] > BAND:
        label = "FRONTIER-SO"
    elif rho["ub"] < -BAND:
        label = "FRONTIER-FO"
    elif rho["lb"] > -BAND and rho["ub"] < BAND:
        label = "FRONTIER-MATCH"
    else:
        label = "UNRESOLVED"
    return {"label": label, "rho_match": rho, "rho_match_rev": rho_rev, "ce_identical": ce_ok,
            "per_seed": per_seed}


def inverse_sigma(g0: Path, g0b: Path, preset="mini-m", seeds=(0, 1, 2)) -> dict:
    per_seed = {}
    for s in seeds:
        fo = _points([g0b, g0], preset, FO, s, ("ce", "isig4", "isig4shuf", "iconst4", "sig4"))
        so = _points([g0b, g0], preset, SO, s, ("ce", "isig4", "isig4shuf", "sig4"))
        if "ce" not in fo or "ce" not in so:
            continue
        dM = so["ce"]["M"] - fo["ce"]["M"]
        k = {n: (r["M"] - fo["ce"]["M"]) / dM if dM > MIN_GAP else math.nan for n, r in fo.items()}
        k_so = {n: (r["M"] - so["ce"]["M"]) / dM if dM > MIN_GAP else math.nan for n, r in so.items()}
        dL = {n: r["L"] - fo["ce"]["L"] for n, r in fo.items()}
        per_seed[s] = {"kappa": k, "kappa_SO_own": k_so, "dL": dL}
    get = lambda r, n: r["kappa"].get(n, math.nan)  # noqa: E731
    k_inv = _iv([get(r, "isig4") for r in per_seed.values()])
    d_shuf = _iv([get(r, "isig4") - get(r, "isig4shuf") for r in per_seed.values()])
    d_const = _iv([get(r, "isig4") - get(r, "iconst4") for r in per_seed.values()])
    dL_inv = _iv([r["dL"].get("isig4", math.nan) for r in per_seed.values()])
    if d_shuf["n"] < len(seeds) or d_const["n"] < len(seeds):
        label = "NE"
    elif d_shuf["lb"] > 0 and d_const["lb"] > 0 and k_inv["mean"] >= INV_MIN:
        label = "SIGMA-INV-SELECTIVE" + ("+LM-COST" if dL_inv["mean"] > LM_GUARD else "")
    elif d_shuf["ub"] < 0 and d_const["ub"] < 0:
        label = "SIGMA-INV-ADVERSE"
    elif d_shuf["ub"] < .05 and d_const["ub"] < .05:
        label = "SIGMA-INV-NULL"
    else:
        label = "UNRESOLVED"
    sym = _iv([get(r, "isig4") + get(r, "sig4") for r in per_seed.values()])
    so_inv = _iv([r["kappa_SO_own"].get("isig4", math.nan) - r["kappa_SO_own"].get("isig4shuf", math.nan)
                  for r in per_seed.values()])
    return {"label": label, "kappa_isig4": k_inv, "d_shuf": d_shuf, "d_const": d_const,
            "dL_isig4": dL_inv, "kappa_iconst4": _iv([get(r, "iconst4") for r in per_seed.values()]),
            "kappa_isig4shuf": _iv([get(r, "isig4shuf") for r in per_seed.values()]),
            "symmetry_isig_plus_sig": sym, "SO_isig_minus_shuf_own": so_inv, "per_seed": per_seed}


def analyse(g0, g0b, preset="mini-m") -> dict:
    g0, g0b = Path(g0), Path(g0b)
    return {"preset": preset, "frontier": frontier(g0, g0b, preset),
            "inverse_sigma": inverse_sigma(g0, g0b, preset)}


def _f(x, nd=3):
    return "NE" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def markdown(res: dict) -> str:
    fr, inv = res["frontier"], res["inverse_sigma"]
    iv = lambda v: f"{_f(v['mean'])} [{_f(v['lb'])}, {_f(v['ub'])}]"  # noqa: E731
    lines = [f"# G0b analysis ({res['preset']})", "",
             f"- Frontier: **{fr['label']}**; rho_match (SO advantage kept at SO's L) {iv(fr['rho_match'])}; "
             f"reverse (SO at FO's L) {iv(fr['rho_match_rev'])}; ce reads identical to G0: {fr['ce_identical']}",
             f"- Inverse sigma: **{inv['label']}**; kappa(isig4) {iv(inv['kappa_isig4'])}; "
             f"isig4 - shuf {iv(inv['d_shuf'])}; isig4 - const {iv(inv['d_const'])}; dL {iv(inv['dL_isig4'])}",
             f"- Symmetry kappa(isig4) + kappa(sig4): {iv(inv['symmetry_isig_plus_sig'])}; "
             f"SO own isig4 - shuf: {iv(inv['SO_isig_minus_shuf_own'])}", "",
             "| seed | arm | read | eta | L_adapt | M |", "|---|---|---|---|---|---|"]
    for s, r in fr["per_seed"].items():
        for arm, curve in (("FO", r["fo_curve"]), ("SO", r["so_curve"])):
            for n, p in curve.items():
                lines.append(f"| {s} | {arm} | {n} | {p['eta']:g} | {p['L']:.4f} | {p['M']:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("g0")
    ap.add_argument("g0b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analyse(a.g0, a.g0b)
    out = Path(a.out)
    (out / "g0b_analysis.json").write_text(json.dumps(res, indent=2, allow_nan=True, default=str) + "\n")
    (out / "g0b_analysis.md").write_text(markdown(res))
    print(json.dumps({"frontier": res["frontier"]["label"], "inverse_sigma": res["inverse_sigma"]["label"]}))


if __name__ == "__main__":
    main()
