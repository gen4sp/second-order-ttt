"""Registered G0 analysis: write-gain split and eval-time gain titration (mini-m, 3 seeds).

Layout: ``root/<preset>/<arm>/s<seed>/{g0_read_<name>.npz, gain_split.npz, g0_calib.json}``.
Every rule below was fixed before the data existed; this module only implements it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

T_CRIT = 1.886            # one-sided 90 %, n = 3 seeds
SO, FO = "e2e2", "e2e1-ce"
ETA_READS = ("eta125", "eta150", "eta200")
LM_GUARD = 0.01           # nats/token: max mean L_adapt increase over the CE read
MIN_GAP = 0.02            # per-seed Delta M below this leaves kappa undefined
MIN_LOG_GAP = 0.05        # |SO-FO mean log||g||| below this: no per-token gain gap
FACTORS = ("e", "u", "gam", "proj", "s", "R")
GROUPS = {"ERROR": ("e",), "HEAD": ("u", "gam", "proj"), "SIGMA": ("s",), "BODY": ("R",)}
SCM_B1 = {"beta_sigma": -1.143, "beta_sigma_ci": [-1.263, -1.006], "beta_H": 0.286}


def _iv(values, t=T_CRIT):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": math.nan, "sd": math.nan, "lb": math.nan, "ub": math.nan,
                "values": []}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else math.nan
    half = t * sd / math.sqrt(len(x)) if math.isfinite(sd) else math.nan
    return {"n": int(len(x)), "mean": mean, "sd": sd,
            "lb": mean - half if math.isfinite(half) else math.nan,
            "ub": mean + half if math.isfinite(half) else math.nan, "values": x.tolist()}


def _reads(cell: Path) -> dict:
    out = {}
    for p in sorted(cell.glob("g0_read_*.npz")):
        name = p.stem.removeprefix("g0_read_")
        with np.load(p, allow_pickle=False) as z:
            out[name] = {"M": float(np.mean(z["M_doc"])), "M_doc": np.asarray(z["M_doc"]),
                         "L": float(z["L_adapt"]) if "L_adapt" in z else float(np.mean(z["L_adapt_doc"])),
                         "stats": json.loads(str(z["stats"])),
                         "sha": str(z["ckpt_sha256"]),
                         "rows": np.asarray(z["eval_rows"])}
    return out


def collect(root: str | Path, preset: str = "mini-m") -> dict:
    root = Path(root)
    cells = {}
    for arm in (SO, FO):
        for d in sorted((root / preset / arm).glob("s[0-9]*")):
            seed = int(d.name[1:])
            rec = {"reads": _reads(d)}
            sp = d / "gain_split.npz"
            if sp.exists():
                with np.load(sp, allow_pickle=False) as z:
                    rec["split"] = {k: np.asarray(z[k]) for k in z.files}
            cp = d / "g0_calib.json"
            if cp.exists():
                rec["calib"] = json.loads(cp.read_text())
            cells[(arm, seed)] = rec
    return cells


def titration(cells: dict) -> dict:
    seeds = sorted({s for (a, s) in cells if a == FO} & {s for (a, s) in cells if a == SO})
    per_seed, rows_ok = {}, True
    for s in seeds:
        fo, so = cells[(FO, s)]["reads"], cells[(SO, s)]["reads"]
        if "ce" not in fo or "ce" not in so:
            continue
        rows_ok &= bool(np.array_equal(fo["ce"]["rows"], so["ce"]["rows"]))
        gap = so["ce"]["M"] - fo["ce"]["M"]
        rec = {"M_FO_ce": fo["ce"]["M"], "M_SO_ce": so["ce"]["M"], "dM": gap, "kappa": {},
               "dL": {}, "clip_bind": {}, "mult_mean": {}}
        for name, r in fo.items():
            rec["kappa"][name] = (r["M"] - fo["ce"]["M"]) / gap if gap > MIN_GAP else math.nan
            rec["dL"][name] = r["L"] - fo["ce"]["L"]
            st = r["stats"].get("fP_focal", {})
            rec["clip_bind"][name] = st.get("clip_bind_frac", math.nan)
            rec["mult_mean"][name] = st.get("mult_mean", math.nan)
        rec["SO"] = {name: {"kappa_own": (r["M"] - so["ce"]["M"]) / gap if gap > MIN_GAP
                            else math.nan, "dL": r["L"] - so["ce"]["L"]}
                     for name, r in so.items()}
        if "eta200" in so and "eta200" in fo:
            rec["matched_gap_eta200"] = so["eta200"]["M"] - fo["eta200"]["M"]
        per_seed[s] = rec
    names = sorted({n for r in per_seed.values() for n in r["kappa"]})
    kappa = {n: _iv([r["kappa"].get(n, math.nan) for r in per_seed.values()]) for n in names}
    dL = {n: _iv([r["dL"].get(n, math.nan) for r in per_seed.values()]) for n in names}
    dM = _iv([r["dM"] for r in per_seed.values()])

    # B1: the largest registered eta dose that passes the LM guard.
    passing = [n for n in ETA_READS if n in dL and dL[n]["mean"] <= LM_GUARD]
    dose = passing[-1] if passing else ("eta125" if "eta125" in kappa else None)
    if dose is None:
        b1 = "NE"
    else:
        k = kappa[dose]
        if k["n"] < 3:
            b1 = "NE"
        elif k["mean"] >= .5 and k["lb"] > .25:
            b1 = "GAIN-LIMITED"
        elif k["lb"] > 0:
            b1 = "PARTIAL-DOSE"
        else:
            b1 = "DOSE-INSENSITIVE"
        if b1 != "NE" and not passing:
            b1 += "+LM-COST"

    # B2: sigma selectivity against the shuffled and the constant-matched controls.
    b2, d_shuf, d_const = "NE", None, None
    if all(n in kappa for n in ("sig4", "sig4shuf", "const4")):
        k_ = lambda r, n: r["kappa"].get(n, math.nan)  # noqa: E731
        d_shuf = _iv([k_(r, "sig4") - k_(r, "sig4shuf") for r in per_seed.values()])
        d_const = _iv([k_(r, "sig4") - k_(r, "const4") for r in per_seed.values()])
        if d_shuf["n"] < 3:
            b2 = "NE"
        elif d_shuf["lb"] > 0 and d_const["lb"] > 0 and kappa["sig4"]["mean"] >= .25:
            b2 = "SIGMA-SELECTIVE"
        elif d_shuf["ub"] < 0 and d_const["ub"] < 0:
            b2 = "SIGMA-ADVERSE"
        elif d_shuf["ub"] < .10 and d_const["ub"] < .10:
            b2 = "SIGMA-NULL"
        else:
            b2 = "UNRESOLVED"
        if b2 == "SIGMA-SELECTIVE" and dL["sig4"]["mean"] > LM_GUARD:
            b2 += "+LM-COST"
    if not rows_ok:  # SO and FO must be read on the same eval rows
        b1, b2 = "NE", "NE"
    matched = [r.get("matched_gap_eta200", math.nan) for r in per_seed.values()]
    return {"seeds": seeds, "eval_rows_match": rows_ok, "dM": dM, "kappa": kappa, "dL": dL,
            "B1": {"label": b1, "dose": dose, "passing_doses": passing},
            "B2": {"label": b2, "d_shuf": d_shuf, "d_const": d_const},
            "matched_gap_eta200": _iv(matched),
            "matched_gap_ratio": _iv([m / r["dM"] if r["dM"] > MIN_GAP else math.nan
                                      for m, r in zip(matched, per_seed.values())]),
            "per_seed": per_seed}


def _logs(sp: dict, sel: np.ndarray) -> dict:
    out = {f: np.log(sp[f][sel].astype(float)) for f in FACTORS[:-1]}
    out["g"] = np.log(sp["gnorm"][sel].astype(float))
    out["R"] = out["g"] - sum(out[f] for f in FACTORS[:-1])
    return out


def _ols(y, *xs):
    X = np.column_stack([np.ones_like(y)] + list(xs))
    ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
    if ok.sum() <= X.shape[1]:
        return [math.nan] * X.shape[1]
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0].tolist()


def decomposition(cells: dict) -> dict:
    seeds = sorted({s for (a, s) in cells if a == FO and "split" in cells[(a, s)]} &
                   {s for (a, s) in cells if a == SO and "split" in cells[(a, s)]})
    per_seed, scm = {}, {SO: [], FO: []}
    for s in seeds:
        means = {}
        for arm in (SO, FO):
            sp = cells[(arm, s)]["split"]
            foc = sp["focal"].astype(bool)
            lf, ln = _logs(sp, foc), _logs(sp, ~foc)
            means[arm] = {k: float(np.mean(v)) for k, v in lf.items()}
            means[arm]["focal_minus_nonfocal_g"] = float(np.mean(lf["g"]) - np.mean(ln["g"]))
            blk = np.log(sp["gblk"][foc].astype(float))
            means[arm]["log_gblk"] = blk.mean(0).tolist()
            all_ = np.ones(len(sp["gnorm"]), dtype=bool)
            la = _logs(sp, all_)
            log_sig = -la["s"]
            H = sp["H"].astype(float)
            b_es = _ols(la["g"], la["e"], log_sig)
            b_s = _ols(la["g"], log_sig)
            b_H = _ols(log_sig, H)
            scm[arm].append({"seed": s, "beta_sigma_given_e": b_es[2], "beta_sigma": b_s[1],
                             "beta_H": b_H[1], "n_tok": int(len(H)),
                             "corr_logsigma_H": float(np.corrcoef(log_sig, H)[0, 1])})
        d = {k: means[SO][k] - means[FO][k] for k in FACTORS + ("g",)}
        per_seed[s] = {"SO": means[SO], "FO": means[FO], "delta": d,
                       "delta_blk": (np.asarray(means[SO]["log_gblk"]) -
                                     np.asarray(means[FO]["log_gblk"])).tolist()}
    delta = {k: _iv([r["delta"][k] for r in per_seed.values()]) for k in FACTORS + ("g",)}
    gap = delta["g"]["mean"]
    shares = {g: (sum(delta[f]["mean"] for f in fs) / gap if np.isfinite(gap) and gap != 0
                  else math.nan) for g, fs in GROUPS.items()}
    signs = {np.sign(r["delta"]["g"]) for r in per_seed.values()}
    if len(per_seed) < 3:
        label = "NE"
    elif abs(gap) < MIN_LOG_GAP or len(signs) > 1:
        label = "NO-GAP"
    else:
        top = max(shares, key=lambda g: shares[g])
        label = f"{top}-MAJOR" if shares[top] >= .5 else "MIXED"
    scm_iv = {arm: {k: _iv([r[k] for r in rows]) for k in
                    ("beta_sigma_given_e", "beta_sigma", "beta_H", "corr_logsigma_H")}
              for arm, rows in scm.items()}
    return {"seeds": seeds, "delta_log": delta, "shares": shares, "label": label,
            "per_seed": per_seed, "scm": scm_iv, "scm_per_seed": scm, "scm_b1_125m": SCM_B1}


def analyse(root: str | Path, preset: str = "mini-m") -> dict:
    cells = collect(root, preset)
    shas = {f"{a}/s{s}": next(iter(r["reads"].values()))["sha"] for (a, s), r in cells.items()
            if r["reads"]}
    calib = {f"{a}/s{s}": r.get("calib") for (a, s), r in cells.items()}
    return {"preset": preset, "cells": sorted(shas), "ckpt_sha256": shas, "calib": calib,
            "titration": titration(cells), "decomposition": decomposition(cells)}


def _f(x, nd=3):
    return "NE" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def markdown(res: dict) -> str:
    t, d = res["titration"], res["decomposition"]
    lines = [f"# G0 analysis ({res['preset']})", "",
             f"- B1 (eta dose): **{t['B1']['label']}** (dose {t['B1']['dose']}, "
             f"guard-passing {t['B1']['passing_doses']})",
             f"- B2 (sigma rule): **{t['B2']['label']}**",
             f"- Decomposition: **{d['label']}**", "",
             f"Delta M (SO-FO, CE): {_f(t['dM']['mean'])} [{_f(t['dM']['lb'])}, {_f(t['dM']['ub'])}]",
             "", "| read | kappa mean | lb | ub | dL_adapt | clip-bind (fP_focal) |",
             "|---|---|---|---|---|---|"]
    for n, k in t["kappa"].items():
        cb = np.nanmean([r["clip_bind"].get(n, np.nan) for r in t["per_seed"].values()])
        lines.append(f"| {n} | {_f(k['mean'])} | {_f(k['lb'])} | {_f(k['ub'])} | "
                     f"{_f(t['dL'][n]['mean'], 4)} | {cb:.3f} |")
    if t["B2"]["d_shuf"]:
        lines += ["", f"sig4 - sig4shuf: {_f(t['B2']['d_shuf']['mean'])} "
                      f"[{_f(t['B2']['d_shuf']['lb'])}, {_f(t['B2']['d_shuf']['ub'])}]; "
                      f"sig4 - const4: {_f(t['B2']['d_const']['mean'])} "
                      f"[{_f(t['B2']['d_const']['lb'])}, {_f(t['B2']['d_const']['ub'])}]"]
    lines += ["", f"Matched-dose gap (eta x2): {_f(t['matched_gap_eta200']['mean'])}; "
                  f"ratio to CE gap {_f(t['matched_gap_ratio']['mean'])}", "",
              "| factor | SO-FO mean log (focal) | lb | ub |", "|---|---|---|---|"]
    for k, v in d["delta_log"].items():
        lines.append(f"| {k} | {_f(v['mean'])} | {_f(v['lb'])} | {_f(v['ub'])} |")
    lines += ["", "Shares of the log-gap: " + ", ".join(f"{g} {_f(v, 2)}" for g, v in d["shares"].items()),
              "", "SCM B1 regressions (per arm, mean over seeds; 125M reference "
                  f"beta_sigma {SCM_B1['beta_sigma']}, beta_H {SCM_B1['beta_H']}):"]
    for arm, v in d["scm"].items():
        lines.append(f"- {arm}: beta_sigma|e {_f(v['beta_sigma_given_e']['mean'])}, "
                     f"beta_sigma {_f(v['beta_sigma']['mean'])}, beta_H {_f(v['beta_H']['mean'])}")
    return "\n".join(lines) + "\n"


def write(root, out_json, out_md, preset="mini-m") -> dict:
    res = analyse(root, preset)
    Path(out_json).write_text(json.dumps(res, indent=2, allow_nan=True, default=str) + "\n")
    Path(out_md).write_text(markdown(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--preset", default="mini-m")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out or a.root)
    res = write(a.root, out / "g0_analysis.json", out / "g0_analysis.md", a.preset)
    print(json.dumps({"B1": res["titration"]["B1"]["label"], "B2": res["titration"]["B2"]["label"],
                      "decomposition": res["decomposition"]["label"]}))


if __name__ == "__main__":
    main()
