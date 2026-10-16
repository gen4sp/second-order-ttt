"""Registered SOQ-FS matched-LM frontier analyses.

The module consumes only the small per-document ``g0_read_*.npz`` files written
by :func:`tttmem.p2.mini._g0_item`; checkpoints are never loaded here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

FO = "e2e1-ce"
SO = "e2e2"
FO_READS = ("ce", "eta125", "eta150", "eta200", "eta250", "eta300", "eta400", "eta500", "eta600")
SO_READS = ("ce", "eta150")
T90_DF2 = 1.886
MINI_M_RHO = 0.089
MINI_M_RHO_UB = 0.245


def interval(values, t: float = T90_DF2) -> dict:
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if not len(finite):
        return {"n": 0, "mean": math.nan, "sd": math.nan, "lb": math.nan, "ub": math.nan, "values": []}
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if len(finite) > 1 else math.nan
    half = t * sd / math.sqrt(len(finite)) if math.isfinite(sd) else math.nan
    return {"n": int(len(finite)), "mean": mean, "sd": sd,
            "lb": mean - half if math.isfinite(half) else math.nan,
            "ub": mean + half if math.isfinite(half) else math.nan,
            "values": finite.tolist()}


def _payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        lsum = np.asarray(z["L_adapt_sum"], dtype=float)
        lcount = np.asarray(z["L_adapt_count"], dtype=float)
        return {
            "name": path.stem.removeprefix("g0_read_"),
            "eta": float(z["eta"]),
            "M_doc": np.asarray(z["M_doc"], dtype=float),
            "M": float(np.mean(z["M_doc"])),
            "L_doc": lsum / np.maximum(1.0, lcount),
            "L": float(lsum.sum() / max(1.0, lcount.sum())),
            "L_noadapt_doc": np.asarray(z["L_noadapt_doc"], dtype=float),
            "L_noadapt": float(z["L_noadapt"]),
        }


def load_cells(root: str | Path, preset: str = "mini-s", seeds=(0, 1, 2)) -> dict:
    root = Path(root)
    cells = {}
    for arm, names in ((SO, SO_READS), (FO, FO_READS)):
        for seed in seeds:
            cell = root / preset / arm / f"s{seed}"
            reads = {}
            for name in names:
                path = cell / f"g0_read_{name}.npz"
                if path.exists():
                    reads[name] = _payload(path)
            cells[(arm, seed)] = reads
    return cells


def interpolate(points: list[dict], target: float) -> dict | None:
    """Return eta and linearly interpolated M at target L, with no extrapolation."""
    pts = sorted(points, key=lambda p: p["eta"])
    if not pts or not math.isfinite(target):
        return None
    for p in pts:
        if math.isclose(p["L"], target, rel_tol=0.0, abs_tol=1e-12):
            return {"eta": p["eta"], "M": p["M"], "w": 0.0,
                    "lo": p["name"], "hi": p["name"]}
    for lo, hi in zip(pts, pts[1:]):
        if hi["L"] <= lo["L"]:
            continue
        if lo["L"] <= target <= hi["L"]:
            w = (target - lo["L"]) / (hi["L"] - lo["L"])
            return {"eta": lo["eta"] + w * (hi["eta"] - lo["eta"]),
                    "M": lo["M"] + w * (hi["M"] - lo["M"]), "w": float(w),
                    "lo": lo["name"], "hi": hi["name"]}
    return None


def _rho(so: dict, fo_ce: dict, match: dict | None) -> float:
    if match is None:
        return math.nan
    denom = so["M"] - fo_ce["M"]
    return (so["M"] - match["M"]) / denom if denom > 0 else math.nan


def analysis_a(cells: dict, seeds=(0, 1, 2)) -> dict:
    per_seed = {}
    frontier = []
    for seed in seeds:
        so = cells.get((SO, seed), {})
        fo = cells.get((FO, seed), {})
        if "ce" not in so or not all(n in fo for n in FO_READS):
            per_seed[seed] = {"status": "NE", "reason": "missing registered read"}
            continue
        match = interpolate(list(fo.values()), so["ce"]["L"])
        rho = _rho(so["ce"], fo["ce"], match)
        per_seed[seed] = {
            "status": "ok" if math.isfinite(rho) else "NE",
            "reason": "" if math.isfinite(rho) else "target outside FO grid or non-positive denominator",
            "eta_star": match["eta"] if match else math.nan,
            "M_star": match["M"] if match else math.nan,
            "L_SO": so["ce"]["L"], "M_SO": so["ce"]["M"],
            "L_noadapt_SO": so["ce"]["L_noadapt"],
            "L_noadapt_FO": fo["ce"]["L_noadapt"],
            "rho_match": rho,
            "dM": so["ce"]["M"] - fo["ce"]["M"],
        }
        for arm, reads in ((SO, so), (FO, fo)):
            for name, p in reads.items():
                frontier.append({"seed": seed, "arm": arm, "read": name, "eta": p["eta"],
                                 "L_adapt": p["L"], "M": p["M"],
                                 "L_noadapt": p["L_noadapt"]})
    rhos = [per_seed[s].get("rho_match", math.nan) for s in seeds]
    lm_gap = [per_seed[s].get("L_noadapt_SO", math.nan) - per_seed[s].get("L_noadapt_FO", math.nan)
              for s in seeds]
    return {"per_seed": per_seed, "rho_match": interval(rhos),
            "noadapt_lm_gap_SO_minus_FO": interval(lm_gap),
            "noadapt_lm_gap_per_seed": {str(s): lm_gap[i] for i, s in enumerate(seeds)},
            "frontier": frontier,
            "target_outside_grid": any(v.get("status") != "ok" for v in per_seed.values())}


def _split_result(so: dict, fo: dict, cal: np.ndarray, test: np.ndarray) -> dict:
    fo_points = []
    for name in FO_READS:
        p = fo[name]
        fo_points.append({"name": name, "eta": p["eta"], "L": float(p["L_doc"][cal].mean()),
                          "M": p["M"]})
    target = float(so["ce"]["L_doc"][cal].mean())
    match = interpolate(fo_points, target)
    if match is None:
        return {"eta_star": math.nan, "rho_match": math.nan, "L_match_error_TEST": math.nan,
                "target": target, "status": "NE"}
    # The same linear weight is used for the TEST adapted-L check.
    by_name = {p["name"]: p for p in fo_points}
    lo, hi = by_name[match["lo"]], by_name[match["hi"]]
    lt_lo = float(fo[match["lo"]]["L_doc"][test].mean())
    lt_hi = float(fo[match["hi"]]["L_doc"][test].mean())
    l_test_star = lt_lo + match["w"] * (lt_hi - lt_lo)
    l_so_test = float(so["ce"]["L_doc"][test].mean())
    rho = _rho(so["ce"], fo["ce"], match)
    return {"eta_star": match["eta"], "rho_match": rho,
            "L_match_error_TEST": l_test_star - l_so_test,
            "target": target, "status": "ok"}


def analysis_b(cells: dict, seeds=(0, 1, 2), random_splits: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    per_seed = {}
    for s in seeds:
        so, fo = cells.get((SO, s), {}), cells.get((FO, s), {})
        if "ce" not in so or not all(n in fo for n in FO_READS):
            per_seed[s] = {"status": "NE", "reason": "missing registered read"}
            continue
        n = len(so["ce"]["L_doc"])
        if n != 128 or any(len(fo[nm]["L_doc"]) != n for nm in FO_READS):
            per_seed[s] = {"status": "NE", "reason": "held-out document count is not 128"}
            continue
        even = np.arange(0, n, 2)
        odd = np.arange(1, n, 2)
        fixed = _split_result(so, fo, even, odd)
        random = []
        for _ in range(random_splits):
            perm = rng.permutation(n)
            random.append(_split_result(so, fo, perm[: n // 2], perm[n // 2 :]))
        rv = np.asarray([r["rho_match"] for r in random], dtype=float)
        ev = np.asarray([r["L_match_error_TEST"] for r in random], dtype=float)
        per_seed[s] = {"fixed_even_odd": fixed,
                       "random_rho": {"n": int(np.isfinite(rv).sum()),
                                       "median": float(np.nanmedian(rv)),
                                       "p05": float(np.nanpercentile(rv, 5)),
                                       "p95": float(np.nanpercentile(rv, 95))},
                       "random_abs_L_error": {"median": float(np.nanmedian(np.abs(ev))),
                                              "p95": float(np.nanpercentile(np.abs(ev), 95))},
                       "status": "ok" if fixed["status"] == "ok" else "NE"}
    fixed_rho = [per_seed[s].get("fixed_even_odd", {}).get("rho_match", math.nan) for s in seeds]
    fixed_err = [per_seed[s].get("fixed_even_odd", {}).get("L_match_error_TEST", math.nan) for s in seeds]
    return {"per_seed": per_seed, "fixed_rho": interval(fixed_rho),
            "fixed_L_match_error_TEST": interval(fixed_err),
            "random_splits": random_splits, "rng_seed": seed}


def analyse(root: str | Path, preset: str = "mini-s", seeds=(0, 1, 2)) -> dict:
    cells = load_cells(root, preset, seeds)
    a = analysis_a(cells, seeds)
    b = analysis_b(cells, seeds)
    mini_s = a["rho_match"]["mean"]
    delta = MINI_M_RHO - mini_s if math.isfinite(mini_s) else math.nan
    if not (math.isfinite(mini_s) and a["rho_match"]["n"] == len(seeds)
            and b["fixed_rho"]["n"] == len(seeds)):
        trend = "NE"
    elif delta > 0.15:
        trend = "TREND-RISING"
    elif delta < -0.15:
        trend = "TREND-FALLING"
    else:
        trend = "TREND-FLAT"
    return {"preset": preset, "analysis_a": a, "analysis_b": b,
            "comparison": {"mini_m_rho": MINI_M_RHO, "mini_m_rho_ub": MINI_M_RHO_UB,
                            "mini_s_rho": mini_s, "delta_mini_m_minus_mini_s": delta,
                            "flag_delta_gt_0.15": bool(math.isfinite(delta) and delta > 0.15),
                            "verdict": trend}}


def markdown(res: dict) -> str:
    a, b, c = res["analysis_a"], res["analysis_b"], res["comparison"]
    iv = lambda x: "NE" if not math.isfinite(x.get("mean", math.nan)) else \
        f"{x['mean']:+.3f} [{x['lb']:+.3f}, {x['ub']:+.3f}]"
    lines = [f"# SOQ-FS mini-s results", "",
             f"- Verdict: **{c['verdict']}**; mini-s minus prior mini-m comparison Δρ = {c['delta_mini_m_minus_mini_s']:+.3f}.",
             f"- Analysis A ρ_match: {iv(a['rho_match'])}; Analysis B fixed even/odd: {iv(b['fixed_rho'])}.",
             f"- SO − FO no-adapt LM gap (nats/token): {iv(a['noadapt_lm_gap_SO_minus_FO'])}.",
             f"- Target outside FO grid: {a['target_outside_grid']}; mini-m reference ρ = +.089, UB +.245.", "",
             "## Frontier", "", "| seed | arm | read | η | L_adapt | M | L_noadapt |", "|---:|---|---|---:|---:|---:|---:|"]
    for row in a["frontier"]:
        lines.append(f"| {row['seed']} | {row['arm']} | {row['read']} | {row['eta']:.2f} | {row['L_adapt']:.6f} | {row['M']:.6f} | {row['L_noadapt']:.6f} |")
    lines += ["", "## Per-seed matches", "", "| seed | η* A | ρ A | η* B fixed | ρ B fixed | TEST L error |", "|---:|---:|---:|---:|---:|---:|"]
    for s, rec in a["per_seed"].items():
        rb = b["per_seed"].get(s, {}).get("fixed_even_odd", {})
        lines.append(f"| {s} | {rec.get('eta_star', math.nan):.3f} | {rec.get('rho_match', math.nan):+.3f} | {rb.get('eta_star', math.nan):.3f} | {rb.get('rho_match', math.nan):+.3f} | {rb.get('L_match_error_TEST', math.nan):+.6f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = analyse(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fs_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    (out / "fs_analysis.md").write_text(markdown(result))
    # Keep the compact per-document reads beside the report. Checkpoints are
    # never copied; the NPZ files contain only M/L vectors and metadata.
    src_root = Path(args.root)
    data_out = out / "data"
    for src in src_root.glob("mini-s/*/s*/g0_read_*.npz"):
        rel = src.relative_to(src_root)
        dst = data_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(json.dumps({"verdict": result["comparison"]["verdict"],
                      "rho_match": result["analysis_a"]["rho_match"],
                      "rho_match_caltest": result["analysis_b"]["fixed_rho"]}, allow_nan=True, sort_keys=True))


if __name__ == "__main__":
    main()
