"""P0.0 analysis: is the lin inner loss free on icl_mc?

Reads the CELL_RESULT lines of one run log and prints a P00_SUMMARY JSON line plus a
table. The gate is a seed-paired non-inferiority test: for d_s = L(e2e1_lin) - L(e2e1)
on seed s, the one-sided 95% upper bound mean(d) + t_{.95, n-1} sd(d) / sqrt(n) must be
below 0.01 nats for both eval CE and late CE (intersection-union, each at .05).

Usage: python scripts/analyze_p00.py <run.log> [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ARMS = ("e2e1", "e2e1_lin", "e2e2")
MARGIN = 0.01
MIN_PAIRS = 5
# one-sided 95% Student t quantiles by degrees of freedom
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
       9: 1.833, 10: 1.812}
# round-1 (R1) late CE on icl_mc seed 0, the reproduction check
R1_LATE = {"e2e1": 2.018, "e2e2": 1.990}
R1_TOL = 0.02


def parse(path: str | Path) -> dict:
    """``(seed, arm) -> record`` from the CELL_RESULT lines (the last one wins)."""
    cells = {}
    for line in Path(path).open():
        if line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            cells[(c["seed"], c["arm"])] = c
    return cells


def usable(rec: dict | None) -> bool:
    """A finished, finite, non-diverged cell with an eval block."""
    if not rec or rec.get("skipped") or rec.get("diverged") or "eval" not in rec:
        return False
    ev = rec["eval"]
    return all(math.isfinite(ev.get(k, float("nan"))) for k in ("ce", "ce_late"))


def upper_bound(diffs: list[float]) -> dict:
    """Mean, sd and the one-sided 95% t upper bound of paired differences."""
    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    if n < 2:
        return dict(n=n, mean=mean, sd=None, ub95=None)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    return dict(n=n, mean=mean, sd=sd, ub95=mean + T95[min(n - 1, 10)] * sd / math.sqrt(n))


def analyze(cells: dict) -> dict:
    seeds = sorted({s for s, _ in cells})
    rows, pairs = [], []
    for s in seeds:
        row = {"seed": s}
        for arm in ARMS:
            rec = cells.get((s, arm))
            ok = usable(rec)
            row[f"{arm}_ce"] = rec["eval"]["ce"] if ok else None
            row[f"{arm}_late"] = rec["eval"]["ce_late"] if ok else None
            row[f"{arm}_eta"] = rec.get("eta_final") if ok else None
            row[f"{arm}_ms"] = rec.get("ms_step") if ok else None
            row[f"{arm}_status"] = ("ok" if ok else "missing" if rec is None else
                                    "skipped" if rec.get("skipped") else
                                    "diverged" if rec.get("diverged") else "no_eval")
        rows.append(row)
        if row["e2e1_ce"] is not None and row["e2e1_lin_ce"] is not None:
            pairs.append(row)
    d_ce = [r["e2e1_lin_ce"] - r["e2e1_ce"] for r in pairs]
    d_late = [r["e2e1_lin_late"] - r["e2e1_late"] for r in pairs]
    ce, late = upper_bound(d_ce), upper_bound(d_late)
    if len(pairs) < MIN_PAIRS:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if ce["ub95"] < MARGIN and late["ub95"] < MARGIN else "FAIL"
    so = [r for r in rows if r["e2e2_late"] is not None and r["e2e1_late"] is not None]
    so_gap = upper_bound([r["e2e1_late"] - r["e2e2_late"] for r in so])
    lin_so = [r for r in rows if r["e2e2_late"] is not None and r["e2e1_lin_late"] is not None]
    lin_vs_so = upper_bound([r["e2e1_lin_late"] - r["e2e2_late"] for r in lin_so])
    s0 = {r["seed"]: r for r in rows}.get(0, {})
    sanity = {}
    for arm, ref in R1_LATE.items():
        val = s0.get(f"{arm}_late")
        sanity[f"{arm}_s0_late"] = val
        sanity[f"{arm}_s0_matches_r1"] = (val is not None and abs(val - ref) <= R1_TOL)
    return dict(n_paired=len(pairs), margin=MARGIN,
                lin_minus_ce=dict(ce=ce, late=late),
                gate=dict(verdict=verdict, rule=f"ub95(ce) < {MARGIN} and ub95(late) < {MARGIN}"
                          f", n >= {MIN_PAIRS}"),
                e2e1_minus_e2e2_late=so_gap, lin_minus_e2e2_late=lin_vs_so,
                sanity=sanity, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = analyze(parse(args.log))
    print("seed | " + " | ".join(f"{a} ce / late" for a in ARMS))
    for r in res["rows"]:
        cells = []
        for a in ARMS:
            cells.append("—" if r[f"{a}_ce"] is None else f"{r[f'{a}_ce']:.4f} / {r[f'{a}_late']:.4f}")
        print(f"{r['seed']} | " + " | ".join(cells))
    g = res["lin_minus_ce"]
    for key in ("ce", "late"):
        x = g[key]
        print(f"lin - CE ({key}): n={x['n']} mean={x['mean']:+.5f} sd={x['sd']} ub95={x['ub95']}")
    print(f"GATE {res['gate']['verdict']} ({res['gate']['rule']})")
    print("P00_SUMMARY " + json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "p00-cells.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
            w.writeheader()
            w.writerows(res["rows"])
        (out / "p00-summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
