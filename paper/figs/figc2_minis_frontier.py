"""Fig. 5 (App. C): the mini-s equal-token frontier (matched-loss protocol, 3 seeds).

Input (repository-relative): results/text-mini-s-frontier/analysis.json, whose
`analysis_a.frontier` lists one measured read per (arm, eta, seed) with the adapted loss
L_adapt and the probe-memory contrast M.

Only measured points are drawn; the bands are two-sided 95% t intervals of M over the three
seeds at each eta (df = 2). SO's eta grid stops at x1.5, which is a grid edge and is marked
as such. Both arms' L_adapt rise monotonically with eta here, so the band is well defined.
Run from the repository root:
  uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/figc2_minis_frontier.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from figstyle import BASELINE, COLUMN, PALETTE, figure, mean_ci, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, os.pardir, "results", "text-mini-s-frontier")
ARMS = {"e2e2": ("so", PALETTE["blue"]), "e2e1-ce": ("fo_ce", PALETTE["black"])}
LABEL = {"so": "SO (1200 steps)", "fo_ce": "FO-CE (1200 steps, equal steps)"}


def curves(rows):
    """Seed-mean (eta, L, M) per arm, sorted by eta, plus the per-seed M for the band."""
    out = {}
    for code, (key, _) in ARMS.items():
        pts = sorted({(r["eta"], r["seed"]) for r in rows if r["arm"] == code})
        etas = sorted({e for e, _ in pts})
        out[key] = []
        for e in etas:
            sel = [r for r in rows if r["arm"] == code and r["eta"] == e]
            out[key].append((e,
                             float(np.mean([r["L_adapt"] for r in sel])),
                             float(np.mean([r["M"] for r in sel])),
                             np.array([r["M"] for r in sel])))
    return out


def main():
    use_style()
    d = json.load(open(os.path.join(RES, "analysis.json")))
    rows = d["analysis_a"]["frontier"]
    cur = curves(rows)
    fig, ax = figure(width=COLUMN, ratio=1.02)
    for key, color in ARMS.values():
        pts = cur[key]
        x = np.array([p[1] for p in pts])
        runs = np.array([p[3] for p in pts])
        assert np.all(np.diff(x) > 0), "band needs L rising with eta"
        mean, lo, hi = mean_ci(runs.T)  # runs is (points, seeds)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, marker="o", markersize=2.6, linewidth=1.1,
                label=LABEL[key], zorder=3)
        for e, xi, yi, _ in pts:
            if e in (1.0, 1.5):
                ax.annotate(f"$\\times${e:g}", (xi, yi), textcoords="offset points",
                            xytext=(5 if key == "so" else -5, 0),
                            ha="left" if key == "so" else "right", va="center",
                            fontsize=5.0, color=color)
    l_so = float(np.mean([d["analysis_a"]["per_seed"][s]["L_SO"] for s in ("0", "1", "2")]))
    ax.axvline(l_so, color=BASELINE, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    ax.annotate("SO's floor", xy=(l_so, 0.66), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="top", fontsize=5.0, color="#555555")
    ax.annotate("SO's grid ends at $\\times1.5$, so SO's\nhigh-memory edge is a grid-edge result\n"
                "($\\rho_{match}=-.249$ [$-.446$, $-.052$])",
                xy=(0.995, 0.985), xycoords="axes fraction", ha="right", va="top",
                fontsize=5.0, color="#555555", linespacing=1.35)
    ax.set_xlabel("adapted held-out loss $L_{adapt}$ (nats/token)", fontsize=7)
    ax.set_ylabel("probe-memory contrast $M$ (nats)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(4.44, 4.62)
    ax.set_ylim(0.32, 0.72)
    ax.legend(fontsize=5.6, loc="lower left", frameon=False, handlelength=1.5,
              borderpad=0.1, labelspacing=0.25)
    save(fig, os.path.join(HERE, "figc2-minis-frontier"), formats=("pdf",))


if __name__ == "__main__":
    main()