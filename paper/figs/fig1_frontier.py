"""Fig. 1: the memory-LM frontier under the two compute matches (original study, mini-m, 3 seeds).

Inputs (repository-relative):
  results/text-original-study/frontier_seed_means.csv  seed means of (eta, adapted L_CAL, M),
                                                        one row per (arm, eta)
  results/text-original-study/analysis.json            per-seed eta grids, the selected points
                                                        and L_SO,min per seed

Panel (a) equal GPU-seconds: SO@1200 against FO-CM@3469, with FO-CE@1200 in grey as the
          equal-step reference, SO's floor + .01 operating point, the CAL cover budget and
          both arms' cover points (R_cov is the vertical gap from the star to each).
Panel (b) equal tokens: SO@1200 against FO-CE@1200 near SO's floor, regions annotated.

Every drawn point is measured. Bold markers are seed means at each eta (the CSV), joined in
eta order; thin lines are the three single-seed eta curves (the JSON grids), drawn instead of
a band because L(eta) is U-shaped, so M cannot be interpolated as a function of L. Nothing is
fitted or extrapolated. The selected points are seed means of the selected M, placed at the
seed-mean budget (for seed 2, FO-CM's maximum lies inside the budget at x5).
Run from the repository root:
  uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/fig1_frontier.py
"""
import csv
import json
import os

import numpy as np

from figstyle import BASELINE, PALETTE, TEXT, figure_grid, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, os.pardir, "results", "text-original-study")

ARMS = ("so", "fo_ce", "fo_cm")
NAMES = {
    "so": "SO (1200 steps)",
    "fo_ce": "FO-CE (1200 steps, equal steps)",
    "fo_cm": "FO-CM (3469 steps, equal GPU-s)",
}
COLORS = {"so": PALETTE["blue"], "fo_ce": "#6E6E6E", "fo_cm": PALETTE["orange"]}
GRAY = "#555555"
FS = 5.0  # annotation font size (pt)


def load():
    """Seed means from the CSV; per-seed grids and selected points from the JSON."""
    d = json.load(open(os.path.join(RES, "analysis.json")))
    means = {a: [] for a in ARMS}
    with open(os.path.join(RES, "frontier_seed_means.csv")) as f:
        for row in csv.DictReader(f):
            means[row["arm"]].append(
                (float(row["eta"]), float(row["L_cal_mean"]), float(row["M_mean"]))
            )
    for a in ARMS:
        means[a].sort()
    return d, means


def per_seed(seed, arm):
    """(eta, L_CAL, M) of one seed's eta grid for one arm, sorted by eta."""
    return np.array(sorted((float(t[0]), float(t[1]), float(t[2])) for t in seed["grids"][arm]))


def curve(ax, d, means, arm, lw=1.1):
    """Single-seed eta curves (thin) under the seed-mean eta curve (bold, measured points)."""
    for s in d["seeds"]:
        g = per_seed(s, arm)
        ax.plot(g[:, 1], g[:, 2], color=COLORS[arm], linewidth=0.45, alpha=0.45, zorder=2)
    pts = np.array(means[arm])
    ax.plot(pts[:, 1], pts[:, 2], color=COLORS[arm], marker="o", markersize=2.4,
            linewidth=lw, label=NAMES[arm], zorder=3)


def eta_label(ax, means, arm, eta, dx, dy, ha="center", va="center"):
    """Label one measured seed-mean point by its eta multiplier, offset in points."""
    hit = [p for p in means[arm] if abs(p[0] - eta) < 1e-6]
    _, x, y = hit[0]
    ax.annotate(f"$\\times${eta:g}", (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=FS, color=COLORS[arm], zorder=6)


def callout(ax, xy, xytext, text, color, ha="left", va="center"):
    """A label in data space joined to its point by a thin leader line.

    The leader is a separate line artist so that only the text itself counts as text."""
    ax.text(*xytext, text, ha=ha, va=va, fontsize=FS, color=color, linespacing=1.25,
            zorder=6)
    ax.plot([xy[0], xytext[0]], [xy[1], xytext[1]], color=color, linewidth=0.5, zorder=5)


def panel_a(ax, d, means):
    """Equal GPU-seconds: SO@1200 and FO-CM@3469, FO-CE@1200 as the equal-step reference."""
    ax.set_xlim(3.84, 4.56)
    ax.set_ylim(0.22, 0.77)
    curve(ax, d, means, "fo_ce", lw=0.9)
    curve(ax, d, means, "fo_cm")
    curve(ax, d, means, "so")
    l_so = np.array([s["L_SO_min"] for s in d["seeds"]])
    budget = float(np.mean(l_so + .02))
    so_op = (float(np.mean(l_so + .01)), float(np.mean([s["so"]["M"] for s in d["seeds"]])))
    cm_cov = (budget, float(np.mean([s["fo_cm"]["M"] for s in d["seeds"]])))
    ce_cov = (budget, float(np.mean([s["fo_ce"]["M"] for s in d["seeds"]])))

    ax.axvline(budget, color=BASELINE, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    ax.annotate("CAL cover budget\n$B_{FO}=L_{SO,\\min}+.02$", xy=(budget, 0.765),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="top",
                fontsize=FS, color=GRAY, linespacing=1.3)
    ax.plot(*so_op, marker="*", markersize=8, color=COLORS["so"], markeredgecolor="white",
            markeredgewidth=0.4, linestyle="none", zorder=7)
    ax.plot(*cm_cov, marker="s", markersize=4.2, markerfacecolor="white",
            markeredgecolor=PALETTE["red"], markeredgewidth=1.0, linestyle="none", zorder=7)
    ax.plot(*ce_cov, marker="D", markersize=3.6, markerfacecolor="white",
            markeredgecolor=COLORS["fo_ce"], markeredgewidth=1.0, linestyle="none", zorder=7)
    callout(ax, so_op, (4.235, 0.640), "SO operating point\n$L_{SO,\\min}+.01$",
            COLORS["so"], ha="right")
    callout(ax, cm_cov, (4.40, 0.355), "FO-CM cover point\n(damage-flagged)",
            PALETTE["red"], ha="left")
    callout(ax, ce_cov, (4.40, 0.655), "FO-CE cover point", COLORS["fo_ce"], ha="left")

    eta_label(ax, means, "fo_cm", 1.0, 5, -1, ha="left")
    eta_label(ax, means, "fo_cm", 4.0, 0, 6)
    eta_label(ax, means, "fo_cm", 8.0, 0, 6)
    eta_label(ax, means, "fo_ce", 1.0, -5, 0, ha="right")
    eta_label(ax, means, "fo_ce", 6.0, -2, 7, ha="right")
    eta_label(ax, means, "so", 0.75, 0, -7)
    eta_label(ax, means, "so", 3.0, 5, 0, ha="left")


def panel_b(ax, d, means):
    """Equal tokens: SO@1200 against FO-CE@1200 near SO's floor, with the two regions."""
    ax.set_xlim(4.25, 4.46)
    ax.set_ylim(0.26, 0.75)
    curve(ax, d, means, "fo_ce")
    curve(ax, d, means, "so")
    l_so = float(np.mean([s["L_SO_min"] for s in d["seeds"]]))
    ax.axvline(l_so, color=BASELINE, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    ax.annotate("SO's floor\n$L_{SO,\\min}$", xy=(l_so, 0.745), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="top", fontsize=FS, color=GRAY,
                linespacing=1.3)
    ax.text(4.253, 0.61, "FO-CE owns the\nlow-loss end:\nno SO point here", ha="left",
            va="top", fontsize=FS, color=COLORS["fo_ce"], linespacing=1.25)
    ax.text(4.457, 0.745, "SO owns the\nhigh-memory end", ha="right", va="top",
            fontsize=FS, color=COLORS["so"], linespacing=1.25)
    eta_label(ax, means, "so", 0.75, 5, -3, ha="left")
    eta_label(ax, means, "so", 1.5, 5, 0, ha="left")
    eta_label(ax, means, "so", 3.0, 5, -3, ha="left")
    eta_label(ax, means, "fo_ce", 0.75, 5, -3, ha="left")
    eta_label(ax, means, "fo_ce", 4.0, 2, -7, ha="left")


def main():
    use_style()
    d, means = load()
    fig, axes = figure_grid(nrows=1, ncols=2, width=TEXT, ratio=0.42)
    panel_labels(axes, ["(a)", "(b)"])
    for ax in axes:
        ax.set_xlabel("adapted held-out loss $L_{CAL}$ (nats/token)", fontsize=7)
        ax.set_ylabel("probe-memory contrast $M$ (nats)", fontsize=7)
        ax.tick_params(labelsize=6)
    panel_a(axes[0], d, means)
    panel_b(axes[1], d, means)
    handles, labels = axes[0].get_legend_handles_labels()
    by = dict(zip(labels, handles))
    order = [NAMES[a] for a in ("so", "fo_ce", "fo_cm")]
    fig.legend([by[k] for k in order], order, fontsize=6.0, loc="outside lower center",
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
    save(fig, os.path.join(HERE, "fig1-frontier"), formats=("pdf",))


if __name__ == "__main__":
    main()
