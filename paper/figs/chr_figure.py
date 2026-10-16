"""Fig. 3: every arm loses the binding at first; only FO-CE's kv write residual then collapses.

Input (repository-relative): results/toy-chronology/
  series.csv  per cell, branch and logged step: query accuracy `a` and median kv residual `r`
  events.csv  per cell and branch: acquisition t_acq, starvation onset t_starve, binding loss t_loss

All series are aligned at each unit's registered t_loss. Arms (echo, fixed eta = 40, main
runs): FO-CE (units s0, s2-s6), SO and FO-CE + 3% copy noise (s0-s5). Panel (c) shows the FO
forks taken at t_acq (CE control, residual floor 0.03, direction control); grey is the shared
main trajectory before the fork. Triangles mark acquisition (in (c), the fork step, which is
t_acq); dots mark the registered starvation onset (first of two consecutive logs with residual
< .02), which only FO-CE units reach inside the window. Thin lines are seeds, thick lines
medians over the seeds present at that offset (at least 3).
Run from the repository root:
  uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/chr_figure.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))
from figstyle import BASELINE, PALETTE, TEXT, figure_grid, label_ends, panel_labels, save, use_style  # noqa: E402

SC = HERE.parents[1] / "results" / "toy-chronology"
FO_UNITS = (0, 2, 3, 4, 5, 6)
ARMS = (("e2e1emb", FO_UNITS, "FO-CE", PALETTE["orange"]),
        ("e2e1emb_cn97", range(6), "3% noise", PALETTE["green"]),
        ("e2e2emb", range(6), "SO", PALETTE["blue"]))
FORKS = (("fork-ce-twin", "CE control", PALETTE["orange"]),
         ("fork-nc03", "direction\ncontrol", PALETTE["purple"]),
         ("fork-rp03", "floor 0.03", PALETTE["red"]))
LO, HI = -320, 600
R_STAR, BIND = 0.02, 0.90
ACQ = dict(marker="^", ms=3.7, mew=0.4, mec="white", ls="none", zorder=4)
STV = dict(marker="o", ms=3.3, mew=0.4, mec="white", ls="none", zorder=4)

series = defaultdict(list)
with open(SC / "series.csv") as fh:
    for row in csv.DictReader(fh):
        series[(row["cell_id"], row["branch"])].append(
            (int(row["step"]), float(row["a"]), float(row["r"]) if row["r"] else np.nan))
series = {k: sorted(v) for k, v in series.items()}
ev = {}
with open(SC / "events.csv") as fh:
    for row in csv.DictReader(fh):
        ev[(row["cell_id"], row["branch"])] = row


def cell(arm, seed):
    return f"echo/s{seed}/{arm}/eta40/st1500"


def aligned(cid, branch, t_loss, until=None):
    """(offset, acc, resid) arrays within [LO, HI] of t_loss, optionally only steps <= until."""
    rows = [r for r in series[(cid, branch)] if until is None or r[0] <= until]
    st, a, r = (np.array(x, dtype=float) for x in zip(*rows))
    keep = (st - t_loss >= LO) & (st - t_loss <= HI)
    return st[keep] - t_loss, a[keep], r[keep]


def at(c, off, col):
    """Value of column `col` of an aligned curve at offset `off` (the logged point itself)."""
    hit = np.flatnonzero(c[0] == off)
    return c[col][hit[0]] if len(hit) else np.nan


def median_over(curves, col):
    grid = np.arange(LO, HI + 1, 5)
    out = np.full(grid.shape, np.nan)
    for i, x in enumerate(grid):
        vals = [c[col][c[0] == x][0] for c in curves if np.any(c[0] == x)]
        if len(vals) >= 3:
            out[i] = np.nanmedian(vals)
    return grid, out


use_style()
fig, axes = figure_grid(1, 3, width=TEXT, ratio=0.36, sharey=False)
ax_r, ax_a, ax_f = axes

ends_r, ends_a = [], []
for arm, seeds, name, color in ARMS:
    evs = [ev[(cell(arm, s), "main")] for s in seeds]
    curves = [aligned(cell(arm, s), "main", int(e["t_loss"])) for s, e in zip(seeds, evs)]
    for c in curves:
        ax_r.plot(c[0], c[2], color=color, lw=0.4, alpha=0.3)
        ax_a.plot(c[0], c[1], color=color, lw=0.4, alpha=0.3)
    g, m = median_over(curves, 2)
    ends_r.append(ax_r.plot(g, m, color=color, lw=1.6)[0])
    g, m = median_over(curves, 1)
    ends_a.append(ax_a.plot(g, m, color=color, lw=1.6)[0])
    # registered events on each unit's own line
    for c, e in zip(curves, evs):
        off = int(e["t_acq"]) - int(e["t_loss"])
        ax_r.plot([off], [at(c, off, 2)], color=color, **ACQ)
        ax_a.plot([off], [at(c, off, 1)], color=color, **ACQ)
        if e["t_starve"]:
            off = int(e["t_starve"]) - int(e["t_loss"])
            if LO <= off <= HI:
                ax_r.plot([off], [at(c, off, 2)], color=color, **STV)

# The three forks of a unit lose the binding at the same step, so the CE control's t_loss aligns all.
loss_f = {s: int(ev[(cell("e2e1emb", s), "fork-ce-twin")]["t_loss"]) for s in FO_UNITS}
pre = [aligned(cell("e2e1emb", s), "main", loss_f[s], until=int(ev[(cell("e2e1emb", s), "main")]["t_acq"]))
       for s in FO_UNITS]
for c in pre:
    ax_f.plot(c[0], c[1], color=BASELINE, lw=0.4, alpha=0.5)
ends_f = []
for branch, name, color in FORKS:
    curves = [aligned(cell("e2e1emb", s), branch, loss_f[s]) for s in FO_UNITS]
    for c in curves:
        ax_f.plot(c[0], c[1], color=color, lw=0.4, alpha=0.3)
    g, m = median_over(curves, 1)
    ends_f.append(ax_f.plot(g, m, color=color, lw=1.6)[0])
for c in pre:  # the fork step = the main run's acquisition = last point of the grey pre-fork line
    ax_f.plot(c[0][-1:], c[1][-1:], color="#555555", **ACQ)

ax_r.set_yscale("log")
ax_r.set_ylim(1e-4, 1.5)
ax_r.axhline(R_STAR, color=BASELINE, ls="--", lw=0.8)
ax_r.annotate("0.02", xy=(HI, R_STAR), xytext=(3, -2), textcoords="offset points",
              ha="left", va="top", color=BASELINE, fontsize="small")
ax_r.set_ylabel("kv write residual")
for ax in (ax_a, ax_f):
    ax.set_ylim(0, 1.03)
    ax.axhline(BIND, color=BASELINE, ls=":", lw=0.8)
ax_a.set_ylabel("query accuracy")
ax_f.set_ylabel("query accuracy, FO forks")
for ax in axes:
    ax.axvline(0, color=BASELINE, ls=":", lw=0.8)
    ax.set_xlim(LO, HI)
    ax.set_xticks([-300, 0, 300, 600])
    ax.set_xlabel(r"steps from binding loss")
key = [Line2D([], [], color="#555555", **ACQ), Line2D([], [], color="#555555", **STV)]
ax_r.legend(key, ["acquisition (c: fork)", "starvation onset"], loc="lower left",
            fontsize=6.5, handletextpad=0.1, labelspacing=0.3, borderaxespad=0.2)
label_ends(ax_r, ends_r, [a[2] for a in ARMS])
label_ends(ax_a, ends_a, [a[2] for a in ARMS])
label_ends(ax_f, ends_f, [f[1] for f in FORKS])
panel_labels(axes)
print(save(fig, str(OUT / "chr-chronology"), formats=("pdf",)))
