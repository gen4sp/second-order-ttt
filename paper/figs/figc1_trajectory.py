"""Fig. 4 (App. C): probe memory at default eta and the no-adapt loss against the training
GPU-seconds each arm was given (original study, mini-m, 3 seeds).

Input (repository-relative):
  results/text-original-study/analysis.json  per-seed M at the fixed eta x1 of each arm
                                             (M_fixed), the FO-CM M(x1) at the un-annealed
                                             step 2400 (M_x1_traj) and the no-adapt H_LM of
                                             each arm; bars are 95% t intervals over the
                                             three seeds (df = 2)

The x axis is wall-clock GPU-seconds relative to SO@1200, from the timing pilot r = 2.891
(RTX 4090; results/text-original-study/timing_pilot.json): an FO step costs 1/2.891 of an SO
step, so FO-CE@1200 sits at 0.35, FO-CM@2400 at 0.69 and FO-CM@3469 at 1.0. Filled markers are
annealed endpoints; the open marker is the un-annealed FO-CM@2400 checkpoint and is descriptive
only. No line joins the FO-CM checkpoints.
Run from the repository root:
  uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/figc1_trajectory.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from figstyle import PALETTE, TEXT, figure_grid, mean_ci, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, os.pardir, "results", "text-original-study")
R = 2.891  # wall-clock SO/FO step-time ratio, RTX 4090 timing pilot (timing_pilot.json, r)
SO_EQ = 1200 * R  # FO steps that cost what SO@1200 costs; N_CM = 3469
COL = {"so": PALETTE["blue"], "fo_ce": PALETTE["black"], "fo_cm": PALETTE["orange"]}


def load():
    """Per-seed M(x1), the FO-CM step-2400 M(x1) and the H_LM of each arm, from the JSON."""
    d = json.load(open(os.path.join(RES, "analysis.json")))
    m = {a: np.array([s["M_fixed"][a]["1"] for s in d["seeds"]]) for a in ("so", "fo_ce", "fo_cm")}
    hlm = {a: np.array([s["H_LM"][a] for s in d["seeds"]]) for a in ("so", "fo_ce", "fo_cm")}
    traj = {int(k): np.array([s["M_x1_traj"][k] for s in d["seeds"]])
            for k in d["seeds"][0]["M_x1_traj"]}
    return m, hlm, traj


def point(ax, x, y, arm, label, filled=True):
    """One endpoint with its 95% seed band and a short label."""
    y = np.asarray(y, float)
    ax.plot([x], [y.mean()], marker="o" if filled else "o", markersize=4.0,
            markerfacecolor=COL[arm] if filled else "white",
            markeredgecolor=COL[arm], markeredgewidth=1.0, linestyle="none", zorder=4)
    mean, lo, hi = mean_ci(y[:, None])
    ax.errorbar([x], mean, yerr=[mean - lo, hi - mean], color=COL[arm], linewidth=0.8,
                capsize=1.6, linestyle="none", zorder=3)
    # labels sit beside the marker, clear of its error bar: right of FO-CE@1200, left of
    # the later checkpoints
    right = x < 0.5
    ax.annotate(label, xy=(x, y.mean()), xytext=(6 if right else -6, 0),
                textcoords="offset points", ha="left" if right else "right", va="center",
                fontsize=5.0,
                color=COL[arm])


def panel(ax, values, ylabel, ylim, fmt, note):
    """One endpoint panel: y against training GPU-seconds."""
    x_ce, x_cm, x_cm24 = 1200 / SO_EQ, 3469 / SO_EQ, 2400 / SO_EQ
    point(ax, x_ce, values["fo_ce"], "fo_ce", f"FO-CE@1200\n{fmt.format(np.mean(values['fo_ce']))}")
    point(ax, 1.0, values["so"], "so", f"SO@1200\n{fmt.format(np.mean(values['so']))}")
    point(ax, x_cm, values["fo_cm"], "fo_cm",
          f"FO-CM@3469\n{fmt.format(np.mean(values['fo_cm']))}")
    if "fo_cm24" in values:
        point(ax, x_cm24, values["fo_cm24"], "fo_cm",
              f"FO-CM@2400\n{fmt.format(np.mean(values['fo_cm24']))}", filled=False)
    ax.set_xlim(0.25, 1.14)
    ax.set_ylim(*ylim)
    ax.set_xlabel("training GPU-seconds (SO@1200 = 1)", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.text(0.265, ylim[0] + 0.04 * (ylim[1] - ylim[0]), note, fontsize=5.0, color="#555555",
            ha="left", va="bottom")


def main():
    use_style()
    m, hlm, traj = load()
    fig, axes = figure_grid(nrows=1, ncols=2, width=TEXT, ratio=0.46)
    panel_labels(axes, ["(a)", "(b)"])
    panel(axes[0],
          {"fo_ce": m["fo_ce"], "so": m["so"], "fo_cm": m["fo_cm"], "fo_cm24": traj[2400]},
          "probe-memory contrast $M$ at $\\eta\\times1$ (nats)", (0.15, 0.62), "{:.2f}",
          "open marker: un-annealed,\ndescriptive only")
    panel(axes[1],
          {"fo_ce": hlm["fo_ce"], "so": hlm["so"], "fo_cm": hlm["fo_cm"]},
          "no-adapt held-out loss $H_{LM}$ (nats/token)", (3.70, 4.70), "{:.3f}",
          "bars: 95% $t$ interval over\nthree seeds")
    save(fig, os.path.join(HERE, "figc1-trajectory"), formats=("pdf",))


if __name__ == "__main__":
    main()
