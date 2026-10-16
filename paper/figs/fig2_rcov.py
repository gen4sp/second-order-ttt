"""Fig. 2: cover residual R_cov per arm and seed, both waves where available (mini-m).

Inputs (repository-relative):
  results/text-crossfit-audit/crossfit.json   original study, seeds 0-2, cross-fit
                                              (versions/all/crossfit: aggregate)
  results/text-replication/analysis.json      replication, seeds 3-5, cross-fit
                                              (versions/all/crossfit: aggregate, labels)

Each row is one arm in one wave. Small markers are single seeds; the large marker is the
three-seed mean and its bar the pair of one-sided 90% t bounds [LB90, UB90] (n = 3, t = 1.8856),
i.e. a two-sided 80% interval. The numbers themselves are in Table 1, so the figure carries no
numeric column; the script applies the registered rule to R_cov, checks it against the
replication's stored labels, and prints means, bounds, labels and per-seed values. The dashed
line is ROPE_M = .04: an arm covers SO's operating point only if UB90 <= .04; it does not cover
if LB90 > .04. Measured values only.
Run from the repository root:
  uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/fig2_rcov.py
"""
import argparse
import json
import os

from figstyle import BASELINE, PALETTE, TEXT, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, os.pardir, os.pardir, "results")
ORIG = os.path.join(RESULTS, "text-crossfit-audit", "crossfit.json")
REPL = os.path.join(RESULTS, "text-replication", "analysis.json")
ROPE = 0.04
FS = 6.0

GREY = "#6E6E6E"
COL = {"fo-ce": GREY, "wsd-1200": GREY, "fo-cm": PALETTE["orange"], "wsd-3469": PALETTE["orange"],
       "mini-p": PALETTE["green"]}
# (arm, wave, row label); top to bottom.
ROWS = [
    ("fo-ce", "orig", "FO-CE, 1200 steps"),
    ("fo-ce", "repl", "FO-CE, 1200 steps"),
    ("wsd-1200", "repl", "FO-WSD, 1200 steps"),
    ("fo-cm", "orig", "FO-CM, 3469 steps"),
    ("fo-cm", "repl", "FO-CM, 3469 steps"),
    ("wsd-3469", "repl", "FO-WSD, 3469 steps"),
    ("mini-p", "repl", "SO-2/16, 2793 steps"),
]
WAVE = {"orig": "original", "repl": "replication"}


def rule(a):
    """Registered cover label from R_cov alone (COVERS would also need UB90(Delta_fn) <= .04)."""
    if a["lb90"] > ROPE:
        return "NOT-COVERED (large)" if a["mean"] >= 0.08 else "NOT-COVERED"
    if a["ub90"] <= ROPE:
        return "COVERS-M-ONLY"
    return "UNDERPOWERED"


def load():
    orig = json.load(open(ORIG))["versions"]["all"]["crossfit"]["aggregate"]
    repl = json.load(open(REPL))["versions"]["all"]["crossfit"]
    return {"orig": orig, "repl": repl["aggregate"]}, repl["aggregate"]["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", ""))
    ap.parse_args()
    use_style()
    agg, labels = load()
    fig, ax = figure(width=TEXT, ratio=0.30)
    n = len(ROWS)
    ys = [n - 1 - i for i in range(n)]
    ax.axvline(0.0, color=BASELINE, linewidth=0.5, zorder=1)
    ax.axvline(ROPE, color=BASELINE, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.text(ROPE + 0.003, n - 0.45, "ROPE .04", ha="left", va="center", fontsize=FS, color=GREY)
    ticklabels = []
    for (arm, wave, name), y in zip(ROWS, ys):
        a = agg[wave][f"R_cov_{arm}"]
        filled = wave == "repl"
        c = COL[arm]
        ax.plot([a["lb90"], a["ub90"]], [y, y], color=c, linewidth=1.4, zorder=3,
                solid_capstyle="butt")
        ax.plot(a["mean"], y, marker="o", markersize=5.0, color=c, zorder=4,
                markerfacecolor=c if filled else "white", markeredgewidth=1.1, linestyle="none")
        for v in a["per_seed"]:
            ax.plot(v, y + 0.28, marker="|", markersize=4.0, color=c, markeredgewidth=0.9,
                    linestyle="none", zorder=4)
        lab = rule(a)
        if wave == "repl":
            assert lab == labels[arm], (arm, lab, labels[arm])
        print(f"{name} ({WAVE[wave]}): {a['mean']:.4f} [{a['lb90']:.4f}, {a['ub90']:.4f}] {lab}; "
              "seeds " + " / ".join(f"{v:.4f}" for v in a["per_seed"]))
        ticklabels.append(f"{name} ({WAVE[wave]})")
    ax.set_yticks(ys)
    ax.set_yticklabels(ticklabels, fontsize=6.0)
    ax.set_ylabel("arm (wave)", fontsize=7)
    ax.set_xlim(-0.012, 0.27)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xticks([0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
    ax.set_xlabel(r"cover residual $R_{\mathrm{cov}}$ (nats): SO's $M^*$ at its operating point "
                  r"minus the arm's $M^*$", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    save(fig, os.path.join(HERE, "fig2-rcov"), formats=("pdf",))


if __name__ == "__main__":
    main()
