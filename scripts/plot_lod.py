# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "setproctitle"]
# ///
"""Render registered LOD outputs in a separate plotting environment.

Usage: uv run --isolated --no-project scripts/plot_lod.py outputs_lod --out outputs_lod/analysis
The source receipts and analyzer tables must already exist. This never imports torch.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np
from lod_figstyle import BASELINE, PALETTE, TEXT, figure, figure_grid, panel_labels, save, use_style


def render(path, out):
    use_style()
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((path / "manifest.json").read_text())
    scores = json.loads((out / "lod_scores.json").read_text())
    cells = []
    for cell in manifest["cells"]:
        name = f"{cell['group']}_{cell['arm']}_N{cell['load']}_s{cell['seed']}.json"
        if (path / name).exists():
            row = json.loads((path / name).read_text())
            if row["status"] in ("COMPLETE", "DIVERGED"):
                cells.append(row)
    colors = dict(lin=PALETTE["blue"], so=PALETTE["orange"], aice=PALETTE["green"])
    captions = {}
    fig, ax = figure(width=TEXT)
    for arm, color in colors.items():
        cohort = [c for c in cells if c["arm"] == arm and c["layout"] == "main"]
        ns = sorted({c["load"] for c in cohort})
        medians = []
        for n in ns:
            values = [c["composite"]["A"] for c in cohort if c["load"] == n]
            medians.append(np.median(values))
            ax.scatter([n] * len(values), values, color=color, s=8, alpha=.35)
        ax.plot(ns, medians, "o-", color=color, label=arm, ms=3)
    reference = Path(__file__).resolve().parents[1] / "configs/toy/lod/capacity-reference.txt"
    refs = [line.split() for line in reference.read_text().splitlines() if line.split() and line.split()[0].isdigit()]
    ax.plot([int(r[0]) for r in refs], [float(r[1].split('/')[0]) for r in refs], "--", color=BASELINE,
            label="stylized centred Hebbian")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Written pairs (N)")
    ax.set_ylabel("Streaming probe accuracy")
    ax.set_ylim(0,1.03)
    ax.legend(frameon=False)
    save(fig, str(out / "lod_load"))
    captions["lod_load"] = "Load accuracy under the registered streaming schedule. Points show individual seeds; lines show medians. The dashed synthetic reference is stylized, not a prediction for this model."
    fig, axes = figure_grid(1,2,width=TEXT,ratio=.45,sharey=True)
    for ax,h in zip(axes,("H1a","H3b")):
        rows = [(int(n),r) for n,r in scores[h].items() if r.get("mean") is not None]
        if rows:
            ns=np.array([n for n,r in rows]); means=np.array([r["mean"] for n,r in rows])
            err=np.array([[r["mean"]-r["LB"] for n,r in rows],[r["UB"]-r["mean"] for n,r in rows]])
            ax.errorbar(ns,means,yerr=err,fmt="o-",ms=3,capsize=2,color=colors["lin" if h=="H1a" else "aice"])
        ax.axhline(-.05,color=BASELINE,ls="--",lw=.7)
        ax.set_xscale("log",base=2)
        ax.set_xlabel("Written pairs (N)")
        ax.set_ylabel("Accuracy difference" if h=="H1a" else "")
    panel_labels(axes)
    save(fig,str(out/"lod_differences"))
    # e(N) is AICE-lin; H3b above is AICE-SO. The e table/figure is written separately.
    fig, ax = figure(width=TEXT)
    ns, means, all_values = [], [], []
    for n in sorted({c["load"] for c in cells if c["arm"]=="aice"}):
        ap={c["seed"]:c for c in cells if c["arm"]=="aice" and c["load"]==n}
        lp={c["seed"]:c for c in cells if c["arm"]=="lin" and c["group"] in ("core","smoke") and c["load"]==n}
        vals=[ap[s]["composite"]["A"]-lp[s]["composite"]["A"] for s in ap.keys() & lp.keys()]
        if vals:
            ns.append(n); means.append(np.mean(vals)); all_values.append(vals)
            ax.scatter([n]*len(vals),vals,color=colors["aice"],s=8,alpha=.4)
    ax.plot(ns,means,"o-",color=colors["aice"],ms=3)
    for n, row in scores["H3_difference"].items():
        if row["mean"] is not None:
            ax.errorbar(int(n), row["mean"], yerr=[[row["mean"] - row["LB"]], [row["UB"] - row["mean"]]],
                        fmt="none", color=colors["aice"], capsize=2)
    ax.set_xscale("log",base=2); ax.set_xlabel("Written pairs (N)"); ax.set_ylabel("AICE minus lin accuracy")
    ax.axhline(0,color=BASELINE,lw=.7)
    save(fig,str(out/"lod_aice_lin"))
    captions["lod_differences"]="Paired contrasts lin–SO (a) and AICE–SO (b). Bars are registered 90% t intervals across the complete pairs at each load; sample counts are in lod_verdicts.csv. Gates apply to the verdicts."
    captions["lod_aice_lin"]="AICE minus lin streaming accuracy. Points are paired seed differences; the line is their mean. Bars show registered 90% t intervals across complete seed pairs; no curve fit or extrapolation is made."
    fig, axes=figure_grid(1,2,width=TEXT,ratio=.45)
    for arm,color in colors.items():
        high=[c for c in cells if c["arm"]==arm and c["load"]==128 and c["layout"]=="main"]
        if not high:
            continue
        metrics=("A","A_freeze","A_snap")
        for c in high:
            axes[0].plot(range(3),[c["composite"][k] for k in metrics],color=color,alpha=.25,lw=.6)
        axes[0].plot(range(3),[np.mean([c["composite"][k] for c in high]) for k in metrics],"o-",color=color,label=arm,ms=3)
        profiles=[c.get("diagnostic",{}).get("profile",[]) for c in high]
        chunks=sorted({r["chunk"] for ps in profiles for r in ps if r["type"]=="write"})
        axes[1].plot(chunks,[np.mean([r["kv_drive"] for ps in profiles for r in ps if r["chunk"]==ch and r["type"]=="write"]) for ch in chunks],"o-",color=color,ms=3)
    axes[0].set_xticks(range(3),["A","freeze","snapshot"])
    axes[0].set_xlabel("Evaluation phase rule"); axes[0].set_ylabel("Probe accuracy")
    axes[1].set_xlabel("Write chunk index"); axes[1].set_ylabel("Pre-update kv drive norm")
    panel_labels(axes)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False)
    save(fig,str(out/"lod_phases"))
    captions["lod_phases"]="Phase-update effects and the per-write-chunk kv drive at N=128. Faint lines show seed-level endpoint effects; bold lines show means. This decomposition alone does not identify interference."
    fig,ax=figure(width=TEXT)
    for arm,color in colors.items():
        rows=[r for r in scores["H5"]["error_share"] if r["arm"]==arm and r["load"]>=64 and r["above_median_error_share"] is not None]
        ns=sorted({r["load"] for r in rows})
        for n in ns:
            vals=[r["above_median_error_share"] for r in rows if r["load"]==n]
            ax.scatter([n]*len(vals),vals,color=color,s=8,alpha=.4)
        ax.plot(ns,[np.mean([r["above_median_error_share"] for r in rows if r["load"]==n]) for n in ns],"o-",color=color,label=arm,ms=3)
    ax.set_xlabel("Written pairs (N)"); ax.set_ylabel("Errors predicting frequent values (share)")
    ax.set_ylim(0,1.03); ax.legend(frameon=False)
    save(fig,str(out/"lod_value_errors"))
    captions["lod_value_errors"]="Error share predicting an above-median-count value, based on the full per-sequence write histogram. Points show individual seeds; lines show means. Cells with no errors have an undefined share and are omitted."
    fig, ax = figure(width=TEXT)
    rank_rows = scores["H5"].get("error_rank_share", [])
    for arm, color in colors.items():
        distributions = [r["shares"] for r in rank_rows if r["arm"] == arm and r["load"] >= 64 and r["shares"] is not None]
        if distributions:
            for values in distributions:
                ax.plot(range(5), values, color=color, alpha=.25, lw=.5)
            ax.plot(range(5), np.mean(distributions, axis=0), "o-", color=color, label=arm, ms=3)
    ax.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "non-value"])
    ax.set_xlabel("Predicted value count rank (quartile)")
    ax.set_ylabel("Errors in rank bin (share)")
    ax.set_ylim(0, 1.03)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    save(fig, str(out / "lod_error_rank"))
    captions["lod_error_rank"] = "Error shares by predicted-value count rank at N≥64. Ties receive midranks over the 16-class full-write histogram. Faint lines show cells; bold lines show cell means. Non-value predictions form a separate bin."
    fig,axes=figure_grid(1,2,width=TEXT,ratio=.45,sharey=True)
    for arm,color in colors.items():
        for n in sorted({c["load"] for c in cells if c["arm"]==arm and c["layout"]=="main"}):
            cohort=[c for c in cells if c["arm"]==arm and c["load"]==n and c["layout"]=="main"]
            times=sorted({r["step"] for c in cohort for r in c.get("logs",[])})
            for ax,metric in zip(axes,("acquired","t50")):
                fraction=[np.mean([c[metric] is not None and c[metric]<=step for c in cohort]) for step in times]
                ax.step(times,fraction,where="post",color=color,label=f"{arm}, N={n}",alpha=.8)
    for ax in axes:
        ax.set_xlabel("Completed training steps"); ax.set_ylabel("Seeds reaching threshold (share)")
        ax.set_ylim(0,1.03)
    panel_labels(axes)
    save(fig,str(out/"lod_acquisition"))
    captions["lod_acquisition"]="Empirical acquisition (a, threshold .90) and t50 (b, threshold .50) curves. Never-reaching cells remain in the denominator and are censored at 1500. Each series is an arm/load cohort."
    (out/"lod_captions.json").write_text(json.dumps(captions,indent=2)+"\n")
    print("LOD_FIGURES",out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--process-name",default=os.environ.get("PROCESS_NAME","TTM-lod-75-figures"))
    args=parser.parse_args()
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    render(args.input,args.out or args.input/"analysis")


if __name__=="__main__":
    main()
