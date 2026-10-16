# Figures

Each figure in `paper.tex` is built by one script in this folder. A script reads its inputs from
`results/` and writes its PDF beside itself. Figures are built at their final printed width
(5.5 in, `TEXT` in `figstyle.py`; 3.25 in, `COLUMN`, for Fig. 5) and are not rescaled in LaTeX.
`figstyle.py` is the shared style module; its `save()` prints a figure audit, which reads `clean`
for every figure.

Run from the repository root, for example:

```
uv run --isolated --no-project --with matplotlib --with numpy python paper/figs/fig1_frontier.py
```

| Figure | Script | Input files | Output |
|---|---|---|---|
| Fig. 1 | `fig1_frontier.py` | `results/text-original-study/frontier_seed_means.csv`, `results/text-original-study/analysis.json` | `fig1-frontier.pdf` |
| Fig. 2 | `fig2_rcov.py` | `results/text-crossfit-audit/crossfit.json`, `results/text-replication/analysis.json` | `fig2-rcov.pdf` |
| Fig. 3 | `chr_figure.py` | `results/toy-chronology/series.csv`, `results/toy-chronology/events.csv` | `chr-chronology.pdf` |
| Fig. 4 (App. C) | `figc1_trajectory.py` | `results/text-original-study/analysis.json` | `figc1-trajectory.pdf` |
| Fig. 5 (App. C) | `figc2_minis_frontier.py` | `results/text-mini-s-frontier/analysis.json` | `figc2-minis-frontier.pdf` |

`fig2_rcov.py` also prints the cover residuals, bounds and labels that Table 1 reports.

`tab1_floors.py` is a table helper, not a figure. It prints the adapted CAL floors and no-adapt
losses of Table 1 for both waves, from `results/text-original-study/analysis.json` and
`results/text-replication/floors_hlm_per_seed.csv`. It checks them against
`results/text-replication/analysis.json` and against the rounded table values. It needs only numpy:

```
uv run --isolated --no-project --with numpy python paper/figs/tab1_floors.py
```

The PDFs are written without creator, producer or date metadata, so rerunning a script
reproduces the same file.
