# Post hoc analyses

These scripts produced the post hoc numbers of the paper and the matching files in `results/`.
One folder per results folder. All are CPU only. Run them from the repository root; default paths
are relative to it, and outputs go to `outputs/analysis/<folder>/` unless a script prints to stdout.

Most inputs are per-document evaluation reads or saved probe outputs. These are not included in the
repository. For each script, the list below names the files and fields it needs. The `outputs/...`
defaults are the locations the released configs write to.

numpy is a project dependency. Some scripts need packages that are not; add them with `--with`:
- pandas for `text-repeat-exposure`;
- jax for `performative_correspondence.py`;
- sympy for `key_pair_symbolic.py` and `so_residual_checks.py`.

Scripts that accept `--process-name` set the process title when `setproctitle` is installed.

Some scripts print tables to stdout. In that case the released CSV holds the same numbers, one row
per printed line.

## text-equal-token-check/

Paper: §3.4 and App. C, "Equal tokens: two post hoc controls". This is a post hoc re-reading of the
original study's SO and FO-CE frontiers at equal tokens (both 1200 steps), for mini-m and mini-s,
seeds 0–2. Every script prints to stdout.

| Script | Computes | Released file(s), paper numbers |
|---|---|---|
| `decomposition.py` | Per arm and η multiplier (mini-m): adapted loss L, `fP_focal`, `fP_nonfocal` and M = focal − non-focal. | `decomposition.csv`. SO η×1→×2: non-focal probe loss 2.246 → 2.093, focal −.014. FO-CE η×3→×4: focal +.051. |
| `frontier_exchange_rate.py` | Per model and seed: loss floors and floor gap (document SE), every η point, the exchange rate λ*, and FO-CE's best M and best `fP_nonfocal` at SO's loss. | `frontier_points.csv`, `frontier_summary.csv`, `matched_loss_comparison.csv`. SO's `fP_nonfocal` is lower by .109–.195 at η×1.5 and by +.097/+.010/+.040 at η×1. |
| `robustness.py` | R_B for b ∈ {0, .005, .01, .02} under three rules: piecewise-linear, quadratic in log η, and best of the CE and inverse-σ curves. Also a document bootstrap (SD, 5th percentile). | `robustness.csv` |
| `per_document.py` | Per-document distribution of the M difference at b = .01. | `per_document.csv` |
| `cover.py` | R_B(.01); cover with +.01 loss for FO-CE; SO's M at its floor +.01 minus FO-CE's largest M; own-floor M difference; floor gap. | `cover.csv`. The .056–.085 margin; floor gap .037/.052/.050. |
| `symmetric_frontier.py` | Two-sided budgeted frontier with a CAL/TEST split of the loss documents. Per-seed R, 300 random splits, seed mean, one-sided 90% t bounds. | `symmetric_frontier_per_seed.csv`, `symmetric_frontier_aggregate.csv`. mini-m: +.108 (lower bound +.080) at b = .01; +.016 at b = 0. |

**Inputs.** Per-document reads `<dir>/<arm>/s<seed>/g0_read_<name>.npz`, with arms `e2e2` (SO) and
`e2e1-ce` (FO-CE).
- `<name>` is `ce` for η×1 and `etaNNN` (100 × multiplier) otherwise. `robustness.py` also reads
  `isig4` and `isig4_etaNNN`.
- Fields: `L_adapt_sum`, `L_adapt_count`, `M_doc`, `fP_focal`, `fP_nonfocal`.
- mini-m reads are written by `configs/g0`, `configs/g0b` and `configs/g0c`. They are searched in
  that order (`--mini-m-reads`), and the first directory that holds a read is used.
- mini-s reads are written by `configs/soq/fs.toml` (`--mini-s-reads`).

```sh
uv run --locked python scripts/analysis/text-equal-token-check/frontier_exchange_rate.py \
    [--mini-m-reads outputs/mini/matrix/g0/mini-m outputs/mini/matrix/g0b/mini-m outputs/mini/matrix/g0c/mini-m] \
    [--mini-s-reads outputs/soq/fs/mini-s]          # likewise the other five scripts
```

## text-repeat-exposure/

Paper: §3.4. Cross-segment repeats are 1.61% of DCLM targets (`w_tok` ≈ .016).

`exposure_geometry.py` is token-only. It covers three sets of 1024 documents each, every one 8192
tokens after BOS:
- **DCLM-end**: the P2-1 endpoint documents;
- **code-file**: one source file per document;
- **code-repo**: one repository's files per document, sorted by path.

The layout is segment-blocked with S = 2048. For each set, the script reports:
- the shares of non-repeat, within-segment and cross-segment 4-gram repeats;
- barrier probes (`tttmem.p2.probe.find_probes`, barriers 2/4/6) and their strata and distances;
- the shares with whitespace/punctuation targets removed.

Released file: `exposure_geometry.csv`. Its variants map to flags as follows:
- `all_files`: the default run;
- `programming_languages`: `--prog-only`;
- `programming_languages_min12000_chars`: `--prog-only --min-chars 12000`.

**Inputs.**
- `data/p2data/eval.bin`: the P2 evaluation pool, uint32 [n, 8193].
- `data/p2data/excluded_vocab.npy`.
- `eval_rows` of a P2-1 `eval_meta.npz` (default `outputs/p21/e2e2/attempt0/eval_meta.npz`).
- From the Hugging Face hub: `codeparrot/github-code-clean`, shard 0, and the Llama-3 tokenizer at
  the revision pinned in `tttmem.p2.data`.

```sh
RAYON_NUM_THREADS=1 uv run --locked --with pandas python scripts/analysis/text-repeat-exposure/exposure_geometry.py \
    [--prog-only] [--min-chars 12000]               # peak 2.5-2.8 GB with one tokenizer thread
```

## text-copyability-distance/

Paper: §5 (MINI-D). The deficit is smaller on copyable strata, K1 = −.082 [−.092, −.071] (mini-m),
and grows with distance, K2 = +.114. Both presets are labelled DISTANCE-LIKE.

`copyability_distance_breakdown.py` splits the per-probe memory m = fP_focal − fP_nonfocal of SO and
FO-CE at step 1200 by probe level:
- exposure stratum;
- distance;
- 8-gram match;
- chunk;
- domain.

For each level it reports ρ = ΔM / M_SO, seed means and one-sided 90% t intervals. It then computes
the K1 and K2 contrasts and the label, after a validity gate.

Released file: `breakdown.json` (identical to the script's JSON).

**Inputs.** The MINI matrix reads under `<root>/g40` … `g45` (default root `outputs/mini/matrix`;
configs `configs/mini/matrix-g40.toml` … `g45`). The files are
`runs/<preset>/<arm>/s<seed>/eval_step1200.npz`:
- `tok_fP_focal`, `tok_fP_nonfocal`, `doc_fP_focal`, `doc_fP_nonfocal`, `eval_sha1`, `eval_domain`;
- the `eval_meta.npz` beside it: `probe_offset`, `probe_stratum`, `probe_dist`, `probe_pos`,
  `probe_match8`.

```sh
uv run --locked python scripts/analysis/text-copyability-distance/copyability_distance_breakdown.py \
    [outputs/mini/matrix] [--out-json F] [--out-md F]
```

## text-param-counts/

Paper: §2.
- mini-m: 25.6M non-embedding parameters, 91.3M in total.
- mini-s: 5.1M non-embedding parameters.

`parameter_counts.py` builds each preset with `tttmem.p2.model.build_model` and counts it with
`param_count`. It writes `parameter_counts.csv` (identical to the released file) and needs no data.
Peak memory is about 1.5 GB.

```sh
uv run --locked python scripts/analysis/text-param-counts/parameter_counts.py
```

## theory-checks/

Paper: App. A. The first four scripts are self-contained.

| Script | Computes | Released file, paper numbers |
|---|---|---|
| `performative_correspondence.py` | float64 JAX surrogate of the echo toy. FO meta-gradient = RGD direction; SO = gradient of the performative risk. Finite-difference checks C1–C6 per inner rule and clip setting. | `performative_correspondence_check.json` (App. A.1: agreement to 5×10⁻⁸) |
| `key_pair_symbolic.py` | SymPy identities of the key–pair model (P3.1 … P8.2), each checked symbolically and at 40 random states. `numeric_max_abs` depends on Python's string-hash seed; `symbolic_zero` and `passed` do not. | `key_pair_symbolic_check.json` (App. A.2) |
| `so_residual_checks.py` | Checks S7–S11 of the SO fixed point. Includes the residual 1 − p* at λbη ∈ {1.01, 1.1, 2, 10, 40}. | `so_residual_roots.csv` (App. A.2: roots .00928, .08495, .44969, .41009, .14706; from the S8 line) |
| `reduced_model.py`, `reduced_model_predictions.py` | The reduced model: calibration, reproduction table, discretisation check, predictions, sensitivity over 27 conventions, phase diagram. | `reduced_model_predictions.csv` (= `predictions-base.csv`), `reduced_model_prediction_sensitivity.csv` (= `predictions-sensitivity-summary.csv`) (App. A.4) |
| `probe_summary.py` | Per-checkpoint probe summary: bootstrap intervals of every per-sequence quantity, registered counts Q1–Q11, identity maxima, set-level medians. Q8: D_SO = D_FO under the linear loss at the kv positions, relative difference ≤ 1e-9. | No released file. App. A.3: "checked 9/9" (Q8, set L). Writes the summary JSON read by the next two scripts. |
| `reduced_model_attribution.py` | The reduced model's linear-read attribution test on the SO final checkpoints, seeds 1–9. | `reduced_model_attribution_test.csv` (App. A.4: 0/9; errors .526–.950) |
| `performative_replay.py` | Relative difference between the toy's own FO/SO training loss and the replayed loss, and the query-logit difference, per checkpoint. | `performative_replay_agreement.json` (App. A.1: to 4×10⁻¹⁶) |
| `write_factorisation.py` (+ `write_factorisation_bootstrap.py`) | Registered write-factorisation verdicts after identity-check validity, with seed-family / shared-sequence bootstraps. | `write_factorisation_results.json`, `write_factorisation_units.csv` (the `--scope b` outputs, App. A.3) |
| `write_factorisation_kappa.py` | Per-key factor κ per unit and group (stdout). | `write_factorisation_kappa.csv`: κ .45–1.41 for FO, .62–1.03 for bound SO (App. A.3). Groups: R = FO responder, NR = FO non-responder, SO = SO bound, SOu = SO unbound, dev = development unit. |

**Inputs of the last five rows.** These are probe outputs of the toy checkpoints, not included in
the repository. The probe that produced them is not part of this release. Each script's docstring
lists the fields. In brief:
- `probe_summary.py` reads the per-checkpoint probe JSONs (one per registered checkpoint) with
  `label`, `arm`, `n_eval`, `etas`, `checks` and `per_seq`.
- `write_factorisation.py` reads one JSON per unit with `per_seq`, `n_eval`, `checks`, `rho`,
  `etas`, `wall_s`, `peak_rss_gb` and `ckpt_sha256`.
- `write_factorisation_kappa.py` reads `<label>.npz` with `w_resid`, `w_n_ce_T` and `w_n_lin_T`.
- `reduced_model_attribution.py` reads a per-checkpoint summary JSON whose `checkpoints` entries have
  `i_Mkv_exact` and `i_Mkv_lin`.
- `performative_replay.py` reads:
  - the per-checkpoint probe JSONs, with `ckpt`, `arm` and `checks.gamma1_fo` / `gamma1_so` /
    `logits_vs_eval`;
  - three compatibility-gate JSONs with `G3_checks`;
  - the summary JSON.

```sh
uv run --locked --with jax python scripts/analysis/theory-checks/performative_correspondence.py
uv run --locked --with sympy python scripts/analysis/theory-checks/key_pair_symbolic.py
uv run --locked --with sympy python scripts/analysis/theory-checks/so_residual_checks.py
uv run --locked python scripts/analysis/theory-checks/reduced_model_predictions.py   # about 8 min
uv run --locked python scripts/analysis/theory-checks/write_factorisation.py <probe dir> [<results dir>] --scope b
uv run --locked python scripts/analysis/theory-checks/write_factorisation_kappa.py <probe dir> > kappa.txt
uv run --locked python scripts/analysis/theory-checks/probe_summary.py <probe dir> [<results dir>]   # writes probe_summary.json
uv run --locked python scripts/analysis/theory-checks/reduced_model_attribution.py <summary json>
uv run --locked python scripts/analysis/theory-checks/performative_replay.py <summary json> <probe dir> <gate dir>
```

## toy-chronology/

Paper: App. B. The write residual separates failing from correct queries at chance (AUC .47–.56).

`residual_failing_key_auc.py` uses six FO-CE (`e2e1emb`) final checkpoints of the fixed-η runs:
η 40 with seeds 0–3, and η 160 with seeds 0 and 2. For each it reports:
- the failing and correct query counts;
- the median CE residual 1 − p_y at the chunk-0 write position, for each group;
- the AUC of residual(correct) > residual(failing), computed by ranks and checked against a pairwise
  count.

Released file: `residual_failing_key_auc.csv`.

**Inputs.** Per-key probe outputs, not included in the repository:
- `<label>.json` with `ckpt`, `eta_base` and `n_eval`;
- `<label>.npz` with `nat_acc` and `w_resid`.

```sh
uv run --locked python scripts/analysis/toy-chronology/residual_failing_key_auc.py <probe dir>
```
