# results/

Numeric outputs behind the claims of *What Second-Order Meta-Gradients Buy in TTT-E2E: Probe
Memory at a Language-Modelling Price*. There is one folder per experiment series of the paper's
experiment index (App. G). `index.csv` lists every index row in the paper's order with the
paper's wording and the folder that holds its data.

The folders contain summaries, per-seed and per-cell tables and analysis outputs. They do not contain
model checkpoints, logs, figures or the raw per-document read arrays, which are large.

## Formats

- JSON files are UTF-8. Some text analyses contain `NaN` for undefined values (for example a
  bound over fewer than two seeds). Python's `json` reads `NaN`; strict parsers need an allow-NaN
  option.
- CSV files have one header row. The exception is two pivot tables in `toy-mechanism-probes/`, which
  have a three-row header. Booleans are written `True`/`False`, and an empty cell means no value.
- Values are stored at full precision; the paper rounds them. Losses are in nats per token.
- Keys and column names are the analysis code's own. Where a key is a registered hypothesis or
  gate label (for example `H1`, `P7.1`, `R2`, `V1`, `F1`, `G2'`), its entry holds the counts, the
  statistic and the verdict for that rule.
- Verdict strings are the registered labels, copied verbatim (for example `NOT-COVERED (large)`,
  `LATE-FAILS`, `NOT QUALIFIED`). A `registered` key or value marks a rule that was fixed before the
  data. `posthoc`, `exploratory`, `descriptive` and `smoke` mark the other kinds of output.

## Uncertainty conventions (paper §2)

- **Three-seed text statistics.** Bounds are one-sided 90 % t bounds over n = 3 training seeds,
  with t = 1.8856 (stored as `t90` or `t_critical`; the paper prints 1.886). The keys are
  `lb90`/`ub90` or `lb`/`ub`, printed in the paper as [LB90, UB90]. Together they form a two-sided
  80 % interval. The primary rule is one-sided at α = .10, and secondary labels are not corrected
  for multiplicity. `sd` is the sample standard deviation over seeds.
- **Cross-fit.** η is selected on one fixed half of the probe-memory documents and M is reported on
  the other half, then the halves swap. Fold results are weighted by report-fold document count.
  The replication's primary analysis is cross-fit. In the original study, cross-fit is a
  retrospective audit (`text-crossfit-audit/`).
- **Toy runs.** Results are counts of runs (k/n acquired or retained). The tests are one-sided
  exact McNemar against the paired anchor, Holm-adjusted within the family named in the file. `ci95`
  is a 95 % interval as labelled in the file.
- **Load extension (`toy-load-extension/`).** Two-sided 90 % t interval over 12 seeds (t = 1.7959, 11 df).
- **125M (`125m-*`).** 90 % URL-domain cluster bootstrap over documents (`lo`/`hi`, `*_lo90_doc_boot`).
  These intervals do not include training-seed variance.
- **MINI-E cotangent ratio.** 90 % document bootstrap.
- **WM2 (`wm2/`).** AUROC over the three cup places, where chance is .5. Bootstrap intervals over
  seeds are labelled where quoted.

## Arm codes

The paper's names are on the right.

| Code in files | Meaning |
|---|---|
| `so`, `e2e2` | SO: second-order meta-gradient (TTT-E2E as released) |
| `fo_ce`, `fo-ce`, `e2e1-ce`, `e2e1` | FO-CE: first-order rule, CE inner loss, equal steps (1200 text steps) |
| `fo_cm`, `fo-cm` | FO-CM: FO-CE trained 3469 steps, equal GPU-seconds |
| `wsd-1200`, `wsd-3469` | FO-CE with a warmup-stable-decay schedule, 1200 / 3469 steps |
| `mini-p` | SO-2/16: SO on 2 of 16 fast-weight rows, 2793 steps at equal GPU-seconds |
| `e2e1-rp`, `e2e1-nc` | text residual floor (RP) and its direction control (NC); `-mix` = mixed DCLM + code corpus |
| `mini-m`, `mini-s` | text model presets, 25.6M and 5.1M non-embedding parameters (App. F) |
| `emb` suffix | toy arm with a frozen key code (FO+emb, SO+emb in Table 3) |
| `e2e1k`, `e2e1e` | toy FO with key-code or error-code variants (R4-1) |
| `kvm` | outer-loss mask at kv; `rnd` = same-count mask at adjacent targets; `embP` = control arm of check P |
| `lin` | linear inner loss everywhere; `linkv` at kv only; `linvW` at the next position (site control); `linxkv` everywhere except kv |
| `cnNN` | copy noise: the copied value is correct with probability .NN (q = 1 − .NN) |
| `rf95` | matched-count filler-noise control for `cn95` |
| `edge` | copy-edge attention mask; `kveta`, `kveta10` = separate kv step size |
| `kamin`, `kaminctl`, `kaminlin` | Kamin two-stage arm, its control, and its linear-loss variant |
| `aice` | AICE inner rule; `b9` = address auxiliary loss; `b9f` = filler auxiliary control |
| `w600` | 600 warm-up steps instead of the default |
| `ctrl`, `echo_ctrl` | copy-free control task |
| `jac1`, `jac2`, `jac2so` | Jacobian-SO variants (R1, R2); `swa` = sliding-window attention only, `fa` = full attention |
| `rpfNN`, `ncfNN`, `rp10` | toy cap-free residual floor at d = NN/100, its norm-matched direction control, and the capped floor (App. B) |

Toy seeds are listed in each table. Text seeds 0–2 are the original study and 3–5 the replication.
WM2 seeds 101–105 are the fresh training seeds.

## Folders

Each entry gives the paper section, then each file with its contents and main keys.

### Text: the budget test and its replication

**text-original-study/** — §3.1, Table 1. The original study: SO, FO-CE and FO-CM at equal GPU-seconds, seeds 0–2.
- `analysis.json`: per-seed and aggregate frontier reads.
  - `seeds[]`: per seed, with `so`/`fo_ce`/`fo_cm` → `M` (probe memory), `F`/`N` (focal / non-focal probe loss, M = F − N), `Lt` (TEST loss), `eta` (selected η multiplier) and `clip` (inner clip fraction).
  - Per seed also: `R_cov_cm`/`R_cov_ce`, `dfn_*` (Δfn), `x`, `floor_cal` and `H_LM` per arm, and `L_SO_min`.
  - `agg`: three-seed `mean`, `sd`, `lb90`, `ub90` and `per_seed` for every measure, in-sample.
  - `sens`: the same analysis with the 130 near-duplicate evaluation rows dropped.
  - `label`, `label_sens`: registered verdicts.
- `frontier_seed_means.csv`: seed-mean η frontier per arm (Fig. 1).
  - `arm`, `eta` (multiplier), `L_cal_mean` (adapted CAL loss), `M_mean`.
- `timing_pilot.json`: compute match.
  - `t_so_ms`/`t_fo_ms`: median step times at micro-batch `so_micro_batch`/`fo_micro_batch`.
  - `r`: the SO/FO time ratio; `n_cm` = 3469 FO-CM steps.
  - `pilot/settings[]`: per arm and micro-batch timings.
  - `flop_ratio_counted`: counted-FLOP ratio.

**text-crossfit-audit/** — §2, App. C. Cross-fit re-selection on the original study's reads; no training.
- `crossfit.json`: the audit.
  - `split_rule`: the document-half rule; `t90`; `fold_weighting`.
  - `versions/{all,overlap_drop}/{in_sample,crossfit}/{per_seed,aggregate}`: R_cov, Δfn, x and M per arm.
  - `aggregate/<measure>_<arm>`: `mean`, `sd`, `lb90`, `ub90`, `per_seed`.
  - `selection_optimism`: in-sample minus cross-fit values.
  - `top1`: per seed, arm and η, the probes' top-1 rate and mean rank (focal / non-focal).

**text-replication/** — §3, §3.2, §5, Table 1. The replication on seeds 3–5.
- `analysis.json`: same layout as the cross-fit audit; `primary` = `crossfit`.
  - Arms: `so`, `fo-ce`, `fo-cm`, `wsd-1200`, `wsd-3469`, `mini-p`.
  - Includes `x` (FO-CM − FO-CE), `x_P` (SO-2/16 − FO-CM) and the passkey reads.
- `floors_hlm_per_seed.csv`: per arm and seed.
  - `cal_floor`: the adapted CAL loss at the η-grid minimum.
  - `cal_floor_eta_multiplier`: the η at that minimum.
  - `H_LM`: the no-adapt loss.
- `timing_pilot.json`: timing pilot for the partial-SO arm.
  - `n_p` = 2793 steps; `t_so_ms`, `t_p_ms`.
  - `common_timing`: re-measured SO/FO ratio `r`, its difference from the original study, and `rho_s14` (budget match).

**text-equal-token-check/** — §3.4. Post hoc re-reading of the original study's SO and FO-CE frontiers at equal tokens (mini-m and mini-s).
- `frontier_points.csv`: every η point.
  - `model`, `seed`, `arm`, `eta_multiplier`, `adapted_loss_L`, `M`, `fP_nonfocal`, `fP_focal`.
- `decomposition.csv`: the same quantities at the decomposition points, with `n_docs`.
- `frontier_summary.csv`: per model and seed.
  - `L_floor_SO`, `L_floor_FO`, `floor_gap` with document SE.
  - `lambda_star`: the exchange rate (M per nat).
  - The arg-min η of each arm.
- `matched_loss_comparison.csv`: SO's points against FO's best M within the same loss.
  - `R`: FO's shortfall.
  - The non-focal probe-loss difference.
- `robustness.csv`: R under alternative interpolation rules, with a document-bootstrap SD and 5th percentile.
- `per_document.csv`: the per-document distribution of the M difference.
- `cover.csv`: cover quantities per seed and their mean.
- `symmetric_frontier_per_seed.csv`, `symmetric_frontier_aggregate.csv`: the two-sided frontier.
  - `R` at budget `budget_b`.
  - `lb90_one_sided` and `ub90_one_sided` over seeds.

**text-repeat-exposure/** — §3.4. Exposure of cross-segment repeats.
- `exposure_geometry.csv`: one row per variant (`all_files`, `programming_languages`, …) and evaluation set (DCLM, code).
  - Repeat shares: `share_non_repeat`, `share_within_segment`, `share_cross_segment`.
  - Probe counts.
  - Stratum shares `stratum_E1` … `stratum_E3`.
  - Probe-distance quantiles.

### Text: mechanism diagnostics and levers (mini-m, mini-s)

**text-mini-baseline/** — §3.3, Table 5. From-scratch SO, FO-CE and residual-floor arms at equal steps.
- `analysis_core.json`: the registered core analysis.
  - `preset_stats` and `pooled_stats`: ΔM, D and ρ with bounds.
  - `contrasts`: per preset.
  - `M0_gate` and `preflight_rules`.
  - `H3_point_contrasts`: `C_own`, `C_T`, `C_E`, `C_NC`, `S`.
  - `code_deficit_strata`.
- `analysis.json`: the exploratory corpus comparison (DCLM, code, mix).
  - `metrics[]`: per arm, seed, step and corpus, with `L`, `G` and `M`.
  - Contrasts and `seed_sd_final`.
- `code_eval_summary.json`: the code evaluation set: document counts, language shares and the near-duplicate check.

**text-copyability-distance/** — §5. Copyable versus distance split of the equal-step deficit.
- `breakdown.json`: `presets/{mini-m,mini-s}` holds the deficit by copyability class and probe distance.

**text-write-read-alignment/** — §5. Write–read alignment.
- `analysis.json`: alignment statistics `per_preset` and `labels` (registered verdicts).

**text-floors-eval-read/** — Table 5. Eval-carried read of the residual floors.
- `analysis.json`: the core analysis fields plus `minie`.
  - `minie`: per-preset eval-carried contrasts (C_E) and the starvation class.
- `cotangent_ratio_strata.json`: FO/SO cotangent ratio per preset and stratum (E1 … E3), as a mean with per-seed values.

**text-partial-so-equal-steps/** — Table 5. Partial SO (2/16 rows) at equal steps.
- `partial_so.json`: `partial_so/mini-m` holds φ and the transfer contrast with bounds.

**text-late-so/** — Table 5. 300 late SO steps after FO-CE.
- `analysis.json`: the analysis.
  - `paired[]`: per seed, `M` of the late-SO (`L-SO`), late-FO (`L-FO`) and reference (`L-REF`) arms.
  - `D`, `R`, `psi` (ψ) and `psi_prime`.
  - `rows[]`: M and G by arm, seed and step.
  - `label`.

**text-micro-testbed/** — App. G (background). Reduced-vocabulary micro testbed.
- `analysis.json`: qualification gates `F1` … `F6` and `M0`, with `pairs[]` (per seed `M_SO`, `M_FO`, `delta_M`) and `label`.
- `cotangent_ratio_strata.json`: cotangent ratio by stratum.
- `testbed_summary.json`: corpus counts, probe counts and stratum shares.
  - `tokenizer_sha256` and `dataset_revision` identify the public tokenizer and dataset.

**text-write-gain/** — §5, Table 5. Write-gain split, eval-time η and σ-CE.
- `analysis.json`: `calib` (per arm and seed calibration), `titration` (dose: ΔM, κ, ΔL, `B1`, `B2`) and `decomposition` (shares, label).
- `reads_by_seed.csv`: every read.
  - `arm`, `seed`, `read` (condition), `M`, `L_adapt`, `n_docs`, `n_ho`.

**text-matched-loss-frontier/** — §3.4, Table 5. Matched-loss frontier and the inverse-σ rule.
- `analysis.json`: `frontier` (`rho_match` and per seed) and `inverse_sigma` (κ for each rule, ΔL, shuffled and constant controls).
- `effective_gain_posthoc.json`: post hoc effective-gain moments per arm and seed (focal / non-focal).

**text-inverse-sigma-lift/** — Table 5. Inverse-σ × η frontier lift.
- `analysis.json`: `lambda_lin`, `lambda_quad`, `rho_match_*` (`n`, `mean`, `sd`, `lb`, `ub`, `values`), `lift_by_point`, `per_seed` and `label`.

**text-mini-s-frontier/** — App. C. mini-s matched-loss frontier.
- `analysis.json`: two analyses.
  - `analysis_a`: frontier and `rho_match`.
  - `analysis_b`: fixed-L match on random splits.
  - `comparison`: mini-m versus mini-s, with `verdict`.

**text-echo-template/** — §5. Echo-in-text template validity.
- `analysis.json`: per arm (`e2e1-ce`, `e2e2`), item-level reads, `thresholds` and `verdict`.

**text-injected-binding-pilot/** — App. C. Injected-binding pilot.
- `analysis.json`: `steps` (reads at 0, 300, 600 and 1200), `exposure` (injected-token counts) and `v2_endpoint` (mean, one-sided LB90, margin, verdict).

**text-injected-binding/** — §5. Injected binding, version 3.
- `analysis.json`: registered reads to step 2400.
  - `endpoint_rules`: `C_plus`, `V2_prime` and `verdict`.
  - `steps`: per checkpoint.
- `late_storage_analysis.json`: the same rules applied to the continuation to step 3200.
- `posthoc_analysis.json`: post hoc reads.
  - `checkpoints`.
  - `document_bootstrap`.
  - `v3x`: the late-storage label.
- `posthoc_selected_reads.json`: post hoc read conditions per `step/split/template`.
  - Templates: N (no binding), A, Z, T, T-A.
  - Each holds blocked-segment reads and consistency checks.

**text-param-counts/** — §2. Parameter counts of the text models.
- `parameter_counts.csv`: per preset, total, tied-embedding, non-embedding and fast-weight parameters (millions).

### 125M continuation

**125m-timing/** — App. F. Port and timing profile of the released 125M model.
- `profile_<gpu>.json`: one record per measurement.
  - `kind`: `env`, `model`, `train`, `eval` or `done`.
  - `cfg`: preset, sequence length, arms and micro-batches.
  - The `train` and `eval` records hold step times and memory per arm and micro-batch.

**125m-continuation/** — §5, App. D. 400-step continuation, SO versus FO-CE.
- `trajectory.csv`: per step.
  - `D` = L_FO − L_SO on barrier probes, with `D_lo`/`D_hi`.
  - `dM` (ΔM) and `dHO` (held-out LM difference), each with bounds.
  - Per-arm `L`, `M`, `G` and `HO`.
- `chunk_gap.csv`: D and the write-gain ratio per chunk.
- `endpoint_gap.json`: the step-400 endpoint.
  - `D`, `delta_M`, paired changes (`est`, `lo`, `hi`).
  - `evaluable` and `reason`.
- `copy_contrast_tests.json`: the copy-contrast tests `P7.1`, `P7.1s` and `P7.2` at steps 100, 200 and 400.
- `futility_check.json`: the step-100 futility rule for the linear-rule arms.
- `sizing.json`: power and sizing simulation for the evaluation set.
- `train_steps_so.json`, `train_steps_fo_ce.json`: per-step training log.
  - `loss`, `grad_norm`, `lr`, `chunk_loss`.
  - Inner clip fraction and inner-step norm.

**125m-write-read-swap/** — App. D. Write/read swap.
- `swap_stepNNNN.json`: one file per checkpoint step.
  - `cell_means`: mean losses of every write × read cell.
  - `decomposition`: `G`, `L`, `M`.
  - `geometry_mean_per_layer`, `param_rel_dist` and `validity`.
- `decomposition.csv`: the swap decomposition by step and scope.
  - Columns: `total`, `write`, `read`, `interaction`, Owen values, layer-0 transplant.
  - Each value has `_lo`/`_hi` 90 % cluster-bootstrap bounds.

### Toy: echo and copy-free tasks (App. B, App. F Table 9)

Common columns in toy tables:
- `bound_ever` / `acquired`: any quick evaluation ≥ 0.90.
- `bound_final` / `retained`: full evaluation at step 1500 ≥ 0.90.
- `first_bound_step`, `t_lose`: event steps.
- `query_acc`, `query_ce`: query accuracy and CE.
- `p_kv`, `e_kv_abs`: the kv probability and the kv residual.
- `kv_write_eff`, `Wv_write_eff`: effective write sizes.
- `addr_a`, `addr_margin`: address readouts.
- `eta_W1` … `eta_b2`: learned inner step sizes.
- `first_batch_sha1`: hash of the first training batch, used as a reproducibility check.

**toy-first-comparison/** — §4.1. First FO versus SO comparison, 1–2 seeds.
- `echo.csv`: echo and control query CE and accuracy per arm.
- `icl_mc.csv`: Markov-task late CE per seed, gain over the SWA baseline, share of the Bayes gap and step time.
- `learned_step_sizes.csv`: learned η per arm and task.

**toy-layout-bundle/** — App. G (background). Random-prefix layout, Jacobian-SO and copy-free control.
- `echo_anchor_quick_evals.csv`: anchor quick evaluations per seed and step.
- `echo_references.json`: chance and occupancy reference CE and accuracy.

**toy-six-seed-contrast/** — §4.1. Six-seed contrast.
- `analysis.json`: `cells[]` per run, `Q1` (binding counts, Fisher p), `Q2` (skip and collapse analysis), `Q3` (Jacobian-SO gain) and `refs`.

**toy-random-prefix/** — App. B. Random-prefix binding and η sweep.
- `analysis.json`: `cells[]`, the `R3-A`/`R3-B`/`R3-C` counts and verdicts, and `lr_control`.
- `quick_evals.csv`: per cell and step.
- `references.json`: reference values.

**toy-write-ablation/** — §4.2. Write-position ablation, eval only.
- `conditions.csv`: one row per model × ablation condition.
  - `family`: `fixed`, `frozen`, `scale`, `tensor`, `transplant`, `dswap`.
  - `query_acc`/`query_ce` with SE, and `alt_map_acc`.

**toy-key-address/** — §4.1. Key addressability probes.
- `key_probes.csv`: per model.
  - Address measures (`address_cos`, `address_margin`, …).
  - `content_inf`, `memory_inf`.
  - Linear-probe accuracy at several regularisations.
- `gate_probes.csv`: gate norms and fixed / frozen-kv query reads.
- `fo_failure_classification.csv`: the registered verdict on why FO's kv write carries nothing (App. G row "FO: content 4/6, address 0/6"). One row per FO model, mode echo.
  - `content_inf`, `memory_inf` and `address_dot` are copied from `key_probes.csv` (join on `label`).
  - The registered rule: `meets_gain` is `memory_inf` ≥ .80. `meets_address` is `memory_inf` ≤ .30 and `content_inf` ≥ .80. `meets_content` is `content_inf` ≤ .30, meaning the matched write's direction does not decode its value.
  - A verdict needs at least 5 of the 6 models; otherwise it is Mixed. The counts are content 4/6, address 0/6 and gain 0/6, so the verdict is Mixed.

**toy-write-code-rank/** — App. G (background). Write-code rank diagnostics.
- `models.csv`: per model.
  - Participation ratios `pr_*`, off-diagonal cosines and `r99_*` ranks.
  - Write sizes and kv-only / Wv-only accuracies.

**toy-key-code-arms/** — Table 3. Key, error and embedding arms, 10 seeds each.
- `cells.csv`: per run, status, binding events, query reads and final probes.
- `analysis.json`: `counts`, `paired_table`, `tests` (K, E, A), `verdicts` and `stabiliser`.
- `quick_evals.csv`, `probes.csv`: per cell and step.
- `kv_surprise.csv`: kv probability and error per arm, seed and step.
- `fp32_check.json`: fp32 re-evaluation of 20 cells.

**toy-fixed-eta-retention/** — Table 3. Fixed-η retention, η ∈ {10, 40, 160}.
- `cells.csv`: per arm, η and seed.
  - `acquired`, `retained`, `first_bound`, `first_loss`, `rebind`, `final_acc`.
  - `kv_resid` and step time.
- `scores.json`: the registered checks `A_P2`, `B_P3`, `B_P6` and timing.

**toy-fresh-seed-replication/** — §4.1. Fresh-seed replication, FO 200–215 and SO / noise 200–211.
- `scores.json`: replication claims `R_FO`, `R_H1`, `R_H3`, `R_H4` and `R_TR`, with `family` and `status`.
- `units.csv`: inclusion of each seed per claim and test.

**toy-outer-mask/** — Table 2. Outer-loss mask at kv.
- `summary.json`: `counts` per arm, `contrasts` (McNemar), `labels` and mediator `details`.
- `paired_seeds.csv`: per seed, every arm's endpoints.
- `per_class_ce.csv`: CE by token class.
- `probes.csv`, `quick_evals.csv`: per step.

**toy-linear-inner-loss/** — Table 2, §4.5. Linear inner loss, schedule and copy-free controls.
- `summary.json`: `counts`, `contrasts` and the labelled checks (`LK`, `LM`, `LS`, `LX`, `NC`, `SC`, `D`, `ML3`).
- `paired_seeds.csv`: per seed, all arms.
- `d4_components.csv`: component ratios against the anchor.
- `schedule_control_times.csv`: loss and peak timing.
- `probes.csv`, `quick_evals.csv`: per step.

**toy-so-halves/** — App. B. Second-order halves (E and K) and the mask replication.
- `summary.json`: `counts`, `labels`, `holm` and `ek_manipulation`.
- `ek_share_checkpoints.csv` / `.json`: the share, cosine and relative size of the E and K halves of the SO correction per checkpoint.
- `eta_diagnostic.csv`: learned η per arm and seed.
- `labels.csv`: label → value.
- `paired_seeds.csv`, `probes.csv`, `quick_evals.csv`.

**toy-copy-interventions/** — Table 2. Copy-edge mask, copy noise, Kamin two-stage and kv step size.
- `cells.csv`: per arm and seed, `bound_ever`, `bound_final`, `query_acc` and `t_lose`.
- `summary.json`: `counts`, `GM2`, `GM3`, `reproducibility` and `other_readings`.
- `a6_trend.csv`: Bayes residual by copy-noise level.
- `a6_within_model.csv`: accuracy on clean versus corrupted keys.
- `a9_trajectories.csv`: kv step-size trajectories.
- `d1.csv`: copy-edge on/off value probabilities.
- `gm2_routes.csv`: route checks.
- `kaminlin.csv`: the stage-2 linear-loss Kamin arm.
- `reproducibility.csv`: re-run agreement per seed.
- `rf95_vs_cn95.csv`: filler noise versus copy noise.
- `switch_checks.csv`: Kamin switch checks.

**toy-aice-kamin/** — App. A. AICE rule and Kamin repeat.
- `summary.json`: `p_*` tests, `retention`, `NI_overall`/`NI_late` (non-inferiority), `reassign_sign`, `Kamin`, `holm` and `validity`.
- `cells.csv`: per run.
  - Training metadata and `eta`.
  - Binding events.
  - Nested evaluation, probe and ablation records stored as JSON strings.
- `eta.csv`: per-cell η record (JSON string).
- `acquisition_losses_recoveries.csv`: per cell, first bound, losses, recoveries and final.
- `kamin_acquisition.csv`, `kaminlin2.csv`: Kamin-stage cells (same columns as `cells.csv`).
- `kamin_switch_audit.csv`: switch-point audit.
- `probes.csv`, `quick_trajectories.csv`: per step.
- `reassign_per_seed.csv`, `reassign_seed_differences.csv`: re-assignment recall (new versus old map).
- `ni_seed_differences.csv`: per-seed non-inferiority differences.

**toy-125m-rehearsal/** — App. D. Toy rehearsal of the 125M continuation.
- `summary.json`: `headline_tests`, `holm_family_size`, `so_minus_fo_contrast`, `chosen_lr_mode` and `interpretation`.
- `tests.csv`, `c1.csv` … `c4.csv`: exact tests by contrast family.
  - Columns: `a`, `b`, `n`, `retained_a`, `retained_b`, `p_exact`, `p_holm`, `reject_holm_05`.
- `cells.csv`: per cell (init, rule, lr mode, seed), binding events and final reads.
- `gamma.csv`, `quick_evals.csv`: per-checkpoint diagnostics.
- `retention_loss.csv`, `token_budget.csv`: retention and loss times by rule.
- `step0.csv`: step-0 reproduction of every cell (`match`).

**toy-noise-floor/** — App. B. Fixed-η copy-noise wave.
- `scorecard.json`: `score` holds one entry per registered card (`primary`, `B-W1c3`, `P1 c1-as-registered`, …) with counts, verdict and `reason`.
  - `meta/warnings`.

**toy-chronology/** — §4.3, Fig. 3. Dense chronology and acquisition forks.
- `scores.json`: the registered hypotheses `H1` … `H8` with sensitivity analyses, plus `disposition` and `status`.
- `disposition.csv`: per cell.
  - Eligibility, `t_acq`, `t_loss`, branches.
  - The hypotheses each cell counts toward.
- `events.csv`: per cell and branch.
  - Acquisition, starvation, loss and onset times.
  - `category` and residual at loss.
- `series.csv`: per cell, branch and step.
  - Accuracy (`a`, `a5`, `a6`) and kv residual `r`.
  - Write sizes, clip and per-chunk diagnostics.
- `residual_failing_key_auc.csv`: post hoc and descriptive. Does the write residual predict which queries fail (App. B, "AUC .47–.56")?
  - The units are six FO-CE (`e2e1emb`) final checkpoints of the fixed-η runs in `toy-fixed-eta-retention/`: η 40, seeds 0–3, and η 160, seeds 0 and 2. They are not chronology runs.
  - Each unit has 128 evaluation sequences × 16 queries. A query fails when its argmax is wrong.
  - The residual is the CE residual 1 − p_y at the key's chunk-0 write position.
  - `auc_residual_correct_above_failing` is P(residual of a random correct query > residual of a random failing query), with ties counted ½; .5 means no separation.

**toy-trigger/** — §4.3. Trigger of the loss.
- `stage1_scores.json`: exploratory stage.
  - `H_T3`, `H_T4`, `H_T5` and `units`.
  - `calibration`.
- `stage1_calibration.json`.
- `stage1_events.csv`: per family, seed and event, the failure step and its candidate triggers (residual, write norm, margins, key cosine).
- `stage1_units.csv`: per unit, disposition and details.
- `stage2_scores.json`: registered stage; `H_T3` (outer mask prevents), `H_T4`, `dispositions` and `units`.

**toy-recovery-prevention/** — §4.3, App. B. Recovery dose (`rdo_*`) and prevention floors (`rpe_*`).
- `rdo_scores.json`, `rpe_scores.json`: registered results.
  - `doses`, `headline`, `recovery_KM` (Kaplan–Meier), `R2` and `D_dose_order`.
  - `provenance/effective_config`: the run configuration.
- `rdo_events.csv`, `rpe_events.csv`: per seed and branch.
  - Loss, re-acquisition and censoring.
  - Retention.
- `rdo_series.csv`, `rpe_series.csv`: per step, accuracy, kv residual quantiles and floor diagnostics.
- `rdo_disposition.csv`: per seed and block.
- `units.json`: per-seed units for rdo, rpe and late.
- `late_and_start_factor.json`: late-start comparisons and the start factor.

**toy-fresh-seed-pool/** — §4.3. Combined fresh-seed runs: 40 main runs and 316 forks.
- `pool_design.json`: the run design.
  - `jobs[]`: cell, fork point, horizon and rules.
  - `states[]`: fork states, with detected and event steps.
  - `route_cohorts`, `trg_dispositions`, `h5_cohort`.
  - `provenance/effective_config`.

**toy-markov-cost/** — §4.5. Markov-task cost of the linear loss.
- `cells.csv`: per seed, CE, late CE, η and step time for FO-CE (`e2e1`), FO-linear (`e2e1_lin`) and SO (`e2e2`).
- `summary.json`: `lin_minus_ce`, the `gate` verdict, late-CE differences and `sanity`.

**toy-mechanism-probes/** — §4.5. Mechanism probes: logit split, copy edge, re-assignment.
- `summary.json`: `gate`, with units and the binomial p.
- `gate_units.csv`: per arm and seed, gate status.
- `anchor_eta_split.csv`, `w600_eta_split.csv`: write split into kv, marker and other positions per step.
- `anchor_kv_write_by_alpha.csv`: kv write by α.
- `anchor_trajectories.csv`: per seed and step.
- `step4_residuals.csv`: value probability and residual norm by position class.
  - Three-row header: statistic, class, index.
- `step6_jacobian.csv`: Jacobian cosine and norm ratio.
- `step7_reassign.csv`: re-assignment recall.
- `titration_units.csv`: query accuracy and CE by α.
  - Three-row header: statistic, condition, index.
- `titration_writes.csv`: write sizes by α.

**toy-write-rule-control/** — §4.5. Write-rule control (DiD).
- `analysis.json`: per rule (`ce`, `lin`, `own`).
  - `did` and `gaps` on echo and copy-free control.
  - `table[]`: per cell.
  - `verdict`, `rope` and `level`.

**toy-plain-echo-linear/** — §4.5. Plain echo × {CE, linear}.
- `scores.json`: `scores/core`, `units[]`, `replication_inputs` and `b9_gate`.
- `units.csv`: per seed and arm, disposition and binding.

**toy-address-auxiliary/** — §4.5. Address auxiliary.
- `scores.json`: `verdict`, `qualifier`, `counts`, `rescue` and `specificity` (exact tests, `holm2`), `selectivity` and `units`.
- `units.csv`: per seed and arm.

**toy-load-scaling/** — App. B. Load scaling, N = 16–128 pairs.
- `scores.json`: per hypothesis (`H1a` … `H5`, `Bridge`, `McNemar`), keyed by load.
  - `Q_SO`, `Q_lin` and `Q_plat`.
  - `claims` and `interpretation`.
- `verdicts.csv`: hypothesis, load, label, n, mean, LB, UB.
- `phase.json`, `geometry.json`, `error_share.json`, `error_rank_share.json`: per-cell diagnostics.
- `within_load_associations.json`: within-load correlations.

**toy-load-extension/** — App. B. Budget extension to 3000 steps.
- `scores.json`: `X1` … `X4` and `R`.
  - `X1` holds the paired differences, LB/UB and two-sided 90 % bounds.
  - `gates` and `interpretation`.
- `verdicts.csv`: hypothesis, load, label, n, mean, LB, UB.

**toy-partial-so-screen/** — App. B. Per-token write-gain gate and partial-SO screen.
- `analysis.json`: per arm.
  - `ever_acquired` and `retention`.
  - `gate` (G arms), `somix` (M arms: early, late, p = 1/8, 1/4, 1/2) and `decisions`.
- `summary_line.json`: per arm, n, acquired, retained, step time and peak memory.

**toy-partial-so/** — App. B. Registered partial-SO confirmation.
- `analysis.json`: same layout as the screen.
- `summary.json`: per-cell results and per-arm summary.

### Other domains and theory

**visual-echo/** — App. E. World-model echo and its diagnostics; predecessor of the SCM study.
- `cells.csv`: per cell (rule, edge, η, seed).
  - Accuracy per class (`acc_U`, `acc_S`, `acc_A`, `acc_X`, `acc_N`).
  - Staleness, residuals and activations.
- `diag_d1.csv` … `diag_d4_gain.csv`: ablation, budget, write and gain diagnostics per cell.
- `scores.json`: registered predictions `P*` per η.

**scm/** — §4.5, App. E. Symbolic world stream; a separate study.
- `write_rules.csv`: one row per round and write rule.
  - Retention counts and medians for answer classes A, Au, N and X.
  - `competence_V3` and a short `note`.
  - Arm labels marked "chosen after R3c" were selected after the previous round's result; the paper calls this "post-result".
- `r1_verdicts.json`, `r23_verdicts.json`: per arm, class and η, the registered scores; `CHECK` holds consistency checks.
- `r3d_scores.json`: the last round.
  - `keys[]`: per-row verdicts.
  - `g2prime_c4`: gate.
  - `describe`.
  - `selection`: states the post-result selection.
  - `tally`.

**wm2/** — §4.5, App. E, Table 7. Visual hidden object; a separate study.
- `original_seeds_fresh_episodes_summary.json`, `fresh_seeds_summary.json`: registered reads on fresh episodes.
  - Original training seeds; fresh training seeds 101–105.
  - `readings/rows` hold the AUROCs, `claims` and `verdict`.
- `fresh_seeds_cells.csv`: per arm and seed, `R_N`, `R_X`, `E_N`, `E_X` and `X_old` (AUROCs), plus the claim check.
- `location_probe_summary.json`, `location_probes.csv`: linear location probes per tag, with `acc`, fold SD and a shuffled control.
- `mechanism_summary.json`, `mechanism_cells.csv`: mechanism claims F1–F5 and the per-tag show / lift / write components.
- `show_write_summary.json`, `show_write_cells.csv`: show-write claims Q1–Q4 and per-tag knock-out reads.
- `mixes_summary.json`, `mixes_beyond.csv`: FO / SO mixture arms.
- `gru_gdn_long_delay_summary.json` and `gru_gdn_long_delay_{curves,so_check,sp}.csv`: GRU / GDN long-delay comparison.
  - AUROC curves and bootstrap differences.
  - `outcome`.
- `gru_same_episodes_summary.json`: a GRU (arm `R`) against E2E-SO (`T-E2E-SO-lo`) on the same training and evaluation episodes, original training seeds 0–2, final checkpoints (App. E, caveat ii).
  - `arms/<arm>/N`, `X` and `S` are beyond-window AUROCs per seed, and `*_mean` is their mean over seeds.
  - `s_per_step` is the time per training step, and `step_time_ratio_so_over_gru` is SO's step time over the GRU's.
  - These reads use the training run's own evaluation sets, not the fresh-episode sets.
- `show_write_size_ratio.csv`: the registered write-size ratio `gamma_show` (App. E), per arm and seed, original seeds 0–2, final checkpoints, same evaluation sets.
  - The numerator is the median update norm of the writes that target the show and cover frames in hold_X episodes.
  - The denominator is the median, over hold_X episodes, of the largest write from the frame after the command to the first lift frame.
- `readtime_analysis.csv`: the post hoc read-time analysis of App. E, per arm and seed, final checkpoints.
  - `place_probe_acc` is a ridge probe of the cup place (3 classes, `chance` 1/3). It reads the memory's response at the lift keys, in hold_N episodes at L = 12, on 749 rows (`probe_n`).
  - `place_probe_shuffled_acc` (shuffled labels) and `place_probe_window_acc` (window state only) are its controls.
  - `place_mean_share_min` and `place_mean_share_max` give the range of s = ‖place means of the show writes‖ / ‖place means of the lift writes‖ in the memory's response. The range is taken over the two fit halves and the three imagined-lift steps.
  - The per-step values of s are `fit{0,1}_scale_step{0,1,2}` in `mechanism_cells.csv`. They exist only for the three arms of that analysis.
- Table 7's `E_X` column is in `original_seeds_fresh_episodes_summary.json`, under `readings/rows/<arm>/<seed>/E_X`.
- Arm codes: `R` = GRU, `T-0` = window only (no memory), and `-lo` / `-hi` = the lower / higher inner step size.
- Some arm labels are in Russian: `шагов` = steps, `сброс` = reset, `ориентир` = reference.

**theory-checks/** — App. A. Performative correspondence, key–pair model and reduced model.
- `performative_correspondence_check.json`: SO − FO gradient differences by parameter group, per rule and clip setting, with `verdict`.
- `key_pair_symbolic_check.json`: one entry per derived identity (P3.1 … P7.1), with `symbolic_zero`, `numeric_max_abs` and `passed`.
- `so_residual_roots.csv`: the SO fixed-point residual `residual_s_star` per λbη.
- `reduced_model_predictions.csv`: reduced-model predictions per arm.
  - First bound, first loss and rebind.
  - Final accuracy and p_kv.
- `reduced_model_prediction_sensitivity.csv`: the same predictions across conventions (ranges).
- `reduced_model_attribution_test.csv`: the reduced model's registered linear-read attribution test, with one row per SO final checkpoint (`e2e2emb`, seeds 1–9; seed 0 was the development unit).
  - `Mkv_exact` is the change in query margin when the chunk-0 kv write is added, with later updates frozen. The margin is projected on the base margin direction and averaged over queries and `n_eval_sequences`.
  - `Mkv_lin` is the same quantity from a central finite-difference linearisation at the base point.
  - `rel_error` = |`Mkv_lin` − `Mkv_exact`| / `Mkv_exact`, and `passes` is `rel_error` ≤ `tolerance` (.25).
  - The count rule is: supported if at least 7 of 9 pass, falsified if at most 4 pass. The result is 0/9.
- `performative_replay_agreement.json`: the replay check on the toy itself, in float64.
  - It holds the relative difference between the toy's own FO / SO training loss and the replayed loss, and the largest query-logit difference, per checkpoint.
  - `registered_checkpoints` lists 45 checkpoints in sets `fo_pre_loss`, `fo_early_post_loss`, `fo_final`, `fo_linkv_final` and `so_final`.
  - `compatibility_gate_checkpoints` lists 3 development checkpoints, and `summary` gives the maxima.
- `write_factorisation_results.json`: write-factorisation units (FO, SO, linear at η 10/40/160).
  - Verdicts `A*`/`B*`, identities, `per_eta`, `set_medians` and `units`.
- `write_factorisation_units.csv`: per unit, the derivative components.
- `write_factorisation_kappa.csv`: κ quantiles per unit and per group.
  - Groups: FO responder / non-responder, SO bound / unbound, development unit.
