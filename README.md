# What Second-Order Meta-Gradients Buy in TTT-E2E: Probe Memory at a Language-Modelling Price

TTT-E2E meta-learns through the gradient steps that write context into its fast weights. This
repository holds the code and configurations for a study of what second-order (SO) differentiation
through those steps buys over a first-order (FO) step. In a registered equal-GPU-second test on a
from-scratch text model (25.6M non-embedding parameters), FO with detached inner gradients and
2.9× the steps learns a better language model, but not SO's probe memory. Fresh seeds replicate
this. A controlled toy shows how attention's copy starves the FO write, and a linear inner loss
rescues FO given a key address. The package covers the text pipeline (data preparation, the
mini-m/mini-s models, the 125M port, reads, passkey and analyses), the toy (echo, Markov and load
tasks, interventions and analyses) and the CPU test suite.

## Install

The code needs Python ≥ 3.11 (`.python-version` pins 3.13) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run --locked pytest -q
```

The test suite runs offline on CPU without private data: 1311 passed and 46 skipped in 3–5
minutes on an Apple M4 Max CPU. Tests that need the prepared DCLM store (`data/p2data/`) or the R4-3
toy checkpoints (`outputs/r4-3/checkpoints/`) are skipped with a reason.

Every experiment uses one entry point:

```sh
uv run --locked python -m tttmem.run <config.toml> [--process-name NAME]
```

The `experiment` key of the config selects the driver, and the table of the same name holds its
settings. The entry point prints the effective config, per-step lines and a final `RESULT_JSON`
line to stdout. Configs that set `output_dir` or `out_dir` also write files there: under
`outputs/` or `data/`, or `outputs_lod*/` for LOD. Toy runs that save checkpoints write them to
`checkpoints/`. All of these are git-ignored. Without `--process-name` the config's
`process_name` is used.

Seal and registration checks are off by default. They switch on only when a config sets
`require_seal = true`, which no shipped config does.

## Data

- **Text: DCLM-Baseline**, `mlfoundations/dclm-baseline-1.0-parquet` at revision
  `817d6752765f6a41261085171dd546b104f60626`. The prepare step streams the parquet shards from the
  Hugging Face Hub, filters documents of at least 30 000 characters and removes near-duplicates. It
  then writes the token splits (`eval`, `ho`, `calib`, `A`, `B`, and `C` for the budget test) to
  `data/p2data/`.
- **Tokenizer: Llama 3** (`tokenizer.json` of `NousResearch/Meta-Llama-3-8B` at revision
  `315b20096dc791d381d514deb5f8bd9c8d6d3061`). The prepare step downloads it from the Hugging Face
  Hub into your local cache. It is not included in this repository.
- **Toy tasks** are generated on the fly from the config seed; they need no data.

```sh
# CPU only, no torch import; writes data/p2data/ and data/p2data.tar
uv run --locked python -m tttmem.run configs/p21/prepare.toml
# adds split C (sha1-deduplicated against every split) to data/p2data/
uv run --locked python -m tttmem.run configs/soq/n3b-prepare-c.toml
```

The expected SHA-256 of every split is in `configs/s14/data/expected-sha256.json` and
`expected-c-sha256.json`. Text jobs with `[p2.data] mode = "uploaded"` verify `data/p2data/`
against these files before training.

The replication jobs (§3) read an upload pack built from this store. Unpack `data/s14/p2data.tar`
in `data/` on the GPU machine. The build re-pins `configs/s14/data/expected-c-sha256.json`;
`git diff` shows whether your split C differs.

```sh
# CPU only: copies the 13 base files to data/s14/p2data/, adds split C, writes data/s14/p2data.tar(.sha256)
uv run --locked python scripts/s14_build_pack.py --base data/p2data --pack-dir data/s14/p2data \
  --report-dir outputs/s14-pack --process-name <NAME>
# full check of the pack and the archive against both hash files
uv run --locked python scripts/s14_verify_pack.py --pack-dir data/s14/p2data \
  --report-dir outputs/s14-pack --process-name <NAME>
# downloads the first shard, rebuilds 64 documents and compares them byte for byte with data/p2data/
uv run --locked python scripts/s14_data_slice.py --local data/p2data \
  --out outputs/s14-pack/slice.json --process-name <NAME>
```

## Hardware

- **Text** (§3, §5, App. D): one CUDA GPU with 24 GB per job. Each text config is one job.
  The data preparation steps run on CPU.
- **Toy** (§4, App. B): CPU or Apple MPS (`device = "auto"`, `"cpu"` or `"mps"`). The series run as
  sessions or pools (POOL, TRG, RDO-RPE, FRS-B, B9, LOD, LOD-X, RTW, RTW-C, P3-a, R4-4c) set
  `device = "cuda"` and ran on one CUDA GPU.

## Reproduction by paper section

All commands below assume the repository root as the working directory. Output locations are the
`out_dir`/`output_dir` of each config. For toy configs without one, redirect stdout to a log, which
the analyzers read.

### §3 Text: the equal-GPU-second budget test and its replication

| Step | Config(s) | Output |
|---|---|---|
| SO and FO-CE reference arms, mini-m, seeds 0–2, 1200 steps | `configs/mini/matrix-g43.toml`, `matrix-g44.toml`, `matrix-g45.toml` | `outputs/mini/matrix/g43` … `g45` |
| Original study (SOQ-N3B): timing pilot, then FO-CM per seed at equal GPU-seconds | `configs/soq/n3b/pilot.toml`, `configs/soq/n3b/fo-cm-s{0,1,2}.toml` | `outputs/mini/matrix/n3b` |
| Frontier reads of SO, FO-CE and FO-CM | `configs/soq/n3b/reads-so-fo-ce.toml`, `reads-fo-cm-s{0,1,2}.toml` | `outputs/mini/matrix/n3b-reads` |
| Replication (S14-R): 12 jobs, 6 arms × seeds 3–5, training, reads and passkey | `configs/s14/r-pod01.toml` … `r-pod12.toml` (the exact files run) | `outputs/s14` |
| Replication GPU check (`canary` mode: 30 steps of every arm plus read and passkey batch benchmarks) | `configs/s14/canary.toml` | `outputs/s14` |

```sh
uv run --locked python -m tttmem.run configs/mini/matrix-g43.toml       # likewise g44, g45
uv run --locked python -m tttmem.run configs/soq/n3b/fo-cm-s0.toml      # likewise s1, s2
uv run --locked python -m tttmem.run configs/soq/n3b/reads-so-fo-ce.toml
uv run --locked python -m tttmem.run configs/s14/r-pod01.toml           # one job per GPU, r-pod01 … r-pod12

# analyses (CPU, numpy only)
# the original-study analyses read the G0, G0b and G0c reads (§5) through these links
mkdir -p outputs/g0 && ln -s ../mini/matrix/g0 outputs/g0/data && ln -s ../mini/matrix/g0b outputs/g0/data-g0b \
  && ln -s ../mini/matrix/g0c outputs/g0/data-g0c
uv run --locked python -m tttmem.p2.s14_n3b_reference                   # original study -> outputs/n3b-results
uv run --locked python -m tttmem.p2.s14_analysis a0 --out-dir outputs/s14-a0          # cross-fit selection
uv run --locked python -m tttmem.p2.s14_analysis s14 --root outputs/s14 --out-dir outputs/s14/analysis
```

The replication analysis reads `outputs/s14/reads/mini-m/<arm>/s<seed>/` from all 12 jobs, the
original study's reads under `outputs/mini/matrix/n3b-reads` and `data/p2data/eval.jsonl`. The
matched-loss frontier at mini-s size (App. C, SOQ-FS) uses `configs/soq/fs.toml`, `fs-probe.toml`
and `fs-full.toml`; analyse it with
`uv run --locked python -m tttmem.p2.fs_analysis outputs/soq/fs --out <dir>`.

### §4 Toy: the starved write

Each row is `uv run --locked python -m tttmem.run configs/toy/<config> > <log>`, followed by the
listed analyzer (commands below). Rows without an analyzer report their counts in the `RESULT_JSON`
line.

| Series | Config | Analysis |
|---|---|---|
| R1 (FO vs SO, echo and Markov task) | `r1.toml` | `RESULT_JSON` |
| R2 bundle, R2b (six-seed contrast), R3 (random-prefix binding) | `r2-bundle.toml`, `r2b.toml`, `r3.toml` | `RESULT_JSON` |
| R2-4 write-position ablation, R2-4b key probes, R4-0 rank | `r2-4.toml`, `r2-4b.toml`, `r4-0.toml` | `RESULT_JSON` |
| R4-1 key/error/embedding arms, TH-RC fixed-η retention | `r4-1.toml`, `th-rc.toml` | `RESULT_JSON` |
| R4-2 outer-loss mask | `r4-2.toml` | `analyze_r42.py` |
| R4-2L linear inner loss | `r4-2l.toml` | `analyze_r42l.py`, `r42l_s5_fp32.py` (S5 in fp32) |
| R4-3 second-order halves | `r4-3.toml` | `analyze_r43.py`, `r43_ek_share_ckpt.py` |
| R4-4a copy-edge mask, copy noise, two-stage | `r4-4a.toml` | `analyze_r44a.py` |
| R4-4b AICE rule | `r4-4b.toml` | `analyze_r44b.py` |
| R4-4c toy rehearsal of the 125M continuation | `r4-4c.toml` | `analyze_r44c.py` |
| BW timing gate; BW1a, BW1b copy noise | `bw-gate-cpu.toml`, `bw-gate-mps.toml`, `bw1a.toml`, `bw1b.toml` | `RESULT_JSON` |
| CHR chronology | `chr.toml` | `analyze_chr.py` |
| POOL: FRS-A, TRG, RDO-RPE fresh-seed session | `pool.toml` (standalone: `frs-a.toml`, `trg.toml`, `rdo-rpe.toml`; re-reads: `trg-reanalysis.toml`, `rdo-rpe-reanalysis.toml`) | `analyze_frs.py`, `analyze_pool_trg.py`, `analyze_trg.py`, `analyze_rdo.py` (also RPE), `plot_rdo.py` |
| P0.0 Markov-task cost of the linear loss | `p0-0.toml` | `analyze_p00.py` |
| P0.1 mechanism probes | `p0-1.toml` | output in `outputs/p0-1` |
| SOQ-D (P3-a) write-rule control | `p3a.toml` | `analyze_p3a.py` |
| FRS-B plain echo × {CE, linear} | `frs-b.toml` | `analyze_frsb.py` |
| B9 address auxiliary | `b9.toml` | `analyze_b9.py` |
| LOD load scaling; LOD-X longer budget | `lod.toml`, `lod-x.toml` | `analyze_lod.py`, `analyze_lodx.py`, `plot_lod.py` |
| RTW write-gain gate and partial-SO screen; RTW-C confirmation | `rtw.toml`, `rtw-c.toml` | `analyze_rtw.py` |
| E1-VE world-model echo (App. E) | `configs/e1-ve/*.toml` (generated by `scripts/e1ve_configs.py`) | `score_e1ve.py` |

Some rows read the outputs of other rows. Move a toy run's saved checkpoints to
`outputs/<series>/checkpoints/`, for example `mv checkpoints outputs/r4-3/checkpoints`. `p0-1.toml`
reads R4-1, R4-2, R4-2L and R4-3 there, and `r4-4c.toml` reads R4-3. TRG (`trg.toml`,
`trg-reanalysis.toml`) forks from CHR and reads `data/trg-inputs/`: the CHR checkpoints in
`checkpoints/`, the gzipped CHR log as `chr-full.log.gz` and the CHR analyzer's `chr_series.csv`.

```sh
# analyzers (CPU); <log> is the stdout of the row's run, <dir> a new output folder
uv run --locked python scripts/analyze_r42.py <r4-2 log> <dir>
uv run --locked python scripts/r42l_s5_fp32.py configs/toy/r4-2l.toml outputs/r4-2l/checkpoints \
  outputs/r4-2l/s5-fp32.json --log <r4-2l log>
uv run --locked python scripts/analyze_r42l.py <r4-2l log> outputs/r4-2l --s5-fp32 outputs/r4-2l/s5-fp32.json
uv run --locked python scripts/analyze_r43.py <r4-3 log> <dir>
uv run --locked python scripts/r43_ek_share_ckpt.py --ckpt-dir outputs/r4-3/checkpoints --out <dir>
uv run --locked python scripts/analyze_r44a.py <r4-4a log> --out <dir> --anchor-table outputs/r4-2l/paired_seeds.csv
uv run --locked python scripts/analyze_r44b.py <r4-4b log> <dir>
uv run --locked python scripts/analyze_r44c.py <r4-4c log> <dir>
uv run --locked python scripts/analyze_chr.py <dir> <chr log>
uv run --locked python scripts/analyze_frs.py outputs/pool-output/pool.json --out <dir>
uv run --locked python scripts/analyze_pool_trg.py outputs/pool-output/pool.json --out <dir>
uv run --locked python scripts/analyze_trg.py outputs/trg-output --process-name <NAME>
uv run --locked python scripts/analyze_rdo.py outputs/pool-output/reanalysis/rdo-rpe/results.json --out <rdo dir>
uv run --locked python scripts/analyze_p00.py <p0-0 log> --out <dir>
uv run --locked python scripts/analyze_p3a.py --root outputs/soq/p3a
uv run --locked python scripts/analyze_frsb.py outputs/frsb-output/result.json <dir>
uv run --locked python scripts/analyze_b9.py outputs/b9-output/result.json outputs/frsb-output/result.json <dir>
uv run --locked python scripts/analyze_lod.py outputs_lod
uv run --locked python scripts/analyze_lodx.py outputs_lodx --lod-ref outputs_lod
uv run --locked python scripts/analyze_rtw.py outputs/rtw/wave-06     # RTW-C: outputs/rtw/wave-07
uv run --locked python scripts/score_e1ve.py <wave-1 log> [<wave-2 log>] --json <file>
# figures; matplotlib is not a project dependency
uv run --locked --with matplotlib python scripts/plot_rdo.py <rdo dir>
uv run --locked --with matplotlib python scripts/plot_lod.py outputs_lod
```

`analyze_rdo.py` writes the RDO and the RPE scores; `analyze_rpe.py` is the same entry point.
`uv run --locked python scripts/e1ve_configs.py` writes the E1-VE configs; `configs/e1-ve/ve.toml`
then runs 8 workers instead of the generated 6. Every analyzer's `--help` lists its options.

Short CPU versions of most toy configs are in `configs/toy/smoke/`. The `*-gpu.toml` files there
are short CUDA checks.

### §5 Text: mechanism diagnostics and levers (mini-m, mini-s)

| Series | Config(s) | Analysis |
|---|---|---|
| MINI: SO, FO-CE and residual-floor arms | `configs/mini/matrix-core-m.toml`, `matrix-core-s.toml` (per-seed splits `matrix-g40` … `g47`) | `mini_analysis` |
| MINI-A write–read alignment | `configs/mini/mini-a.toml` | `mini_analysis --minia` |
| MINI-E eval-carried read of the floors | `configs/mini/mini-e.toml` | `mini_analysis` |
| MINI-P partial SO (2/16 rows) | `configs/mini/matrix-minip.toml` | `mini_analysis` |
| MINI-L late SO steps after FO-CE | `configs/mini/minil-g81.toml`, `g82`, `g83` | `mini_analysis.minil_analysis` (Python function) |
| G0, G0b, G0c write gain, η and σ-CE, frontier | `configs/g0/`, `configs/g0b/`, `configs/g0c/` | `g0_analysis`, `g0b_analysis`, `g0c_analysis` |
| SOQ-V1 echo-in-text template validity | `configs/soq/v1.toml` | `v1_analysis` |
| SOQ-P1 injected binding (pilot; v3, v3x, v3p) | `configs/soq/p1-pilot.toml`; `p1v3-so.toml`, `p1v3-fo.toml`, `p1v3x.toml`, `p1v3p.toml` | `soq_p1_analysis`, `soq_p1v3_analysis`, `soq_p1v3_post_analysis` |
| TBB reduced-vocabulary testbed | `configs/tbb/prepare.toml` (tokenizer from `scripts/train_bpe.py`), `matrix-gpu.toml` | `scripts/tbb_analysis.py` |

```sh
# analyses (CPU, numpy only); <f> is an output file
uv run --locked python -m tttmem.p2.mini_analysis outputs/mini/matrix --out-json <f> --out-md <f>
uv run --locked python -m tttmem.p2.mini_analysis outputs/mini/matrix/minia --minia --out-json <f> --out-md <f>
# MINI-L writes minil_analysis.json and .md into its root
uv run --locked python -c "from tttmem.p2.mini_analysis import minil_analysis; minil_analysis('outputs/mini/matrix/minil')"
uv run --locked python -m tttmem.p2.g0_analysis outputs/mini/matrix/g0
uv run --locked python -m tttmem.p2.g0b_analysis outputs/mini/matrix/g0 outputs/mini/matrix/g0b \
  --out outputs/mini/matrix/g0b
uv run --locked python -m tttmem.p2.g0c_analysis outputs/mini/matrix/g0 outputs/mini/matrix/g0b \
  outputs/mini/matrix/g0c --out outputs/mini/matrix/g0c
uv run --locked python -m tttmem.p2.v1_analysis outputs/soq/v1 --out <f>
uv run --locked python -m tttmem.p2.soq_p1_analysis outputs/soq/p1/pilot-s0 --out <f> --md <f>
# v3: so-s0-a1g and fo-s0-a1g; v3x: so-s0-a1g-x3200
uv run --locked python -m tttmem.p2.soq_p1v3_analysis outputs/soq/p1v3/so-s0-a1g --out <f> --md <f>
uv run --locked python -m tttmem.p2.soq_p1v3_post_analysis outputs/soq/p1v3/post \
  --v3x-start outputs/soq/p1v3/so-s0-a1g-x3200/eval_step2400.npz \
  --v3x-end outputs/soq/p1v3/so-s0-a1g-x3200/eval_step3200.npz \
  --parent outputs/soq/p1v3/so-s0-a1g/eval_step2400.npz --out <f> --md <f>
uv run --locked python scripts/tbb_analysis.py outputs/tbb/runs --out-json <f> --out-md <f>
```

The TBB tokenizer comes from
`uv run --locked python scripts/train_bpe.py --out data/micro16k/tokenizer.json --process-name <NAME>`.

### App. D: the released TTT-E2E 125M continuation

The official checkpoint is not included (see Licences). Download the official TTT-E2E 125M Orbax
`model_weights` checkpoint yourself, then convert it on CPU in an isolated environment that holds
JAX, Orbax and torch (JAX and Orbax are not project dependencies):

```sh
uv run --isolated --no-project --python 3.12 \
  --with 'jax==0.5.3' --with 'jaxlib==0.5.3' --with 'ml-dtypes==0.6.0' \
  --with 'orbax-checkpoint==0.11.13' --with 'torch==2.14.1' --with 'numpy==2.5.3' \
  --with 'setproctitle==1.3.8' \
  python scripts/convert_official_125m.py <checkpoint dir> --out data/ckpt/init.pt --process-name <NAME>
```

`<checkpoint dir>` is the local `model_weights` directory or a step directory that contains it.
The converter writes `init.pt` and `init.pt.sha256` and refuses to overwrite an existing file.

| Series | Config(s) | Output |
|---|---|---|
| P2-0 port and timing profile | `configs/p20/profile.toml` (`profile-cpu-tiny.toml` on CPU) | stdout |
| P2-1 400-step continuation, SO vs FO-CE (plus linear and AICE arms) | `configs/p21/calibrate.toml`, `e2e2.toml`, `e2e1-ce.toml`, `e2e1-lin.toml`, `e2e1-aice.toml` | `outputs/p21/<arm>/` |
| P2-2 write/read swap (eval only) | `configs/p22/swap.toml` | `outputs/p22/swap` |

The P2-1 configs read `data/ckpt/init.pt` and `data/p2data/`. The endpoint statistics (CPU,
numpy only) freeze the endpoint N from the step-60 evaluations, then test at step 400:

```sh
uv run --locked python -m tttmem.p2.decide size --decision-dir outputs/p21/decisions \
  --arm e2e2=outputs/p21/e2e2/attempt0 --arm e2e1-ce=outputs/p21/e2e1-ce/attempt1 \
  --arm e2e1-lin=outputs/p21/e2e1-lin/attempt1 --arm e2e1-aice=outputs/p21/e2e1-aice/attempt1
uv run --locked python -m tttmem.p2.decide endpoint --decision-dir outputs/p21/decisions \
  --arm e2e2=outputs/p21/e2e2/attempt0 --arm e2e1-ce=outputs/p21/e2e1-ce/attempt1 \
  --arm e2e1-lin=outputs/p21/e2e1-lin/attempt1 --arm e2e1-aice=outputs/p21/e2e1-aice/attempt1
uv run --locked python -m tttmem.p2.a7 --decision-dir outputs/p21/decisions --data-dir data/p2data \
  --arm e2e2=outputs/p21/e2e2/attempt0 --arm e2e1-ce=outputs/p21/e2e1-ce/attempt1 \
  --arm e2e1-lin=outputs/p21/e2e1-lin/attempt1 --arm e2e1-aice=outputs/p21/e2e1-aice/attempt1
```

### Post hoc analyses

`scripts/analysis/` holds the scripts behind the post hoc series. Each folder is named after the
results folder it fills:
- the equal-token check;
- repeat exposure (XPO-0);
- the copyability/distance split (MINI-D);
- the theory checks of App. A;
- the residual AUC of App. B;
- the parameter counts.

[scripts/analysis/README.md](scripts/analysis/README.md) lists, for each script:
- what it computes;
- the paper numbers it backs;
- its inputs (per-document reads and probe outputs are not included);
- the command.

## CPU checks that ran for this release

```sh
uv sync --locked
uv run --locked pytest -q
# end-to-end text job (training, reads, passkey, analysis) on a tiny CPU model with a
# synthetic store and test-only hosts/tokenizer; no download, about 30 s and 0.4 GB
uv run --locked python scripts/s14_integration_smoke.py --out outputs/s14-smoke --process-name TTM-s14-01-smoke
# short toy runs (a few seconds each), for example
uv run --locked python -m tttmem.run configs/toy/smoke/r2b.toml
uv run --locked python -m tttmem.run configs/toy/smoke/lod.toml
```

## Results

Per-seed result files and the experiment index are described in [results/README.md](results/README.md).

## Licences

- Code: MIT (see `LICENSE`).
- DCLM-Baseline: CC BY 4.0. It is downloaded by the user and not included.
- Llama 3 tokenizer: Meta Llama 3 Community License. It is downloaded by the user and not included.
- TTT-E2E checkpoint and code: the upstream repository states no licence, so neither the checkpoint
  nor weights derived from it are included. `scripts/convert_official_125m.py` converts a checkpoint
  you download yourself. `tests/jax_parity/official_tiny.npz` is a numerical fixture computed with the
  upstream code; its generator is not included (see `tests/jax_parity/README.md`).
- `tests/ref_ouroboros/`: an unmodified copy of `ttt_pt` from ttt-ouroboros, MIT (see its `LICENSE`),
  used only by the tests.

## Citation

```bibtex
@misc{lapchevskiy2026secondorder,
  title  = {What Second-Order Meta-Gradients Buy in {TTT-E2E}: Probe Memory at a Language-Modelling Price},
  author = {Lapchevskiy, Valentin},
  year   = {2026},
  howpublished = {\url{https://github.com/gen4sp/second-order-ttt}}
}
```
