"""Descriptive EK share on saved checkpoints (registered R4-3 pre-launch amendment).

Usage: python scripts/r43_ek_share_ckpt.py --ckpt-dir checkpoints --out out_dir
       [--config configs/toy/r4-3.toml] [--seeds 0-9] [--steps 200,300] [--n-seq 8]

The in-run gate measures the share of the second-order correction at init (the
``R43_EK_SHARE`` line, read by ``analyze_r43.py``). This script asks where starvation
acts: the share on the *trained* ``e2e1emb`` checkpoints at steps 200 and 300, seeds
0-9, CPU and fp64. If the step-200 median share_X is below 0.05 the wording of an
"X-no retention gain shown" label becomes "X carries < 5% of the correction where
starvation acts" (registered amendment), so the numbers are written out rather than only
reported.

The batch is the first ``n_seq`` sequences of the seed's first training batch, drawn
from ``rng([seed, TRAIN_SEED])`` — the same stream and the same slice as the in-run
gate, so the two measurements differ only in the weights. Everything runs on CPU with
``map_location="cpu"``, so a checkpoint saved on CUDA loads unchanged. The config
supplies the model size and the task; the file name supplies the seed, the arm and the
step, since ``save_checkpoint`` writes ``<cell_id>__step<N>.pt``.
"""

import argparse
import csv
import json
import re
import sys
import tomllib
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tttmem import exp_ttt as t  # noqa: E402

CELL_STEP = re.compile(r"^echo__s(\d+)__(?P<arm>[a-z0-9_]+)__eta[^_]*__st\d+__step(\d+)$")


def parse_name(path):
    """(seed, arm, step) of a step checkpoint, or None when the name is not one."""
    m = CELL_STEP.match(path.name[: -len(".pt")])
    if not m:
        return None
    return int(m.group(1)), m.group("arm"), int(m.group(3))


def expand(text):
    """``0-9`` -> [0..9], ``200,300`` -> [200, 300]."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part[1:]:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for csv and json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/toy/r4-3.toml")
    parser.add_argument("--arm", default=t.EK_ANCHOR,
                        help="whose checkpoints to measure (default: the e2e1emb anchor)")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--steps", default="200,300")
    parser.add_argument("--n-seq", type=int, default=t.EK_SHARE_NSEQ)
    parser.add_argument("--eta-base", type=float, default=10.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.config.open("rb") as f:
        cfg = tomllib.load(f)["ttt"]
    st = t._settings(cfg)
    cell = next(c for c in cfg["cells"] if c["task"] == "echo")
    task = t._task(st, cell["task"])
    seeds, steps = expand(args.seeds), expand(args.steps)

    rows = []
    for path in sorted(args.ckpt_dir.glob("*.pt")):
        parsed = parse_name(path)
        if parsed is None or parsed[1] != args.arm or parsed[0] not in seeds or parsed[2] not in steps:
            continue
        seed, _, step = parsed
        # map_location="cpu": these checkpoints come from CUDA runs
        state = torch.load(path, map_location="cpu", weights_only=True)
        got = t.ek_share(state, task, st, seed, args.n_seq, args.eta_base)
        row = dict(arm=args.arm, seed=seed, step=step, checkpoint=path.name,
                   first_batch_sha1=got["first_batch_sha1"],
                   first_order_norm=round(got["first_order_norm"], 8),
                   correction_norm=round(got["correction_norm"], 8))
        for half, arm in t.EK_HALVES.items():
            for key in ("share", "cos", "rel"):
                value = got[arm][key]
                row[f"{half}_{key}"] = round(value, 6) if value is not None else None
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        print(f"!! no {args.arm} checkpoints at steps {steps} seeds {seeds} in {args.ckpt_dir}",
              flush=True)
    summary = dict(checkpoint_dir=str(args.ckpt_dir), arm=args.arm, n_seq=args.n_seq,
                   eta_base=args.eta_base, batch="first n_seq sequences of rng([seed, TRAIN_SEED])",
                   dtype="float64", device="cpu", threshold=t.EK_SHARE_MIN,
                   n_checkpoints=len(rows), per_step={})
    for step in steps:
        group = [r for r in rows if r["step"] == step]
        if not group:
            continue
        summary["per_step"][str(step)] = dict(
            n=len(group),
            median_share_E=median([r["E_share"] for r in group if r["E_share"] is not None])
            if any(r["E_share"] is not None for r in group) else None,
            median_share_K=median([r["K_share"] for r in group if r["K_share"] is not None])
            if any(r["K_share"] is not None for r in group) else None)
    (args.out / "ek_share_checkpoints.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=2, sort_keys=True) + "\n")
    keys = ["arm", "seed", "step", "checkpoint", "first_batch_sha1", "first_order_norm",
            "correction_norm", "E_share", "E_cos", "E_rel", "K_share", "K_cos", "K_rel"]
    with (args.out / "ek_share_checkpoints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary["per_step"], sort_keys=True), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())