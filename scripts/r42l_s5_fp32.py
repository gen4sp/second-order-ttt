"""Recompute R4-2L S5 on each listed cell's final checkpoint, in fp32 on CPU.

Usage:
  PROCESS_NAME=TTM-r42L-09-s5 uv run --locked python scripts/r42l_s5_fp32.py \
      CONFIG.toml CHECKPOINT_DIR OUT.json --log RUN.log

Raw state-dict checkpoints require CELL_RESULT records from --log. A checkpoint
with a state_dict and explicit arm metadata can also be used without a log.
The JSON list is consumed directly by analyze_r42l.py --s5-fp32.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tomllib

import numpy as np
import torch

from tttmem import exp_ttt as E


def cell_results(log: Path | None) -> dict:
    """Read final cell identities, including their explicit arm and checkpoint."""
    records = {}
    if log is not None:
        with log.open() as stream:
            for line in stream:
                if line.startswith("CELL_RESULT "):
                    record = json.loads(line[len("CELL_RESULT "):])
                    records[record["cell_id"]] = record
    return records


def recompute(cfg: dict, checkpoint_dir: Path, records: dict) -> list[dict]:
    """Use the cell's own loss for both the linear delta and its clipping scale."""
    dev = torch.device("cpu")
    rows = []
    for cell in cfg["cells"]:
        st = E._settings(cfg, cell["task"])
        task = E._task(st, cell["task"])
        ident = E._cell_id(cell, int(cell.get("steps", task.steps)))
        record = records.get(ident, {})
        # Only the final checkpoint; never substitute a step snapshot or init.
        name = Path(record.get("checkpoint", f"{ident.replace('/', '__')}.pt")).name
        checkpoint = torch.load(checkpoint_dir / name, map_location=dev, weights_only=True)
        metadata = checkpoint.get("metadata", {})
        checkpoint_arm = checkpoint.get("arm", metadata.get("arm"))
        record_arm = record.get("arm")
        if record_arm is not None and checkpoint_arm is not None and record_arm != checkpoint_arm:
            raise ValueError(f"{ident}: CELL_RESULT and checkpoint arms disagree")
        arm = record_arm if record_arm is not None else checkpoint_arm
        if arm is None:
            raise ValueError(f"{ident}: arm required from CELL_RESULT or checkpoint metadata")
        if arm != cell["arm"] or arm not in E.EMB_ARMS:
            raise ValueError(f"{ident}: unexpected S5 arm {arm}")
        E.check_arm_task(arm, cell["task"])
        if record and (record.get("seed") != cell["seed"] or record.get("task") != cell["task"]):
            raise ValueError(f"{ident}: CELL_RESULT seed/task mismatch")
        model = E.build_model(task, st, dev, int(cell["seed"]), arm).float()
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        model.inner_sign = float(record.get("inner_sign", cell.get("inner_sign", 1.0)))
        E.set_kamin_regime(model, arm, int(record.get("steps", cell.get("steps", task.steps))), st)
        model.eval()
        data = E.eval_batches(
            task, *task.raw(np.random.default_rng([E.PROBE_SEED, 0]), st.n_probe),
            dev, st.eval_batch)
        probe = E.r4_probe(model, task, data, float(cell["eta_base"]), st.g_clip, arm=arm)
        diff = probe["lin_kv_max_diff"]
        passed = (math.isfinite(diff) and diff <= 1e-4 and
                  probe["lin_kv_acc_diff"] <= .002)
        rows.append(dict(cell_id=ident, arm=arm, seed=int(cell["seed"]),
                         s5_max_diff=diff, passed=bool(passed)))
        print("S5_FP32 " + json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME"))
    args = parser.parse_args()
    if not args.process_name:
        parser.error("set PROCESS_NAME or --process-name")
    E._mark_process(args.process_name)
    torch.set_num_threads(4)
    with args.config.open("rb") as stream:
        cfg = tomllib.load(stream)["ttt"]
    rows = recompute(cfg, args.checkpoint_dir, cell_results(args.log))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(f"S5_FP32 summary: {sum(r['passed'] for r in rows)}/{len(rows)} passed", flush=True)


if __name__ == "__main__":
    main()
