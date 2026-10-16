"""P2-1/A6 rule-dose calibration.

The released checkpoint at step 0, the first ``docs`` rows of the ``calib`` split, the ordinary
full-document NLL with the rule writing in every chunk (the ``heldout`` evaluation mode), over a
fixed grid of inner-lr multipliers. ``s_rule`` = argmin of the mean NLL (exact ties -> larger s).
CE runs on its own grid as a descriptive control; s = 0 (no adaptation) is the reference.

Stdout: ``P2_CALIB`` per (rule, s) and ``P2_CALIB_RESULT``. ``calibration.json`` is write-once.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from .meta import InnerCfg, stream_eval


@dataclass
class CalibCfg:
    init: str = ""
    init_label: str = ""
    data_dir: str = ""
    out_dir: str = "p2_calib"
    preset: str = "125m"
    dtype: str = "bf16"
    device: str = "cuda"
    seq_len: int = 8192
    split: str = "calib"
    docs: int = 128
    batch: int = 4
    fused_attn: bool = True
    inner_lr: float = 1.0
    inner_clip: float = 1.0
    rules: list = field(default_factory=lambda: ["lin", "aice"])
    grid: list = field(default_factory=lambda: [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64])
    ce_grid: list = field(default_factory=lambda: [2, 1, 1 / 2, 1 / 4])

    @classmethod
    def from_dict(cls, d: dict) -> "CalibCfg":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"mode", "process_name"}
        if unknown:
            raise ValueError(f"unknown p2 calibrate keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})


def full_nll(model, docs: np.ndarray, rule: str, lr: float, clip: float, *, batch: int,
             device, fused_attn: bool) -> dict:
    """Per-document full-document mean NLL with ``rule`` writing in every chunk (lr = 0:
    no adaptation), plus inner-gradient norms, clip rate and per-layer update norms."""
    inner = InnerCfg(rule=rule, lr=lr, clip=clip)
    bos = model.cfg.bos_id
    out, norms, upd = [], [], []
    for b0 in range(0, len(docs), batch):
        arr = torch.from_numpy(docs[b0:b0 + batch].astype(np.int64)).to(device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        base = tgt != bos
        r = stream_eval(model, ids, tgt, base, inner, adapt=lr > 0, fused_attn=fused_attn)
        nll = r["nll"]
        out.append(((nll * base).sum(1) / base.sum(1)).double().cpu().numpy())
        if lr > 0 and r["inner_norm"] is not None:
            norms.append(r["inner_norm"].float().cpu())
            upd.append(r["update_norm"].float().cpu())
    res = {"doc": np.concatenate(out)}
    if norms:
        n = torch.cat(norms, 1)
        res["inner_norm_median"] = float(n.median())
        res["clip_frac"] = float((n > clip).double().mean()) if clip > 0 else 0.0
        res["update_norm"] = torch.cat(upd, 1).mean((0, 1)).tolist()
    return res


def select(results: dict, grid: list) -> float:
    """argmin of the mean NLL over the grid; exact ties go to the larger multiplier."""
    return max(grid, key=lambda s: (-results[s]["mean"], s))


def calibrate(cfg: CalibCfg) -> dict:
    from .data import open_split, sha256_file
    from .train import TrainCfg, build

    out = Path(cfg.out_dir)
    path = out / "calibration.json"
    if path.exists():
        raise FileExistsError(f"{path} is write-once")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("calibration on cuda requires a GPU")
    for g in (cfg.grid, cfg.ce_grid):
        if not g or any(not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0) for s in g):
            raise ValueError("grids must be non-empty lists of positive multipliers")
    if not set(cfg.rules) <= {"lin", "aice"}:
        raise ValueError("only lin/aice are calibrated; CE is the fixed control")
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    tcfg = TrainCfg(preset=cfg.preset, init=cfg.init, dtype=cfg.dtype, device=cfg.device,
                    seq_len=cfg.seq_len, data_dir=cfg.data_dir)
    model, meta = build(tcfg)
    model.eval()
    arr, man = open_split(cfg.data_dir, cfg.split)
    if len(arr) < cfg.docs:
        raise ValueError(f"split {cfg.split} has {len(arr)} rows, needs {cfg.docs}")
    docs = np.asarray(arr[: cfg.docs, : cfg.seq_len + 1])
    kw = dict(batch=cfg.batch, device=cfg.device, fused_attn=cfg.fused_attn)
    print("P2_CALIB_CONFIG " + json.dumps(asdict(cfg), sort_keys=True), flush=True)

    def run(rule, s):
        t0 = time.monotonic()
        r = full_nll(model, docs, rule, cfg.inner_lr * s, cfg.inner_clip, **kw)
        rec = {"rule": rule, "s": s, "mean": float(np.mean(r["doc"])),
               "sd": float(np.std(r["doc"], ddof=1)) if len(r["doc"]) > 1 else 0.0,
               **{k: r[k] for k in ("inner_norm_median", "clip_frac", "update_norm") if k in r},
               "wall_s": time.monotonic() - t0}
        print("P2_CALIB " + json.dumps(rec, sort_keys=True), flush=True)
        return dict(rec, doc=r["doc"])

    ref = run("ce", 0.0)  # no adaptation; identical for every rule
    table = {"ce": {s: run("ce", s) for s in cfg.ce_grid}}
    for rule in cfg.rules:
        table[rule] = {s: run(rule, s) for s in cfg.grid}
    chosen = {rule: select(table[rule], cfg.grid) for rule in cfg.rules}

    def paired(a, b):  # mean and SE of per-document differences a - b
        d = a["doc"] - b["doc"]
        return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d)))
                if len(d) > 1 else 0.0}

    summary = {
        "rule_lr_scale": chosen,
        "edge": {rule: chosen[rule] == min(cfg.grid) for rule in cfg.rules},
        "ce_best_s": select(table["ce"], cfg.ce_grid),
        "noadapt_mean": ref["mean"],
        "chosen_minus_s1": {rule: paired(table[rule][chosen[rule]], table[rule][1])
                            for rule in cfg.rules if 1 in table[rule]},
        "chosen_minus_ce_s1": {rule: paired(table[rule][chosen[rule]], table["ce"][1])
                               for rule in cfg.rules if 1 in table["ce"]},
        "chosen_minus_noadapt": {rule: paired(table[rule][chosen[rule]], ref) for rule in cfg.rules},
    }
    strip = lambda rec: {k: v for k, v in rec.items() if k != "doc"}  # noqa: E731
    record = {
        "amendment": "P2-1/A6", "config": asdict(cfg), "init_label": cfg.init_label,
        "init_sha256": sha256_file(cfg.init) if cfg.init else None, "load_meta": meta,
        "doc_sha1": [m["sha1"] for m in man[: cfg.docs]],
        "noadapt": strip(ref),
        "grid": {rule: [strip(table[rule][s]) for s in sorted(table[rule], reverse=True)]
                 for rule in table},
        "per_doc": {f"{rule}@{s}": table[rule][s]["doc"].tolist()
                    for rule in table for s in table[rule]} | {"noadapt": ref["doc"].tolist()},
        "summary": summary, "wall_s": time.monotonic() - began,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30
        if torch.device(cfg.device).type == "cuda" else 0.0,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, path)
    print("P2_CALIB_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    return {"calibration": str(path), **summary}
