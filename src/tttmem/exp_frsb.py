"""FRS-B orchestration only: the inherited R4 cells training path is unchanged.

Cells are independent, persisted atomically and restartable from their paired
initialisation. B40 selects eta_lr=0 through config. The sealed source and plan
digests prevent silently mixing receipts from different implementations.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
import re
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import torch

from . import exp_ttt as t
from .frsb_analysis import CORE_ARMS, analyze, write_scores
from .frs_seal import execution_guard, protocol_hash, source_paths


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    # Keep JSON standards-compliant even for a divergent trace.
    def clean(v):
        import math
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return None if isinstance(v, float) and not math.isfinite(v) else v
    with temp.open("w") as f:
        json.dump(clean(value), f, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_hash():
    root = Path(__file__).resolve().parents[2]
    paths = source_paths(root)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes())
    return digest.hexdigest()


def validate(cfg, b9=False):
    cells = cfg["cells"]
    if cfg.get("smoke", False):
        if any(c["seed"] != 0 for c in cells):
            raise ValueError("smokes use seed 0 only")
        if int(cfg.get("steps", 0)) > 80:
            raise ValueError("pre-seal smoke step cap is 80")
        if cfg.get("workers", 1) != 1 and cfg.get("device") == "cpu":
            raise ValueError("CPU smoke requires one torch process")
    else:
        if cfg.get("device") != "cuda":
            raise ValueError("fresh-seed production requires CUDA; no local production")
        fixed = dict(d_model=128, n_layers=4, n_heads=4, backbone_mult=4, fast_mult=4,
                     T=512, chunk=64, window=64, steps=1500, batch=32, n_keys=16,
                     n_eval=256, n_val=256, n_quick_eval=64, n_probe=32, eval_batch=32,
                     lr=.001, lr_total=3000, warmup=200, lr_floor=.1,
                     beta1=.9, beta2=.95, weight_decay=.01, g_clip=10., grad_clip=1.,
                     eta_lr=.01, quick_every=25, probe_every=25, tf32=False,
                     checkpoint_steps=[200, 400, 600, 1000])
        st = t._settings(cfg)
        for key, value in fixed.items():
            if getattr(st, key) != value:
                raise ValueError(f"sealed {key}={value} required")
        if len(cfg["seeds"]) != 12 or len(set(cfg["seeds"])) != 12:
            raise ValueError("sealed cohort must have 12 distinct seeds")
        if cfg["seeds"] != list(range(200, 212)):
            raise ValueError("sealed seed cohort must be 200–211")
        core = ("e2e1_b9", "e2e1_lin_b9", "e2e1_lin_b9f") if b9 else CORE_ARMS
        expected = {(block, a, s) for block, aa in (("core", core), ("B40", () if b9 else
                    ("e2e1emb_lin", "e2e1_lin"))) for a in aa for s in cfg["seeds"]}
        if {(c["block"], c["arm"], c["seed"]) for c in cells} != expected or len(cells) != len(expected):
            raise ValueError("sealed cells mismatch")
    for c in cells:
        want_eta, want_lr = (40., 0.) if c["block"] == "B40" else (10., .01)
        if (c["eta_base"], c["eta_lr"]) != (want_eta, want_lr) or c.get("diag", True):
            raise ValueError("cell regime/diag mismatch")
    if cfg.get("threads", 2) > 2 or cfg.get("tf32", True):
        raise ValueError("two-thread cap and TF32 off required")


def effective(cfg, cell):
    return {**cfg, "eta_lr": cell["eta_lr"], "cells": [cell], "workers": 1}


@torch.no_grad()
def frsb_cross_eval(model, task, st, eta):
    """Full fixed set under both streaming rules, with each final's learned eta."""
    dev = next(model.parameters()).device
    raw, meta = t.fixed_set(task, st, "eval")
    data = t.eval_batches(task, raw, meta, dev, st.eval_batch)
    result = {}
    for rule, arm in (("ce", "e2e1"), ("lin", "e2e1_lin")):
        result[rule] = dict(**t.evaluate(model, task, arm, data, meta, eta, st)["eval"],
                            sequences=len(raw), token_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                            device=str(dev), authoritative=dev.type == "cuda")
    return result


def cell_key(cell):
    return f"{cell['block']}__{cell['arm']}__s{cell['seed']}"


def receipt_files(record):
    return [record[k] for k in ("checkpoint", "trace") if record.get(k)] + record.get("step_checkpoints", [])


def resolve_output_file(folder, relative):
    root = Path(folder).resolve()
    file = (root / relative).resolve()
    if not file.is_relative_to(root):
        raise ValueError("receipt file escapes output root")
    return file


def verify_receipt(record, folder):
    hashes = record.get("file_sha256", {})
    if not record.get("checkpoint") or not record.get("trace"):
        raise ValueError("receipt file sha256 missing")
    for relative in receipt_files(record):
        file = resolve_output_file(folder, relative)
        if relative not in hashes or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != hashes[relative]:
            raise ValueError("receipt file sha256 mismatch; completed cells must not be retrained")


@contextmanager
def training_log(cfg, directory):
    if cfg.get("smoke", False):
        yield
    else:
        with (directory / "train.log").open("a", buffering=1) as log:
            import sys
            public = sys.stdout
            started = time.monotonic()
            class HealthLog:
                pending = ""
                def write(self, text):
                    log.write(text)
                    self.pending += text
                    while "\n" in self.pending:
                        line, self.pending = self.pending.split("\n", 1)
                        match = re.search(r" step (\d+/\d+) train=(\S+)", line)
                        if match:
                            import math
                            finite = math.isfinite(float(match[2]))
                            print(f"FRSB_HEALTH cell={directory.name} steps={match[1]} loss_finite={finite} elapsed_s={time.monotonic()-started:.1f}", file=public, flush=True)
                    return len(text)
                def flush(self):
                    log.flush()
            with redirect_stdout(HealthLog()), redirect_stderr(log):
                yield


def run_one(cfg, cell, folder, manifest):
    folder = Path(folder).resolve()
    key = cell_key(cell)
    directory = Path(folder)/"cells"/key
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/"receipt.json"
    # One persisted infrastructure retry; numerical divergence is a result.
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior["identity"] != manifest["identity"]:
            raise ValueError("receipt seal mismatch; choose a new output directory")
        if prior["record"].get("trained") or prior["attempts"] >= 2:
            if prior["record"].get("trained"):
                verify_receipt(prior["record"], folder)
            print(f"FRSB_REUSE {key}", flush=True)
            return prior["record"]
        attempts = prior["attempts"]
    else:
        attempts = 0
    previous_cwd = Path.cwd()
    record = {}
    try:
        os.chdir(directory)
        while attempts < 2:
            attempts += 1
            atomic_json(receipt, dict(identity=manifest["identity"], attempts=attempts,
                                      record=dict(cell, error="interrupted infrastructure attempt")))
            local = effective(cfg, cell)
            st = t._settings(local)
            dev = t._device(st.device)
            try:
                with training_log(cfg, directory):
                    out = t.run_cells(local, st, dev)
                record = next(iter(out["cells"].values()))
            except Exception as exc:
                with training_log(cfg, directory):
                    traceback.print_exc()
                record = dict(cell, error=type(exc).__name__)
            record.update(block=cell["block"], eta_lr=cell["eta_lr"], attempts=attempts)
            if record.get("trained"):
                for cross_attempt in range(2):
                    try:
                        model = t.build_model(t._task(st, "echo"), st, dev, cell["seed"], cell["arm"])
                        model.load_state_dict(torch.load(record["checkpoint"], map_location=dev, weights_only=True))
                        model.eval()
                        with training_log(cfg, directory):
                            record["cross_eval"] = frsb_cross_eval(model, t._task(st, "echo"), st, cell["eta_base"])
                        record.pop("cross_eval_error", None)
                        break
                    except Exception as exc:
                        with training_log(cfg, directory):
                            traceback.print_exc()
                        record["cross_eval_error"] = type(exc).__name__
                for path_field in ("checkpoint", "trace"):
                    record[path_field] = str(Path(record[path_field]).resolve().relative_to(folder))
                record["step_checkpoints"] = [str(Path(p).resolve().relative_to(folder)) for p in record["step_checkpoints"]]
                record["file_sha256"] = {p: hashlib.sha256(resolve_output_file(folder, p).read_bytes()).hexdigest()
                                          for p in receipt_files(record)}
            atomic_json(receipt, dict(identity=manifest["identity"], attempts=attempts, record=record))
            if record.get("trained"):
                break
            print(f"FRSB_INFRA_RETRY {key} attempts={attempts}", flush=True)
    finally:
        os.chdir(previous_cwd)
    return record


def worker(index, cfg, folder, manifest, jobs, results, lock):
    import sys
    sys.stdout, sys.stderr = t._LockedLines(sys.stdout, lock), t._LockedLines(sys.stderr, lock)
    t._mark_process(f"{cfg['process_name']}-w{index}")
    torch.set_num_threads(cfg.get("threads", 2))
    t.configure_tf32(False, str(index))
    while True:
        cell = jobs.get()
        if cell is None:
            return
        try:
            record = run_one(cfg, cell, folder, manifest)
        except Exception as exc:
            print(f"FRSB_ERROR cell={cell_key(cell)} error={type(exc).__name__}", flush=True)
            record = dict(cell, error=type(exc).__name__)
        results.put((cell_key(cell), record))
        print(f"FRSB_PROGRESS cell={cell_key(cell)} complete={bool(record.get('trained'))}", flush=True)


def execute(cfg, manifest, folder):
    cells = cfg["cells"]
    if cfg["workers"] == 1:
        result = {}
        for i, cell in enumerate(cells):
            result[cell_key(cell)] = run_one(cfg, cell, folder, manifest)
            print(f"FRSB_PROGRESS completed={i+1}/{len(cells)}", flush=True)
        return result
    ctx = mp.get_context("spawn")
    jobs, results, lock = ctx.Queue(), ctx.Queue(), ctx.Lock()
    for cell in sorted(cells, key=lambda c: c["arm"] != "e2e2"):
        jobs.put(cell)
    for _ in range(cfg["workers"]):
        jobs.put(None)
    processes = [ctx.Process(name=f"{cfg['process_name']}-w{i}", target=worker,
                            args=(i, cfg, folder, manifest, jobs, results, lock))
                 for i in range(cfg["workers"])]
    for p in processes:
        p.start()
    out = {}
    deadline = time.monotonic()+cfg.get("max_minutes", 480)*60
    while len(out) < len(cells):
        try:
            key, record = results.get(timeout=30)
            out[key] = record
        except queue.Empty:
            if time.monotonic() >= deadline:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break
            if not any(p.is_alive() for p in processes):
                break
    for p in processes:
        p.join(timeout=30)
    for cell in cells:
        out.setdefault(cell_key(cell), dict(cell, error="worker exited without a receipt"))
    return out


def prepare(cfg, b9=False):
    validate(cfg, b9)
    seal_sha = execution_guard(cfg, b9=b9)
    t._mark_process(cfg["process_name"])
    torch.set_num_threads(cfg.get("threads", 2))
    t.configure_tf32(False, "main")
    folder = Path(cfg["output_dir"]).resolve()
    root = Path(__file__).resolve().parents[2]
    plan = root / cfg["plan_path"] if cfg.get("plan_path") else None
    manifest = dict(schema=1, smoke=cfg.get("smoke", False), seeds=cfg["seeds"], config=cfg,
                    plan_sha256=None if plan is None else hashlib.sha256(plan.read_bytes()).hexdigest(),
                    source_sha256=source_hash())
    manifest["config_sha256"] = protocol_hash(cfg)
    manifest["seal_sha256"] = seal_sha
    manifest["identity"] = hashlib.sha256(json.dumps({k: manifest[k] for k in
                                          ("plan_sha256", "source_sha256", "config_sha256")}, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"manifest.json"
    if path.exists() and json.loads(path.read_text())["identity"] != manifest["identity"]:
        raise ValueError("output directory belongs to another seal")
    atomic_json(path, manifest)
    print("FRSB_MANIFEST " + json.dumps(manifest), flush=True)
    return manifest, folder


def run(cfg, seed=0):
    manifest, folder = prepare(cfg)
    out = dict(manifest=manifest, cells=execute(cfg, manifest, folder))
    scores = analyze(out, smoke_only=cfg.get("smoke", False))
    out["family_inputs"] = scores["family_inputs"]
    out["replication_inputs"] = scores["replication_inputs"]
    atomic_json(folder/"result.json", out)
    write_scores(folder/"scores", scores)
    # During production the only scientific field shown to the orchestrator is this bit.
    print("B9_GATE " + json.dumps(scores["b9_gate"]), flush=True)
    summary = dict(cells=len(out["cells"]), complete=sum(bool(r.get("trained")) for r in out["cells"].values()),
                   output_dir=str(folder), smoke=manifest["smoke"], b9_gate=scores["b9_gate"],
                   plan_sha256=manifest["plan_sha256"])
    print("FRSB_SUMMARY " + json.dumps(summary), flush=True)
    return summary
