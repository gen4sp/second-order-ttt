"""Single-machine launch rehearsal. Real checkpoint/data; outputs never count as evidence for G2."""
from __future__ import annotations

from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import decide
from .data import PrepareCfg, prepare_config
from .train import ARMS, TrainCfg, train


def smoke(cfg):
    unknown = set(cfg) - {"mode", "process_name", "init", "data_dir", "out_dir", "prepare", "device"}
    if unknown:
        raise ValueError(f"unknown smoke keys: {sorted(unknown)}")
    started = time.monotonic()
    root = Path(cfg.get("out_dir", "outputs/p21-smoke"))
    if root.exists():
        raise FileExistsError("smoke requires a fresh out_dir")
    root.mkdir(parents=True)
    data = Path(cfg.get("data_dir", "data/p21-smoke"))
    if not (data / "meta.json").exists():
        p = PrepareCfg.from_dict(cfg["prepare"])
        prepare_config(replace(p, out_dir=str(data)))
    device = cfg.get("device", "cuda")
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("GPU smoke requires CUDA; tiny CPU correctness lives in tests")
    outputs, peaks = {}, {}
    base = TrainCfg(init=cfg["init"], device=device, data_dir=str(data), global_batch=2,
                    steps=3, warmup=30, n_eval=8, trajectory_docs=8,
                    ho_docs=8, ho_trajectory_docs=8, pilot_docs=8,
                    eval_batch=4, assay_gate=False, eval_steps=[2, 3], full_eval_steps=[0],
                    sizing_steps=[2, 3], pilot_steps=[1, 2, 3], resume_every=2,
                    crossover_rules=[], log_every=1, save_checkpoints=False)
    # P2-1's coordinator is frozen to its four registered arms.  MINI has its
    # own four-arm CPU rehearsal in ``p2.mini``.
    for arm in ("e2e2", "e2e1-ce", "e2e1-lin", "e2e1-aice"):
        torch.cuda.reset_peak_memory_stats()
        arm_root = root / arm
        conf = replace(base, arm=arm, micro_batch=1 if arm == "e2e2" else 2,
                       out_dir=str(arm_root))
        result = train(conf)
        if result["aborted"]:
            raise RuntimeError(f"smoke arm aborted: {arm}: {result}")
        with np.load(arm_root / "eval_step0003.npz") as z:
            if len(z["doc_adapt"]) != 8:
                raise ValueError("smoke requires eight eligible trajectory documents")
        peaks[arm] = torch.cuda.max_memory_allocated() / 2**30
        outputs[arm] = arm_root
        print("P2_SMOKE_ARM " + json.dumps(dict(arm=arm, peak_gb=peaks[arm], **result)), flush=True)
        # Retain only SO's final state for comparison; two full AdamW states per
        # arm would otherwise accumulate >26 GB in this short multi-arm rehearsal.
        for path in arm_root.glob("resume_step*.pt"):
            if arm != "e2e2" or path.name != "resume_step0003.pt":
                path.unlink()
        gc.collect()
        torch.cuda.empty_cache()
    # Identical horizon (3), data and LR: stop at 2, restore, execute the last step.
    split = root / "resume-check"
    train(replace(base, arm="e2e2", micro_batch=1, out_dir=str(split), stop_after=2))
    train(replace(base, arm="e2e2", micro_batch=1, out_dir=str(split), resume=True))
    # CUDA backward kernels are not bitwise deterministic and the first AdamW steps
    # (m / sqrt(v) ~ +-1) amplify that noise to O(lr) per element (observed in a smoke run:
    # identical step-1 loss, grad norms differing at 1e-5, parameters at 9e-4). A second
    # uninterrupted run measures that noise floor; the resumed path must stay within it.
    twin = root / "resume-twin"
    train(replace(base, arm="e2e2", micro_batch=1, out_dir=str(twin)))
    a = torch.load(outputs["e2e2"] / "resume_step0003.pt", map_location="cpu", weights_only=False)["model"]
    n = sum(v.numel() for v in a.values())

    def gap(path):
        b = torch.load(path, map_location="cpu", weights_only=False)["model"]
        same = all(torch.equal(a[k], b[k]) for k in a)
        mx = max(float((a[k] - b[k]).abs().max()) for k in a)
        mean = sum(float((a[k] - b[k]).abs().sum()) for k in a) / n
        return same, mx, mean

    identical, max_abs, mean_abs = gap(split / "resume_step0003.pt")
    _, noise_max, noise_mean = gap(twin / "resume_step0003.pt")
    del a

    def lrs(d):
        # evidence lines are "P2_STEP {json}"
        return [json.loads(x.split(" ", 1)[1])["lr"] for x in (d / "steps.jsonl").read_text().splitlines()]

    same_lr = lrs(split) == lrs(outputs["e2e2"]) == lrs(twin)
    resume = dict(identical=identical, max_abs=max_abs, mean_abs=mean_abs, noise_max_abs=noise_max,
                  noise_mean_abs=noise_mean, same_lr=same_lr)
    print("P2_SMOKE_RESUME " + json.dumps(resume), flush=True)
    if not same_lr or max_abs > max(1e-5, 3 * noise_max) or mean_abs > max(1e-7, 3 * noise_mean):
        raise ValueError(f"smoke resume mismatch beyond the CUDA noise floor: {resume}")
    for path in [split / "resume_step0002.pt", *twin.glob("resume_step*.pt")]:
        path.unlink()
    # Exercise the pilot's restart and final-stop branches on the real model.
    for attempt in (0, 1):
        r = train(replace(base, arm="e2e2", micro_batch=1, attempt=attempt,
                          lr_peak=1e-3 if attempt else 3e-3, pilot_steps=[1],
                          pilot_max_rise=-1e9, out_dir=str(root / f"pilot-trip-{attempt}")))
        if not r["aborted"] or r.get("final", False) != bool(attempt):
            raise RuntimeError("pilot path did not produce its specified decision")
        for path in (root / f"pilot-trip-{attempt}").glob("resume_step*.pt"):
            path.unlink()
    # Adapt actual smoke step labels for the production coordinator's fixed time points.
    # These directories and every decision remain labelled smoke; no scientific gate is inferred.
    for arm, source in outputs.items():
        dest = root / "coordinator-smoke" / arm
        dest.mkdir(parents=True)
        shutil.copy2(source / "eval_meta.npz", dest / "eval_meta.npz")
        for old, new in ((0, 0), (2, 60), (3, 100), (3, 400)):
            shutil.copy2(source / f"eval_step{old:04d}.npz", dest / f"eval_step{new:04d}.npz")
        outputs[arm] = dest
    decisions = root / "coordinator-smoke" / "decisions"
    assay = decide.assay_decision(outputs["e2e2"], n_boot=100, min_clusters=2)
    fut = decide.futility(outputs["e2e2"], outputs["e2e1-ce"], decisions, expected_docs=8)
    size = decide.sizing(outputs, decision_dir=decisions, Ns=[8], n_sim=100, ho_n=8, smoke=True)
    gate = decide.endpoint(outputs, decisions, n_boot=100, min_clusters=2)
    summary = dict(smoke=True, scientific_verdict=False, peaks_gb=peaks,
                   resume=resume, assay=assay,
                   sizing=size, futility=fut, endpoint=gate, wall_s=time.monotonic() - started)
    (root / "smoke_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print("P2_SMOKE " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
