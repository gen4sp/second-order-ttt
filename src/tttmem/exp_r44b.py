"""Registered wave B: barriers, immutable inputs, and complete Kamin continuations.

No generic sweep fallback. CUDA execution is deliberately separate from tiny CPU
unit checks of these functions. A seed group is the admission/budget-stop unit.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from . import exp_ttt as t
from . import exp_p01 as p

PROJECT = "second-order-ttt"
ECHO = ("e2e1emb", "e2e1emb_lin", "e2e1emb_aice")
ICL = ("e2e1", "e2e1_aice")
TWINS = ("kamin2", "kaminlin2", "kaminctl2")
RULE_ARM = dict(kamin2="e2e1emb", kaminlin2="e2e1emb_linkv", kaminctl2="e2e1emb")


class MemorySizingFailure(RuntimeError):
    pass


def finite(x):
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [finite(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def emit(kind, **row):
    print("R44B_" + kind + " " + json.dumps(finite(row), allow_nan=False), flush=True)


@dataclasses.dataclass
class WaveB:
    enabled: bool = True
    smoke: bool = False
    preflight_workers: int = 2
    memory_probe_workers: int = 6
    memory_probe_steps: int = 2
    max_device_usage_gib: float = 14.0
    echo_seeds: list = dataclasses.field(default_factory=lambda: list(range(10)))
    echo_arms: list = dataclasses.field(default_factory=lambda: list(ECHO))
    echo_eta_base: float = 10.0
    echo_steps: int = 1500
    icl_seeds: list = dataclasses.field(default_factory=lambda: list(range(6)))
    icl_arms: list = dataclasses.field(default_factory=lambda: list(ICL))
    icl_eta_base: float = .3
    icl_steps: int = 4000
    kamin_seeds: list = dataclasses.field(default_factory=lambda: list(range(10)))
    kamin_smoke_fixed_switch: int = 0
    kamin_stage1_cap: int = 2000
    kamin_stage2_steps: int = 1000
    kamin_stage2_warmup: int = 25
    kamin_probe_every: int = 25
    kamin_trigger_sequences: int = 32
    kamin_audit_sequences: int = 256
    kamin_cap_failure_smoke_steps: int = 10
    reassign_sequences: int = 256
    reassign_eval_seed: int = 20250308
    query_update_freeze: bool = True
    write_snapshot_eval: bool = True
    historical_lin_audit: bool = True
    checkpoint_input_manifest: str = "configs/toy/r4-4b/historical-manifest.json"
    output_dir: str = "checkpoints/r44b"
    # Shared by the smoke and the full run, outside the run's output directory.
    smoke_receipt_path: str = "outputs/r44b-podsmoke/smoke-receipt.json"
    lock_path: str = "outputs/TTM-r44b.lock"


def validate_config(cfg):
    if "wave_b" not in cfg or not isinstance(cfg["wave_b"], dict):
        raise ValueError("wave B requires [ttt.wave_b]; a plain cells/sweep config is refused")
    known = {f.name for f in dataclasses.fields(WaveB)}
    unknown = set(cfg["wave_b"]) - known
    if unknown:
        raise ValueError(f"unknown wave_b keys: {sorted(unknown)}")
    w = WaveB(**cfg["wave_b"])
    if w.enabled is not True:
        raise ValueError("wave_b.enabled must be true; a plain cells/sweep config is refused")
    settings = {f.name for f in dataclasses.fields(t.Settings)} - {"tables"}
    if set(cfg) - settings - {"wave_b", "icl_mc"}:
        raise ValueError("unknown ttt keys or plain cells/sweep in wave B")
    if set(cfg.get("icl_mc", {})) - settings:
        raise ValueError("unknown icl_mc settings")
    for f in dataclasses.fields(w):
        val, default = getattr(w, f.name), getattr(WaveB(), f.name)
        if type(val) is not type(default) and not (type(default) is float and type(val) is int):
            raise ValueError(f"wave_b.{f.name}: wrong type")
    for key in ("echo_seeds", "icl_seeds", "kamin_seeds"):
        if any(type(s) is not int for s in getattr(w,key)):
            raise ValueError(f"wave_b.{key}: integer seeds required")
    if w.echo_arms != list(ECHO) or w.icl_arms != list(ICL):
        raise ValueError("registered arms/order must be preserved")
    st, icl = t._settings(cfg), t._settings(cfg, "icl_mc")
    common = dict(d_model=128, n_layers=4, n_heads=4, backbone_mult=4, fast_mult=4,
                  window=64, chunk=64, batch=32, lr=.001, eta_lr=.01, beta1=.9,
                  beta2=.95, weight_decay=.01, warmup=200, lr_floor=.1, grad_clip=1.,
                  g_clip=10., tf32=False, spike_skip=0., late_frac=.25, eval_batch=32)
    for key, value in common.items():
        if getattr(st, key) != value or getattr(icl, key) != value:
            raise ValueError(f"registered setting {key}={value} required for both tasks")
    for settings_obj, expected in ((st, dict(T=512, n_keys=16, lr_total=3000)),
                                   (icl, dict(T=1024, n_vocab=32, alpha=.1, lr_total=4000,
                                              probe_every=0))):
        for key, value in expected.items():
            if getattr(settings_obj, key) != value:
                raise ValueError(f"task setting {key}={value} required")
    if st.workers not in (4, 6) or st.device not in ("auto", "cuda"):
        raise ValueError("wave B requires CUDA and four or six workers")
    if (w.echo_eta_base, w.icl_eta_base, w.kamin_stage2_warmup, w.reassign_eval_seed) != (10., .3, 25, 20250308):
        raise ValueError("registered eta/clock/eval seed required")
    if not all((w.query_update_freeze, w.write_snapshot_eval, w.historical_lin_audit)):
        raise ValueError("required diagnostics/audits disabled")
    if w.max_device_usage_gib != 14 or not 1 <= w.preflight_workers <= 2 or w.memory_probe_workers != 6 or w.memory_probe_steps < 2:
        raise ValueError("registered memory preflight required")
    if w.smoke:
        expected = dict(echo_seeds=[0], icl_seeds=[0], kamin_seeds=[0], echo_steps=60,
                        icl_steps=100, kamin_smoke_fixed_switch=30, kamin_stage2_steps=30,
                        kamin_probe_every=10, kamin_trigger_sequences=2,
                        kamin_audit_sequences=8, reassign_sequences=16,
                        kamin_cap_failure_smoke_steps=10)
        if st.process_name != "TTM-r44b-01-podsmoke" or st.max_minutes > 20:
            raise ValueError("smoke identity/deadline mismatch")
        task_values = ((st, dict(steps=60, n_eval=16, n_quick_eval=8, quick_every=10,
                                 probe_every=10, n_probe=2)),
                       (icl, dict(steps=100, n_eval=16, n_quick_eval=8, quick_every=50)))
    else:
        expected = dict(echo_seeds=list(range(10)), icl_seeds=list(range(6)),
                        kamin_seeds=list(range(10)), echo_steps=1500, icl_steps=4000,
                        kamin_smoke_fixed_switch=0, kamin_stage1_cap=2000,
                        kamin_stage2_steps=1000, kamin_probe_every=25,
                        kamin_trigger_sequences=32, kamin_audit_sequences=256,
                        reassign_sequences=256)
        if st.process_name != "TTM-r44b-02" or st.max_minutes != 230:
            raise ValueError("full identity/deadline mismatch")
        task_values = ((st, dict(steps=1500, n_eval=256, n_quick_eval=64, quick_every=25,
                                 probe_every=25, n_probe=32)),
                       (icl, dict(steps=4000, n_eval=512, n_quick_eval=64, quick_every=250)))
    for key, value in expected.items():
        if getattr(w, key) != value:
            raise ValueError(f"wave_b.{key} must be {value}")
    for settings_obj, values in task_values:
        for key, value in values.items():
            if getattr(settings_obj, key) != value:
                raise ValueError(f"task {key} must be {value}")
    return w


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(value):
    """Stable recursive digest (tensor bytes + dtype/shape, no pickle identity)."""
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            v = v.detach().cpu().contiguous()
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype.str, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(type(v)).encode())
            for x in v:
                add(x)
        else:
            h.update(repr(v).encode() + b"\0")
    add(value)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(finite(value), f, allow_nan=False, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save_snapshot(path, model, opt, rng, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, model=model.state_dict(), optimizer=opt.state_dict(),
                   numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                   python_rng=random.getstate(), metadata=metadata,
                   regime=dict(inner_multiplier=model.inner_multiplier, inner_sign=model.inner_sign))
    payload["state_sha256"] = state_hash(payload)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    entry = dict(path=str(path.resolve()), sha256=sha_file(path), state_sha256=payload["state_sha256"],
                 metadata=metadata)
    atomic_json(path.with_suffix(".json"), entry)
    return entry


def load_snapshot(entry, model, opt, rng):
    if sha_file(entry["path"]) != entry["sha256"]:
        raise ValueError("continuation file checksum mismatch")
    payload = torch.load(entry["path"], map_location=next(model.parameters()).device, weights_only=False)
    checksum = payload.pop("state_sha256")
    if payload.get("version") != 1 or state_hash(payload) != checksum or checksum != entry["state_sha256"]:
        raise ValueError("incomplete or corrupt continuation")
    model.load_state_dict(payload["model"], strict=True)
    opt.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])
    random.setstate(payload["python_rng"])
    for k, v in payload["regime"].items():
        setattr(model, k, v)
    return payload["metadata"]


def verify_historical(path, cfg, dev):
    manifest = json.loads(Path(path).read_text())
    entries = manifest["checkpoints"]
    want = {(task, seed) for task, n in (("echo", 10), ("icl_mc", 6)) for seed in range(n)}
    if {(e["task"], e["seed"]) for e in entries} != want or len(entries) != 16:
        raise ValueError("historical cohort incomplete/duplicate")
    for e in entries:
        if sha_file(e["path"]) != e["sha256"]:
            raise ValueError(f"historical input hash mismatch: {e['task']}/s{e['seed']}")
        st = t._settings(cfg, e["task"]); task = t._task(st, e["task"])
        model = t.build_model(task, st, dev, e["seed"], e["arm"])
        model.load_state_dict(torch.load(e["path"], map_location=dev, weights_only=True), strict=True)
        if (model.frozen_emb is not None) != (e["task"] == "echo"):
            raise ValueError("historical code-buffer mismatch")
        del model
    return entries


class Trigger:
    def __init__(self, threshold=.99):
        self.threshold, self.records = threshold, []

    def observe(self, step, mean):
        self.records.append(dict(step=step, mean=mean))
        return len(self.records) >= 2 and all(np.isfinite(r["mean"]) and
                   r["mean"] >= self.threshold for r in self.records[-2:])


def set_clock(opt, step, st, stage2=False):
    scale = t.lr_scale(step, 25 if stage2 else st.warmup,
                       1000 if stage2 else st.lr_total, st.lr_floor)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * scale
    return [g["lr"] for g in opt.param_groups]


def continuation_check(entry, model, opt, rng, task, st, eta):
    """Smoke-only S5 replay; restore the switch snapshot after both paths."""
    dev = next(model.parameters()).device
    initial = state_hash(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng.bit_generator.state))
    def predict_next():
        preview = np.random.default_rng()
        preview.bit_generator.state = rng.bit_generator.state
        raw, meta = task.raw(preview, st.batch)
        idx, y, meta = task.tensors(raw, meta, dev)
        model.inner_multiplier = 1.
        with torch.no_grad():
            z = t.eval_batch_logits(model, task, "e2e1emb", idx, y, eta, st.g_clip,
                                    task.chunk, t.GradStats(), meta=meta)
        return z.detach().cpu(), state_hash(raw)
    try:
        z1, batch1 = predict_next()
        raw1, ok1, _, _ = update(model, opt, task, st, rng, "e2e1emb", eta, 0, stage2=True)
        end1 = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        grads1 = [None if v.grad is None else v.grad.detach().cpu().clone() for v in model.parameters()]
        load_snapshot(entry, model, opt, rng)
        exact = initial == state_hash(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng.bit_generator.state))
        z2, batch2 = predict_next()
        raw2, ok2, _, _ = update(model, opt, task, st, rng, "e2e1emb", eta, 0, stage2=True)
        max_logit = float((z1-z2).abs().max())
        max_update = max(float((v.detach().cpu()-end1[k]).abs().max()) for k,v in model.state_dict().items())
        grads_match = True
        for old, param in zip(grads1, model.parameters()):
            if old is None or param.grad is None:
                grads_match = grads_match and old is None and param.grad is None
            else:
                grads_match = grads_match and bool(torch.allclose(param.grad.detach().cpu(), old, atol=1e-6, rtol=1e-4))
        # Recorded, not asserted, so a failed smoke replay reports which check failed.
        checks = dict(cloned_state_exact=exact, updates_finite=ok1 and ok2,
                      next_batch_exact=batch1 == batch2 and bool(np.array_equal(raw1, raw2)),
                      logits_within_tolerance=max_logit <= 1e-4, update_within_tolerance=max_update <= 1e-4,
                      grads_within_tolerance=grads_match)
        return dict(passed=all(checks.values()), **checks, next_batch_sha256=batch1,
                    max_logit_diff=max_logit, max_update_diff=max_update)
    finally:
        load_snapshot(entry,model,opt,rng)


def update(model, opt, task, st, rng, arm, eta, step, *, stage1=False, masked=False, stage2=False):
    lrs = set_clock(opt, step, st, stage2)
    raw, meta = task.raw(rng, st.batch)
    idx, y, meta = task.tensors(raw, meta, next(model.parameters()).device)
    model.inner_multiplier = 0. if stage1 else 1.
    opt.zero_grad(set_to_none=True)
    rho = model.rho.detach().clone() if stage1 else None
    sel = t.lin_selector(arm, meta.get("write_pos"), len(idx), task.T, idx.device)
    if masked:
        h = model.backbone(idx, t.copy_edge_mask(meta["write_pos"], task.T, idx.device))
        loss = t.sequential_inner(model, h, y, model.per_sequence(len(idx)),
                                  t.regime_etas(model, eta), st.g_clip, False,
                                  t.GradStats(), task.chunk, idx, rule="ce")
    else:
        loss = t.arm_loss(model, arm, idx, y, eta, st.g_clip, task.chunk,
                          t.GradStats(), sel=sel, meta=meta)
    if not bool(torch.isfinite(loss)):
        return raw, False, float(loss.detach()), lrs
    loss.backward()
    if stage1:
        model.rho.grad = None
    norm = torch.nn.utils.clip_grad_norm_(t.clipped_params(model), st.grad_clip)
    rho_norm = model.rho.grad.norm() if model.rho.grad is not None else norm.new_zeros(())
    ok = bool(torch.isfinite(torch.stack([norm, rho_norm])).all())
    if ok:
        opt.step()
    if stage1 and not torch.equal(model.rho.detach(), rho):
        raise AssertionError("rho changed during stage 1")
    return raw, ok, float(loss.detach()), lrs


@torch.no_grad()
def score(model, task, st, arm, eta, which="eval"):
    dev = next(model.parameters()).device
    raw, meta = t.fixed_set(task, st, which)
    parts = t._eval_parts(model, task, arm, t.eval_batches(task, raw, meta, dev, st.eval_batch),
                          eta, st, t.GradStats())
    ce, pred = (torch.cat(xs) for xs in zip(*parts))
    out = dict(ce=float(ce.double().mean()), ce_late=float(ce[:, int(.75*task.T):].double().mean()),
               token_sha256=state_hash(raw), sequences=len(raw), T=task.T, V=task.V)
    if meta:
        pos = torch.as_tensor(meta["query_pos"], device=dev)
        targets = torch.as_tensor(meta["query_tok"], device=dev)
        hits = int((pred.gather(1, pos) == targets).sum().cpu())
        out.update(query_hits=hits, query_events=targets.numel(), query_acc=hits/targets.numel(),
                   query_ce=float(ce.gather(1, pos).double().mean()))
    return out


@torch.no_grad()
def copy_audit(model, task, st, n, split, masked=False):
    raw, meta = task.raw(np.random.default_rng([t.PROBE_SEED, split]), n)
    values = []
    for idx, y, m in t.eval_batches(task, raw, meta, next(model.parameters()).device, st.eval_batch):
        mask = t.copy_edge_mask(m["write_pos"], task.T, idx.device) if masked else None
        h = model.backbone(idx, mask)
        z = model.read(model.fast_block(h, idx, model.per_sequence(len(idx))))
        prob = z.softmax(-1).gather(-1, y[..., None]).squeeze(-1)
        values.append(p.gather(prob, m["write_pos"]).detach().cpu().double())
    a = torch.cat(values)
    return dict(mean=float(a.mean()), quantiles=torch.quantile(a.flatten(), torch.tensor(
                [0., .1, .5, .9, 1.], dtype=torch.float64)).tolist(),
                per_sequence=a.mean(1).tolist(), events=a.numel(), token_sha256=state_hash(raw))


def train_cell(cfg, w, job, dev):
    name, seed = job["task"], job["seed"]
    label = job["arm"]
    arm = RULE_ARM.get(label, label)
    st = t._settings(cfg, name); task = t._task(st, name)
    if job.get("memory"):
        st = dataclasses.replace(st, n_eval=32, n_quick_eval=32, quick_every=1000, probe_every=0)
        task = dataclasses.replace(task, n_eval=32)
    eta = w.icl_eta_base if name == "icl_mc" else w.echo_eta_base
    model = t.build_model(task, st, dev, seed, arm)
    opt = t.build_optim(model, st)
    rng = np.random.default_rng([seed, t.TRAIN_SEED])
    initial = dict(initial_sha256=state_hash(model.state_dict()), code_sha256=state_hash(model.frozen_emb))
    if name == "icl_mc" and model.frozen_emb is not None:
        raise AssertionError("plain icl_mc arm has frozen code")
    stage = job.get("stage", "fresh")
    stage1, masked, stage2 = stage in ("stage1", "masked"), stage == "masked", stage == "stage2"
    stream = hashlib.sha256()
    rolling, quick, probes = [], [], []
    meta = dict(completed_steps=0, rolling_data=[], first_batch_sha1=None)
    if stage2:
        meta = load_snapshot(job["snapshot"], model, opt, rng)
        clone = dict(model_sha256=state_hash(model.state_dict()), optimizer_sha256=state_hash(opt.state_dict()),
                     rng_sha256=state_hash(rng.bit_generator.state), snapshot_sha256=job["snapshot"]["sha256"])
        emit("CLONE", seed=seed, arm=label, **clone)
        model.inner_multiplier = 1.
        clone["audit"] = copy_audit(model, task, st, w.kamin_audit_sequences, 1)
        boundary = score(model, task, st, arm, eta, "quick")
        quick.append(dict(local_step=0, global_step=meta["completed_steps"], **boundary))
    if masked:
        steps = job["duration"]
    elif stage == "stage1":
        steps = job.get("cap", w.kamin_stage1_cap)
    elif stage2:
        steps = w.kamin_stage2_steps
    else:
        steps = job.get("steps", w.icl_steps if name == "icl_mc" else w.echo_steps)
    root = Path(w.output_dir)
    ident = f"{name}__s{seed}__{label}__{stage}"
    trigger, switched, failure = Trigger(), False, None
    first_batch = None
    start = time.perf_counter()
    train_seconds = 0.
    forced = stage == "stage1" and w.smoke and not job.get("cap_failure", False)
    successful = 0
    for u in range(steps):
        tick = time.perf_counter()
        raw, ok, loss, lrs = update(model, opt, task, st, rng, arm, eta, u,
                                   stage1=stage1, masked=masked, stage2=stage2)
        train_seconds += time.perf_counter()-tick
        if first_batch is None:
            first_batch = hashlib.sha1(raw[:, :task.T].tobytes()).hexdigest()
        stream.update(raw.tobytes())
        if not ok:
            failure = "algorithmic_nonfinite"; break
        successful += 1
        completed = u + 1
        global_step = meta["completed_steps"] + completed if stage2 else completed
        if completed % 25 == 0 or completed == steps:
            rolling.append(dict(step=completed, sha256=stream.hexdigest()))
            emit("DATA", cell_id=ident, step=completed, sha256=stream.hexdigest())
        if stage == "stage1":
            every = 5 if job.get("cap_failure") else w.kamin_probe_every
            if completed % every == 0:
                audit = copy_audit(model, task, st, w.kamin_trigger_sequences, 0)
                crossed = trigger.observe(completed, audit["mean"])
                emit("TRIGGER", seed=seed, cap_failure=job.get("cap_failure", False), **audit,
                     step=completed, confirmed=crossed, rho_unchanged=True)
                switched = crossed
            if forced and completed == w.kamin_smoke_fixed_switch:
                switched = True
            if switched:
                break
        quick_every = (w.kamin_probe_every if w.smoke else 25) if stage2 else st.quick_every
        if not stage1 and (completed % quick_every == 0 or completed == steps):
            q = dict(local_step=completed, global_step=global_step, **score(model, task, st, arm, eta, "quick"),
                     lrs=lrs, eta=[float(e.detach()) for e in model.etas(eta)])
            quick.append(q); emit("QUICK", cell_id=ident, **q)
        if name == "echo" and not stage1 and st.probe_every and completed % st.probe_every == 0:
            data = t.eval_batches(task, *task.raw(np.random.default_rng([t.PROBE_SEED, 0]), st.n_probe), dev, st.eval_batch)
            probe = dict(local_step=completed, global_step=global_step, lrs=lrs,
                         **t.r4_probe(model, task, data, eta, st.g_clip, arm=arm))
            if stage2:
                probe["copy_Wref"] = copy_audit(model, task, st, w.kamin_trigger_sequences, 0)
                probe["ordinary_kv_residual"] = 1-probe["copy_Wref"]["mean"]
            probes.append(probe); emit("PROBE", cell_id=ident, **probe)
        save = {25, 100, 250, 500} if stage2 else ({200, 400, 600, 1000} if arm in ("e2e1emb_aice", "e2e1emb_lin") and name == "echo" else set())
        if completed in save:
            save_snapshot(root/f"{ident}__st{completed}.pt", model, opt, rng,
                          dict(seed=seed, arm=label, stage=stage, completed_steps=global_step,
                               local_step=completed, rolling_data=rolling,
                               first_batch_sha1=first_batch,
                               eval_sha256=state_hash(t.fixed_set(task, st, "eval")[0]),
                               switch=meta.get("switch"),
                               quick_sha256=state_hash(t.fixed_set(task, st, "quick")[0])))
    if not rolling or rolling[-1]["step"] != successful:
        rolling.append(dict(step=successful, sha256=stream.hexdigest()))
    switched = switched and failure is None
    audit = copy_audit(model, task, st, w.kamin_audit_sequences, 1) if switched else None
    if stage == "stage1" and not switched and failure is None:
        failure = "cap_miss"
    if switched and audit["mean"] < .99 and not forced:
        failure = "audit_failure"
    switch = dict(n_s=successful, trigger=trigger.records, audit=audit, rho_unchanged=stage1,
                  passed=switched and audit["mean"] >= .99 if switched else False,
                  forced_for_smoke=forced)
    if stage2:
        switch = meta["switch"]
    if masked:
        off = copy_audit(model, task, st, w.kamin_trigger_sequences, 0, masked=True)
        on = copy_audit(model, task, st, w.kamin_trigger_sequences, 0)
        switch.update(masked=off, unmasked=on, chance_pass=.04 <= off["mean"] <= .09)
        if rolling != job["rolling_data"] or first_batch != job["first_batch_sha1"]:
            raise AssertionError("duration-yoked clean stream mismatch")
    global_steps = meta["completed_steps"] + successful if stage2 else successful
    smeta = dict(seed=seed, arm=label, stage=stage, completed_steps=global_steps,
                 local_step=successful, rolling_data=rolling, first_batch_sha1=first_batch,
                 quick_sha256=state_hash(t.fixed_set(task, st, "quick")[0]),
                 eval_sha256=state_hash(t.fixed_set(task, st, "eval")[0]),
                 trigger_sha256=state_hash(task.raw(np.random.default_rng([t.PROBE_SEED, 0]), w.kamin_trigger_sequences)[0]),
                 switch=switch)
    snapshot = save_snapshot(root/f"{ident}__final.pt", model, opt, rng, smeta)
    replay = continuation_check(snapshot,model,opt,rng,task,st,eta) if forced and switched else None
    record = dict(cell_id=ident, task=name, seed=seed, arm=label, rule=t.arm_inner_loss(arm), stage=stage,
                  expected_steps=steps, successful_steps=successful, attempted_steps=u+1,
                  complete=successful == steps if stage != "stage1" else switched and failure is None,
                  failure=failure, snapshot=snapshot, **initial, rolling_data=rolling,
                  first_batch_sha1=first_batch, quick=quick, probes=probes, switch=switch,
                  forced_for_smoke=forced, eta=[float(e.detach()) for e in model.etas(eta)],
                  plain=model.frozen_emb is None, clone=clone if stage2 else None,
                  continuation_check=replay)
    if not stage1:
        model.eval()
        record["eval"] = score(model, task, st, arm, eta)
        record["bound_final"] = bool(quick and quick[-1].get("query_hits", 0)*10 >= 9*quick[-1].get("query_events", 1))
        bound = [q["local_step"] for q in quick if q["local_step"] > 0 and
                 (not stage2 or q["local_step"] % 25 == 0) and
                 q.get("query_hits", 0)*10 >= 9*q.get("query_events", 1)]
        record.update(ever_bound=bool(bound), first_bound=min(bound) if bound else None)
        if name == "echo" and probes:
            last = probes[-1]
            record["s5"] = dict(passed=last["lin_kv_max_diff"] <= 1e-4 and
                                      last["lin_kv_acc_diff"] <= .002,
                                lin_kv_max_diff=last["lin_kv_max_diff"],
                                lin_kv_acc_diff=last["lin_kv_acc_diff"])
        if name == "echo" and not job.get("memory"):
            try:
                data = t.eval_batches(task, *t.fixed_set(task, st, "eval"), dev, st.eval_batch)
                record["ablation"] = t.r42_ablation(model, task, data, eta, st.g_clip, arm=arm)
                record["ablation_ref_match"] = abs(record["ablation"]["all"]-record["eval"]["query_acc"]) <= .01
            except Exception as exc:
                record["ablation_error"] = repr(exc)
    record["wall_s"] = time.perf_counter()-start
    record["train_seconds"] = train_seconds
    record["ms_step"] = train_seconds*1000/max(1,successful)
    if stage1:
        emit("SWITCH", cell_id=ident, seed=seed, arm=label, **switch, failure=failure)
    return record


@torch.no_grad()
def reassign_eval(cfg, w, entry, dev):
    st = t._settings(cfg, "echo"); task = t._task(st, "echo")
    model = t.build_model(task, st, dev, entry["seed"], entry["arm"])
    if "snapshot" in entry:
        opt = t.build_optim(model, st)
        load_snapshot(entry["snapshot"], model, opt, np.random.default_rng())
        del opt
        checkpoint = entry["snapshot"]
    else:
        if sha_file(entry["path"]) != entry["sha256"]:
            raise ValueError("historical checkpoint changed after preflight")
        model.load_state_dict(torch.load(entry["path"], map_location=dev, weights_only=True))
        checkpoint = entry
    model.eval().requires_grad_(False)
    conditions, generator = p.reassign_set(w.reassign_sequences, 512, 64, 4, 16, 64, w.reassign_eval_seed)
    rows, mechanism = [], []
    for condition, (raw, metadata) in conditions.items():
        for mode in ("streaming", "query_freeze", "write_snapshot"):
            totals = torch.zeros(4, dtype=torch.float64, device=dev)
            for lo in range(0, len(raw), st.eval_batch):
                hi = min(lo+st.eval_batch, len(raw))
                # Explicit query and two-write metadata; no single-write summarizer.
                bm = {k: v[lo:hi] for k, v in metadata.items()}
                assert bm["query_pos"].shape[1] == 16
                assert bm["write_pos"].shape[1] == (16 if condition == "removed" else 32)
                idx, y, bm = task.tensors(raw[lo:hi], bm, dev)
                h = model.backbone(idx).detach()
                need_mechanism = mode=="streaming" and condition!="removed"
                ref = (t.reference_logits(model, h, idx, model.per_sequence(len(idx)))
                       if need_mechanism and entry["arm"] not in t.AICE_ARMS else None)
                etas = [e.detach() for e in model.etas(w.echo_eta_base)]
                def observe(c, z, reference, W, raw_grads, grads, norm, clipped):
                    if c != 1 or condition == "removed" or mode != "streaming":
                        return
                    pos = torch.as_tensor(bm["write_pos"][:,16:] - 64, device=dev)
                    old = torch.as_tensor(bm["old_tok"], device=dev)
                    new = torch.as_tensor(bm["new_tok"], device=dev)
                    ordinary = p.gather(z.softmax(-1), pos)
                    baseline = reference if reference is not None else ref
                    increment = p.gather((z-baseline[:,64:128]).softmax(-1), pos)
                    row = dict(condition=condition, sequence_start=lo,
                               write_target_map="f" if condition=="unchanged" else "g",
                               p_f=ordinary.gather(-1,old[...,None]).squeeze(-1).detach().cpu().tolist(),
                               p_g=ordinary.gather(-1,new[...,None]).squeeze(-1).detach().cpu().tolist(),
                               increment_p_f=increment.gather(-1,old[...,None]).squeeze(-1).detach().cpu().tolist(),
                               increment_p_g=increment.gather(-1,new[...,None]).squeeze(-1).detach().cpu().tolist(),
                               raw_gradient_norms=[g.flatten(1).norm(dim=1).detach().cpu().tolist() for g in raw_grads],
                               clip_scalar=torch.where(norm>st.g_clip,st.g_clip/norm,torch.ones_like(norm)).detach().cpu().tolist(),
                               clip_rate=float(clipped.double().mean().cpu()),
                               delta_norms=[(eta*g).flatten(1).norm(dim=1).detach().cpu().tolist() for eta,g in zip(etas,grads)])
                    mechanism.append(row)
                sel = t.lin_selector(entry["arm"], bm["write_pos"], len(idx), 512, dev)
                first_query_chunk = torch.as_tensor(bm["query_pos"].min(1)//64,device=dev)
                z = t.eval_sequential(model, idx, y, etas, st.g_clip, 64, t.GradStats(), sel,
                                      h=h, rule=t.arm_inner_loss(entry["arm"]),
                                      freeze_from=first_query_chunk if mode=="query_freeze" else None,
                                      snapshot_chunk=2 if mode=="write_snapshot" else None, observer=observe)
                q = p.gather(z, bm["query_pos"])
                old, new, target = [torch.as_tensor(bm[k], device=dev) for k in ("old_tok","new_tok","query_tok")]
                pred = q.argmax(-1)
                totals += torch.stack([(pred==new).sum(), (pred==old).sum(),
                                       t.token_ce(q,target).double().sum(), pred.new_tensor(pred.numel())])
            # CPU conversion only at the condition boundary, never NumPy on CUDA.
            new_hits, old_hits, ce_sum, events = totals.detach().cpu().tolist()
            events, new_hits, old_hits = int(events), int(new_hits), int(old_hits)
            row = dict(condition=condition, mode=mode, new_hits=new_hits, old_hits=old_hits,
                       events=events, new_recall=new_hits/events, old_recall=old_hits/events,
                       intrusion=old_hits/events if condition=="reassign" else None,
                       other_hits=events-new_hits-old_hits if condition=="reassign" else None,
                       other_recall=(events-new_hits-old_hits)/events if condition=="reassign" else None,
                       query_ce=ce_sum/events, tokens_sha256=state_hash(raw))
            rows.append(row)
    return dict(seed=entry["seed"], arm=entry["arm"], historical=entry.get("historical",False),
                source=entry.get("source_run"), source_commit=entry.get("source_commit"),
                checkpoint=checkpoint, generator=generator, conditions=rows, mechanism=mechanism,
                rule=t.arm_inner_loss(entry["arm"]))


def audit_icl(cfg, w, entry, dev):
    st = t._settings(cfg,"icl_mc"); task=t._task(st,"icl_mc")
    model=t.build_model(task,st,dev,entry["seed"],entry["arm"])
    if sha_file(entry["path"]) != entry["sha256"]:
        raise ValueError("historical checkpoint hash mismatch")
    model.load_state_dict(torch.load(entry["path"],map_location=dev,weights_only=True))
    return dict(**entry, eval=score(model,task,st,entry["arm"],w.icl_eta_base))


def execute(cfg,w,job,dev):
    if job.get("action") == "reassign":
        return reassign_eval(cfg,w,job,dev)
    if job.get("action") == "historical_icl":
        return audit_icl(cfg,w,job,dev)
    return train_cell(cfg,w,job,dev)


def _worker(index,cfg,w,groups,results,lock,barrier=None):
    sys.stdout=t._LockedLines(sys.stdout,lock)
    sys.stderr=t._LockedLines(sys.stderr,lock)
    t._mark_process(f"{cfg['process_name'].replace('-podsmoke','')}-w{index}")
    torch.set_num_threads(2)
    t.configure_tf32(False,str(index))
    while True:
        group=groups.get()
        if group is None:
            break
        if barrier is not None:
            # All six contexts enter the full-shape tranche together.
            torch.cuda.init()
            barrier.wait(timeout=120)
        rows=[]
        for job in group:
            torch.cuda.reset_peak_memory_stats()
            try:
                row=execute(cfg,w,job,torch.device("cuda"))
            except Exception as exc:
                row=dict(**job,error=repr(exc),complete=False)
                emit("ERROR",**row)
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            if barrier is not None:
                # Every tranche worker has finished its job and still holds its cached
                # blocks: sample device occupancy between two barriers, before any release.
                barrier.wait(timeout=120)
                row["device_used_at_barrier_gib"]=occupancy()
                barrier.wait(timeout=120)
            emit("WORKER_MEMORY",worker=index,cell={k:job.get(k) for k in ("task","seed","arm","stage")},
                 allocated=row["peak_allocated_gib"],reserved=row["peak_reserved_gib"],
                 device_used_at_barrier=row.get("device_used_at_barrier_gib"))
            rows.append(row)
            gc.collect(); torch.cuda.empty_cache()
        results.put(rows)


def occupancy():
    try:
        out=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,check=True,timeout=5).stdout
        return max(float(x) for x in out.splitlines())/1024
    except (OSError,ValueError,subprocess.SubprocessError):
        return None


def run_state_guard():
    """Exclusivity rests on device_guard plus the lock file; no external run registry is queried."""
    emit("RUN_GUARD",source="unavailable on this host",enforced_by="device_guard + lock file")
    return None


IDLE_DEVICE_GIB = 1.0


def device_guard():
    """No other CUDA compute process and an idle device before any phase starts.

    Container PID namespaces can hide other processes from nvidia-smi, so device
    memory in use is checked as well.
    """
    apps = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],
                          capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if apps:
        raise RuntimeError(f"CUDA has an existing compute process; exclusive run guard failed: {apps}")
    used = occupancy()
    if used is None:
        raise RuntimeError("nvidia-smi memory occupancy unavailable; exclusive run guard failed")
    if used > IDLE_DEVICE_GIB:
        raise RuntimeError(f"device already holds {used:.2f} GiB; exclusive run guard failed")
    emit("DEVICE_GUARD",compute_apps=[],used_gib=used,idle_limit_gib=IDLE_DEVICE_GIB)


def _holder_alive(holder):
    pid = holder.get("pid")
    if not isinstance(pid, int) or holder.get("host") != socket.gethostname():
        return True  # unknown or foreign holder: never break it automatically
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path, name):
    """Exclusive O_EXCL lock file; a lock whose local holder PID is dead is stale."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    mine = dict(pid=os.getpid(), host=socket.gethostname(), process_name=name, started_at=time.time())
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = json.loads(path.read_text())
            except (OSError, ValueError):
                holder = {}  # creation interrupted before the record was written
            if holder and _holder_alive(holder):
                raise RuntimeError(f"another wave B driver holds {path}: {holder}")
            emit("STALE_LOCK", path=str(path), previous=holder)
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(mine, f); f.flush(); os.fsync(f.fileno())
        emit("LOCK", path=str(path), **mine)
        return mine
    raise RuntimeError(f"could not acquire {path}")


def release_lock(path, mine):
    path = Path(path)
    try:
        if json.loads(path.read_text()).get("pid") == mine["pid"]:
            path.unlink()
    except (OSError, ValueError):
        pass


class PhaseRows(list):
    """A phase's completed rows plus its terminal status (complete, budget_stop,
    deadline_terminated, failed, skipped) and any infrastructure error."""
    status = "complete"
    error = None


MEMORY_PHASES = ("memory", "memory4", "preflight")


def phase(cfg,w,number,groups,deadline,workers=None):
    """One isolated pool per phase; terminate/join on error, never cross barriers.

    Non-memory phases never raise for infrastructure trouble: completed groups are
    kept, the rest stay incomplete, and the status is recorded, so the driver can
    still write evidence and analysis (incomplete status is preserved).
    """
    ctx=mp.get_context("spawn")
    count=min(workers or cfg["workers"],len(groups))
    output=PhaseRows()
    if count==0:
        output.status="skipped"
        emit("PHASE",phase=number,status="skipped",groups=0)
        return output
    jobs,results=ctx.Queue(),ctx.Queue()
    lock=ctx.Lock()
    barrier=ctx.Barrier(count) if number in ("memory","memory4") else None
    pool=[ctx.Process(target=_worker,args=(i,cfg,w,jobs,results,lock,barrier),name=f"{cfg['process_name'].replace('-podsmoke','')}-w{i}") for i in range(count)]
    outstanding=0; submitted=0; peak=0.; start=time.monotonic()
    status, error, exited = "complete", None, False
    baseline = occupancy()
    emit("PHASE",phase=number,status="start",groups=len(groups),workers=count)
    try:
        for worker in pool:
            worker.start()
        while submitted<len(groups) or outstanding:
            while outstanding<count and submitted<len(groups) and time.time()<deadline:
                jobs.put(groups[submitted]); submitted+=1; outstanding+=1
            if not outstanding:
                break
            try:
                rows=results.get(timeout=.5)
                outstanding-=1; output.extend(rows)
                with lock:
                    for row in rows:
                        emit("REASSIGN" if number==2 else "HISTORICAL_ICL" if number==7 else "CELL",phase=number,**row)
            except queue.Empty:
                if any(not proc.is_alive() and proc.exitcode not in (None,0) for proc in pool):
                    raise RuntimeError("phase worker exited without a result (OOM/crash)")
            used=occupancy()
            if used is not None:
                peak=max(peak,used)
            if time.time()>deadline+600:
                # Hard stop: in-flight groups are terminated and stay incomplete.
                status="deadline_terminated"; break
        if status=="complete":
            for _ in pool:
                jobs.put(None)
            for worker in pool:
                worker.join(timeout=30)
                if worker.is_alive():
                    raise RuntimeError("worker did not exit at phase barrier")
            exited=True
    except Exception as exc:
        status, error = "failed", repr(exc)
    finally:
        for worker in pool:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        jobs.close(); results.close()
    if status=="complete" and submitted<len(groups):
        status="budget_stop"
    barrier_samples=[r.get("device_used_at_barrier_gib") for r in output if r.get("device_used_at_barrier_gib") is not None]
    peak=max([peak,*barrier_samples])
    after = occupancy()
    if baseline is not None and after is not None and after > baseline+.25 and status!="failed":
        status, error = "failed", "device allocations remained after phase workers exited"
    output.status, output.error = status, error
    emit("PHASE",phase=number,status=status,error=error,
         submitted_groups=submitted,total_groups=len(groups),returned_rows=len(output),wall_s=time.monotonic()-start,
         aggregate_peak_gib=peak,aggregate_after_gib=after,aggregate_before_gib=baseline,
         barrier_samples_gib=barrier_samples,
         sum_peak_reserved_gib=sum(r.get("peak_reserved_gib") or 0 for r in output))
    atomic_json(Path(w.output_dir)/f"phase-{number}-manifest.json",dict(phase=number,status=status,error=error,records=output,
                aggregate_peak_gib=peak,barrier_samples_gib=barrier_samples,workers=count,
                workers_exited=exited,wall_s=time.monotonic()-start))
    if number in MEMORY_PHASES:
        if status!="complete":
            raise RuntimeError(f"memory tranche {number} {status}: {error}")
        errors=[r.get("error","") for r in output if r.get("error")]
        if peak==0:
            raise RuntimeError("aggregate occupancy unavailable; memory sizing cannot be certified")
        if peak>w.max_device_usage_gib or any("out of memory" in error.lower() for error in errors):
            raise MemorySizingFailure("aggregate memory exceeded the registered bound or CUDA OOM")
        if errors:
            raise RuntimeError("memory tranche failed for a reason other than OOM")
    return output


def smoke_projection(cells):
    """Planning bound only: one-GPU saturation and diagnostic overhead are uncertain."""
    ms={(r.get("task"),r.get("arm")):r.get("ms_step",0) for r in cells if r.get("stage")=="fresh"}
    echo=2*1500*sum(ms.get(("echo",a),0) for a in ECHO)/60000
    icl=4000*sum(ms.get(("icl_mc",a),0) for a in ICL)/60000
    ce=ms.get(("echo","e2e1emb"),0)
    kamin=2*(2000+2000+3*1000)*ce/60000
    # Seed-group waves above include tail underutilization; 1.5 is a conservative
    # allowance for concurrent slowdown.
    training=echo+icl+kamin
    return dict(training_minutes_at_serial_cell_throughput=training,
                projected_minutes_low=1.25*training+30,projected_minutes_high=1.5*training+60,
                overhead_minutes=[30,60],caveat="smoke samples are short; verify concurrency before the full run")


def verify_smoke_receipt(w):
    """The full run reads the smoke's receipt at ``smoke_receipt_path``, shared by both runs."""
    path=Path(w.smoke_receipt_path)
    if not path.is_file():
        raise ValueError(f"missing full-shape smoke receipt at {path}")
    receipt=json.loads(path.read_text())
    files=source_files()
    if receipt.get("source_sha256")!=state_hash(files):
        old=receipt.get("source_files") or {}
        changed=sorted(k for k in set(old)|set(files) if old.get(k)!=files.get(k))
        raise ValueError(f"smoke receipt source/dependency hash differs; changed files: {changed}")
    if (receipt.get("project_id")!=PROJECT or not receipt.get("passed") or
            receipt.get("memory_workers") not in (4,6) or not receipt.get("six_worker_tranche_attempted") or
            not 0<receipt.get("aggregate_peak_gib",100)<=14 or
            not receipt.get("workers_exited") or not receipt.get("continuation_passed")):
        raise ValueError("failed full-shape smoke receipt")
    emit("SMOKE_RECEIPT",path=str(path),**receipt)
    return receipt


# Tracked inputs that define the code, dependencies, configs and manifests. Content
# hashes only (no mtimes, .git or .venv), so a `git archive` extraction and the
# runner's snapshot of the same commit agree.
SOURCE_FILES = ("pyproject.toml", "uv.lock", ".python-version", "configs/toy/r4-4b/historical-manifest.json",
                "configs/toy/r4-4b/wave-a-anchor-audit.json")
SOURCE_GLOBS = ("src/tttmem/*.py", "scripts/*.py", "configs/toy/r4-4b.toml", "configs/toy/smoke/r4-4b*.toml")


def source_root():
    root=Path(__file__).resolve().parents[2]
    return root if (root/"pyproject.toml").is_file() else Path.cwd()


def source_files(root=None):
    root=Path(root) if root is not None else source_root()
    paths={*(path for pattern in SOURCE_GLOBS for path in root.glob(pattern)),
           *(root/name for name in SOURCE_FILES if (root/name).is_file())}
    return {str(path.relative_to(root)):sha_file(path) for path in sorted(paths)}


def source_hash():
    return state_hash(source_files())


def smoke_complete(cells,reassign,audits,memory):
    required = {("echo","fresh",a):60 for a in ECHO}
    required.update({("icl_mc","fresh",a):100 for a in ICL})
    required.update({("echo","stage2",a):30 for a in TWINS})
    required[("echo","masked","e2e1emb")]=30
    by_key={(r.get("task"),r.get("stage"),r.get("arm")):r for r in cells if r.get("seed")==0}
    return (all(by_key.get(k,{}).get("complete") and by_key[k].get("successful_steps")==n for k,n in required.items()) and
            len(reassign)==12 and all(len(r.get("conditions",[]))==9 and not r.get("error") for r in reassign) and
            len(audits)==6 and all(r.get("eval") and not r.get("error") for r in audits) and
            len(memory.get("records",[]))==memory["workers"] and
            all(r.get("complete") and r.get("successful_steps")==2 and r.get("eval",{}).get("T")==1024 and
                r.get("eval",{}).get("sequences")==32 for r in memory["records"]))


def load_analyzer():
    path=Path(__file__).resolve().parents[2]/"scripts/analyze_r44b.py"
    spec=importlib.util.spec_from_file_location("analyze_r44b",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(cfg,seed):
    w=validate_config(cfg)
    t._mark_process(cfg["process_name"])
    # Never auto-select MPS/CPU for the registered driver.
    if not torch.cuda.is_available():
        raise RuntimeError("wave B driver requires CUDA; use tiny CPU unit fixtures for local checks")
    mine=acquire_lock(w.lock_path,cfg["process_name"])
    try:
        return _run_locked(cfg,w)
    finally:
        release_lock(w.lock_path,mine)


def _run_locked(cfg,w):
    guard_source=run_state_guard()
    device_guard()
    t.configure_tf32(False,"parent")
    files=source_files()
    emit("CONFIG",settings=dataclasses.asdict(t._settings(cfg)),icl_mc=dataclasses.asdict(t._settings(cfg,"icl_mc")),
         wave_b=dataclasses.asdict(w),torch=torch.__version__,cuda=torch.version.cuda,precision="fp32",
         attention="scaled_dot_product_attention auto; inherited kernels",deterministic=torch.are_deterministic_algorithms_enabled(),
         aice=dict(reference="per-sequence current outer W_ref; immutable",detach="complete reference logits",
                   normalization="mean chunk per sequence; batch sum",clip=10,outer="ordinary CE; predict before update"),
         run_guard=guard_source or "device_guard + lock file",source_sha256=state_hash(files),source_files=files)
    # CPU strict-shape/hash preflight avoids leaving a parent CUDA context.
    historical=verify_historical(w.checkpoint_input_manifest,cfg,torch.device("cpu"))
    emit("INPUTS",checkpoints=historical)
    if not w.smoke:
        receipt=verify_smoke_receipt(w)
        if receipt and receipt["memory_workers"]==4:
            cfg={**cfg,"workers":4}
            emit("WORKER_FALLBACK",workers=4,reason="verified smoke memory certificate",effective_settings=dataclasses.asdict(t._settings(cfg)))
    deadline=time.time()+cfg["max_minutes"]*60
    all_cells=[]; reassign=[]; audits=[]; stage1=[]; phases=[]
    memory_phase=None; driver_error=None
    def run_phase(cfg,number,groups,workers=None):
        rows=phase(cfg,w,number,groups,deadline,workers=workers)
        phases.append(dict(phase=number,status=getattr(rows,"status","complete"),error=getattr(rows,"error",None),
                           groups=len(groups),rows=len(rows)))
        return rows
    try:
        if w.smoke:
            memjobs=[[dict(task="icl_mc",seed=i,arm="e2e1_aice",steps=w.memory_probe_steps,memory=True)] for i in range(6)]
            run_phase(cfg,"preflight",memjobs[:w.preflight_workers]+[[dict(task="echo",seed=0,arm="e2e1emb_aice",steps=2,memory=True)]],workers=w.preflight_workers)
            memory_phase="memory"
            try:
                run_phase(cfg,"memory",memjobs,workers=6)
            except MemorySizingFailure:
                phases.append(dict(phase="memory",status="memory_sizing_failure",error="aggregate usage or OOM",groups=6,rows=None))
                emit("WORKER_FALLBACK",workers=4,reason="six-worker aggregate memory/OOM failure")
                cfg={**cfg,"workers":4}
                memory_phase="memory4"
                run_phase(cfg,"memory4",memjobs[:4],workers=4)
        echo=run_phase(cfg,1,[[dict(task="echo",seed=s,arm=a) for a in w.echo_arms] for s in w.echo_seeds])
        all_cells+=echo
        evaljobs=[dict(action="reassign",seed=r["seed"],arm=r["arm"],snapshot=r["snapshot"]) for r in echo
                  if r.get("complete") and r["arm"] in ("e2e1emb_lin","e2e1emb_aice")]
        evaljobs += [dict(action="reassign",historical=True,**e) for e in historical if e["task"]=="echo"]
        reassign=run_phase(cfg,2,[[j] for j in evaljobs])
        all_cells+=run_phase(cfg,3,[[dict(task="icl_mc",seed=s,arm=a) for a in w.icl_arms] for s in w.icl_seeds])
        stage1=run_phase(cfg,4,[[dict(task="echo",seed=s,arm="e2e1emb",stage="stage1")] for s in w.kamin_seeds])
        all_cells+=stage1
        if w.smoke:
            all_cells+=run_phase(cfg,"cap_failure",[[dict(task="echo",seed=99,arm="e2e1emb",stage="stage1",cap_failure=True,cap=10)]],workers=1)
        eligible=[r for r in stage1 if r.get("complete") and not r.get("failure")]
        masked=run_phase(cfg,5,[[dict(task="echo",seed=r["seed"],arm="e2e1emb",stage="masked",duration=r["switch"]["n_s"],
                                     rolling_data=r["rolling_data"],first_batch_sha1=r["first_batch_sha1"])] for r in eligible])
        all_cells+=masked
        controls={r["seed"]:r for r in masked if r.get("complete")}
        groups=[]
        for r in eligible:
            if r["seed"] in controls:
                groups.append([dict(task="echo",seed=r["seed"],arm=a,stage="stage2",
                                    snapshot=(controls[r["seed"]] if a=="kaminctl2" else r)["snapshot"]) for a in TWINS])
        all_cells+=run_phase(cfg,6,groups)
        audits=run_phase(cfg,7,[[dict(action="historical_icl",**e)] for e in historical if e["task"]=="icl_mc"])
    except Exception as exc:
        # Evidence and analysis are still written below; the error is re-raised after.
        driver_error=exc
        emit("DRIVER_ERROR",error=repr(exc))
    wave_a=json.loads(Path("configs/toy/r4-4b/wave-a-anchor-audit.json").read_text())
    emit("WAVE_A_AUDIT",**wave_a)
    payload=dict(config=dataclasses.asdict(w),cells=all_cells,reassign=reassign,historical_icl=audits,historical_wave_a=wave_a,
                 phases=phases,driver_error=None if driver_error is None else repr(driver_error))
    atomic_json(Path(w.output_dir)/"evidence.json",payload)
    emit("PHASES",phases=phases)
    analyzer=load_analyzer()
    summary=analyzer.analyze(payload)
    analyzer.write_tables(payload,summary,Path(w.output_dir)/"analysis")
    print("R44B_SUMMARY "+json.dumps(summary,allow_nan=False),flush=True)
    if w.smoke:
        manifest=Path(w.output_dir)/f"phase-{memory_phase}-manifest.json"
        memory=json.loads(manifest.read_text()) if memory_phase and manifest.is_file() else dict(workers=0,records=[])
        phases_ok=driver_error is None and all(x["status"]=="complete" or (x["phase"]=="memory" and x["status"]=="memory_sizing_failure")
                                               for x in phases)
        good=phases_ok and not any(r.get("error") for r in all_cells+reassign+audits) and smoke_complete(all_cells,reassign,audits,memory)
        replay=any((r.get("continuation_check") or {}).get("passed") for r in stage1)
        cap_ok=any(r.get("failure")=="cap_miss" for r in all_cells if r.get("seed")==99)
        receipt=dict(project_id=PROJECT,passed=bool(good and replay and cap_ok),memory_workers=memory.get("workers"),six_worker_tranche_attempted=True,
                     aggregate_peak_gib=memory.get("aggregate_peak_gib"),barrier_samples_gib=memory.get("barrier_samples_gib"),
                     workers_exited=memory.get("workers_exited"),phases=phases,
                     continuation_passed=replay,cap_failure_exercised=cap_ok,analysis_status=summary.get("status"),
                     source_sha256=state_hash(files),source_files=files,
                     generated_at=time.time(),projection=smoke_projection(all_cells))
        atomic_json(Path(w.output_dir)/"smoke-receipt.json",receipt)
        atomic_json(w.smoke_receipt_path,receipt)
        emit("SMOKE_RECEIPT",path=w.smoke_receipt_path,**receipt)
    if driver_error is not None:
        raise driver_error
    failed=[x for x in phases if x["status"]=="failed"]
    if failed:
        raise RuntimeError(f"phase infrastructure failure(s); evidence and analysis written: {failed}")
    return summary
