"""R4-4c: continue immutable R4-3 finals, with a fresh outer optimizer.

Only orchestration lives here. Model, CE/lin recurrence, fixed evaluation,
gradient clipping, probes and write ablations are shared with exp_ttt.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import exp_ttt as t

RULES = {"e2e2emb": ("e2e2emb", "e2e1emb", "e2e1emb_lin"),
         "e2e2": ("e2e2", "e2e1", "e2e1_lin", "e2e1_code")}
PLAIN_SEEDS = (0, 5, 6, 7, 8, 11, 15, 17, 19)
STEP0_TOLERANCE = 1 / 1024 + 1e-6


def cell_id(cell: dict) -> str:
    return f"c1__s{cell['seed']}__{cell['init']}__{cell['rule']}__{cell['lr_mode']}"


def checkpoint_path(cfg: dict, cell: dict) -> Path:
    return Path(cfg["checkpoint_dir"]) / (
        f"echo__s{cell['seed']}__{cell['init']}__eta10__st1500.pt")


def schedule_multiplier(mode: str, step: int, total: int = 375,
                        warmup: int = 25, floor: float = .1) -> float:
    """Update numbers are 1..total; step 0 is the initial LR.

    Smoke runs retain the 375-step schedule, so they rehearse its prefix.
    Both the network LR and eta_lr multiply this exact value.
    """
    if not 0 <= step <= total or not 0 < warmup < total or not 0 < floor <= 1:
        raise ValueError("invalid schedule arguments")
    if mode == "floor":
        return floor
    if mode != "rewarm":
        raise ValueError(f"unknown LR mode: {mode}")
    if step <= warmup:
        return floor + (1 - floor) * step / warmup
    return floor + (1 - floor) * .5 * (1 + math.cos(
        math.pi * (step - warmup) / (total - warmup)))


def set_lrs(opt, cfg: dict, cell: dict, step: int) -> float:
    scale = schedule_multiplier(cell["lr_mode"], step,
                                int(cfg.get("schedule_steps", 375)),
                                int(cfg.get("warmup", 25)), float(cfg.get("lr_floor", .1)))
    for group in opt.param_groups:
        group["lr"] = group["base_lr"] * scale
    return opt.param_groups[0]["lr"]


def load_initial(cfg: dict, cell: dict, dev: torch.device):
    """Strict model+rho restore before installing anything; never read Adam state."""
    st = t._settings(cfg)
    task = t._task(st, "echo")
    path = checkpoint_path(cfg, cell)
    if not path.is_file():
        raise FileNotFoundError(f"missing R4-3 final checkpoint: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = t.build_model(task, st, torch.device("cpu"), int(cell["seed"]), cell["init"])
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.rho.detach(), state["rho"]):
        raise AssertionError("rho did not restore exactly")
    model = model.float().to(dev)
    return model, task, st


def step0_check(model, task, st, cell, data, eta_base=10.) -> dict:
    """Validate source CE before switching lin or setting the installed code's gamma0.

    Lin changes inference immediately without an outer update, so source-CE
    reproducibility and rule-step-0 accuracy must be separate measurements.
    """
    acc = t.quick_summary(model, task, cell["init"], eta_base, st, data)["query_acc"]
    ref = float(cell["reference_quick_acc"])
    exact = abs(acc - ref) <= 1e-6  # references are rounded to six decimals
    match = abs(acc - ref) <= STEP0_TOLERANCE
    print(f"C1_STEP0 cell={cell_id(cell)} acc={acc:.12g} ref={ref:.12g} "
          f"match={match} exact={exact}", flush=True)
    if not match:
        raise AssertionError(f"step-0 mismatch for {cell_id(cell)}: {acc} != {ref}")
    return dict(acc=acc, ref=ref, match=match, exact=exact, source_rule=cell["init"])


def run_cell(cfg: dict, cell: dict, dev: torch.device) -> dict:
    started = time.perf_counter()
    ident, rule = cell_id(cell), cell["rule"]
    model, task, st = load_initial(cfg, cell, dev)
    steps = int(cfg.get("steps", 375))
    eta = float(cfg.get("eta_base", 10.))
    qraw, qmeta = t.fixed_set(task, st, "quick")
    quick_data = t.eval_batches(task, qraw, qmeta, dev, st.eval_batch)
    # Install before the source-function check; code at gamma=0 must pass too.
    if rule == "e2e1_code":
        t.install_key_code(model, int(cell["seed"]))
    source_step0 = step0_check(model, task, st, cell, quick_data, eta)
    if rule == "e2e1_code":
        with torch.no_grad():
            model.code_gate.fill_(float(cfg.get("gamma0", .5)))
    opt = t.build_optim(model, st)
    if opt.state:
        raise AssertionError("continuation optimizer is not fresh")
    lr = set_lrs(opt, cfg, cell, 0)
    data_seed = int(cell["seed"]) + int(cfg.get("data_seed_offset", 50_000))
    rng = np.random.default_rng([data_seed, t.TRAIN_SEED])
    probe_data = t.eval_batches(task, *task.raw(
        np.random.default_rng([t.PROBE_SEED, 0]), st.n_probe), dev, st.eval_batch)
    quick, probes, losses, step_seconds, checkpoints = [], [], [], [], []

    def measure(step: int, loss=None) -> None:
        summary = t.quick_summary(model, task, rule, eta, st, quick_data)
        etas = [float(e.detach()) for e in model.etas(eta)]
        gamma = float(model.code_gate.detach()) if model.code_gate is not None else None
        gamma_lr = opt.param_groups[-2]["lr"] if model.code_gate is not None else None
        row = [step, summary["ce"], summary["query_ce"], summary["query_acc"], loss,
               None, *etas]
        quick.append(row)
        print("C1_QUICK " + json.dumps(dict(cell_id=ident, step=step, lr=lr,
              eta_lr=opt.param_groups[-1]["lr"], gamma=gamma, gamma_lr=gamma_lr, train_loss=loss,
              **summary)), flush=True)
        if step == 0 or step == steps or step % st.probe_every == 0:
            model.eval()
            probe = dict(cell_id=ident, step=step, lr=lr,
                         eta_lr=opt.param_groups[-1]["lr"], gamma=gamma, gamma_lr=gamma_lr,
                         query_acc=summary["query_acc"],
                         **t.r4_probe(model, task, probe_data, eta, st.g_clip, arm=rule))
            probes.append(probe)
            print("PROBE " + json.dumps(probe), flush=True)

    measure(0)
    t._reset_peak(dev)
    first_hash = None
    for step in range(1, steps + 1):
        model.train()
        lr = set_lrs(opt, cfg, cell, step)
        idx, y, meta = task.tensors(*task.raw(rng, st.batch), dev)
        if first_hash is None:
            first_hash = hashlib.sha1(idx.cpu().numpy().tobytes()).hexdigest()
        sel = t.lin_selector(rule, meta.get("write_pos"), len(idx), task.T, dev)
        t._sync(dev)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = t.arm_loss(model, rule, idx, y, eta, st.g_clip, task.chunk,
                          t.GradStats(), sel=sel)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss: {ident} step={step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(t.clipped_params(model), st.grad_clip)
        if not bool(torch.isfinite(norm)) or model.rho.grad is None or not bool(
                torch.isfinite(model.rho.grad).all()):
            raise FloatingPointError(f"nonfinite gradient: {ident} step={step}")
        opt.step()
        t._sync(dev)
        step_seconds.append(time.perf_counter() - ts)
        losses.append(float(loss.detach()))
        if step % st.quick_every == 0 or step % st.probe_every == 0 or step == steps:
            measure(step, losses[-1])
        if step in cfg.get("checkpoint_steps", [100, 200]):
            checkpoints.append(t.save_checkpoint(model, f"{ident}__step{step}"))
    final_checkpoint = t.save_checkpoint(model, f"{ident}__step{steps}")
    # Retention/loss are training-time evidence, independent of final evaluation.
    flags = t.r4_flags(quick, probes, quick[-1][3],
                       t.echo_references(qmeta, task.n_keys)["occ_minus"]["acc"])
    lost = next((row[0] for row in quick if row[3] < .90), None)
    result = dict(cell_id=ident, init=cell["init"], rule=rule, lr_mode=cell["lr_mode"],
                  seed=int(cell["seed"]), steps=steps, device=str(dev), tf32=False,
                  data_seed=data_seed, first_batch_sha1=first_hash, step0=source_step0,
                  optimizer_fresh=True, quick=quick, probes=probes, losses=losses,
                  t_lose=lost, min_acc=min(row[3] for row in quick),
                  final_quick_acc=quick[-1][3], final_eval_ok=False,
                  eta_final=[float(e.detach()) for e in model.etas(eta)],
                  gamma_final=float(model.code_gate.detach()) if model.code_gate is not None else None,
                  train_s=sum(step_seconds), s_per_step=float(np.median(step_seconds)),
                  mean_s_per_step=float(np.mean(step_seconds)), peak_mem_mb=t._peak_mem(dev),
                  checkpoint=final_checkpoint, step_checkpoints=checkpoints, **flags)
    try:
        model.eval()
        eraw, emeta = t.fixed_set(task, st, "eval")
        eval_data = t.eval_batches(task, eraw, emeta, dev, st.eval_batch)
        final = t.evaluate(model, task, rule, eval_data, emeta, eta, st)["eval"]
        # The same try/except guards the R4-2L write ablation too.
        ablation = t.r42_ablation(model, task, eval_data, eta, st.g_clip, arm=rule)
        diff = abs(ablation["all"] - final["query_acc"])
        ref_match = diff <= .01
        if not ref_match:
            print(f"ABLATION_MISMATCH cell={ident} diff={diff:.12g} "
                  f"all={ablation['all']:.12g} ref={final['query_acc']:.12g}", flush=True)
        final_flags = t.r4_flags(quick, probes, quick[-1][3],
                      t.echo_references(emeta, task.n_keys)["occ_minus"]["acc"])
        result.update(final_acc=final["query_acc"], eval=final, ablation=ablation,
                      ablation_ref_match=ref_match, final_eval_ok=True, **final_flags)
    except Exception as exc:
        print(f"FINAL_EVAL_ERROR cell={ident} error={exc!r}", flush=True)
        traceback.print_exc()
        result["final_eval_error"] = repr(exc)
    result["elapsed_s"] = time.perf_counter() - started
    # JSON null is the interoperable missing value; CSV exports it as NaN.
    print("CELL_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    return result


def validate(cfg: dict) -> torch.device:
    dev = torch.device(cfg.get("device", "cpu"))
    workers, steps = int(cfg.get("workers", 1)), int(cfg.get("steps", 375))
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("C1 supports CPU or CUDA only; MPS is disabled")
    if workers < 1 or steps < 1 or steps > int(cfg.get("schedule_steps", 375)):
        raise ValueError("invalid workers/step count")
    if sys.platform == "darwin" and (workers != 1 or dev.type != "cpu"):
        raise ValueError("macOS requires CPU and workers=1")
    if dev.type == "cpu" and (workers != 1 or steps > 60):
        raise ValueError("local CPU checks require workers=1 and at most 60 steps per cell")
    if cfg.get("tf32", False):
        raise ValueError("C1 requires TF32 off")
    if not math.isfinite(float(cfg.get("max_minutes", 75))) or float(cfg.get("max_minutes", 75)) <= 0:
        raise ValueError("max_minutes must be finite and positive")
    if not math.isfinite(float(cfg.get("gamma0", .5))) or float(cfg.get("gamma0", .5)) <= 0:
        raise ValueError("gamma0 must be finite and positive")
    if not math.isfinite(float(cfg.get("gamma_lr", .01))) or float(cfg.get("gamma_lr", .01)) <= 0:
        raise ValueError("gamma_lr must be finite and positive")
    if int(cfg.get("eval_seed", t.EVAL_SEED)) != t.EVAL_SEED:
        raise ValueError("step-0 reproduction requires R4-3 eval seed 991")
    if int(cfg.get("data_seed_offset", 50_000)) != 50_000:
        raise ValueError("registered continuation data offset is 50000")
    if not cfg.get("cells") or len({cell_id(c) for c in cfg["cells"]}) != len(cfg["cells"]):
        raise ValueError("empty or duplicate cells")
    for cell in cfg["cells"]:
        if cell["init"] not in RULES or cell["rule"] not in RULES[cell["init"]]:
            raise ValueError(f"incompatible init/rule: {cell}")
        if int(cell["seed"]) not in (range(10) if cell["init"] == "e2e2emb" else PLAIN_SEEDS):
            raise ValueError(f"unregistered init seed: {cell}")
        schedule_multiplier(cell["lr_mode"], 0)
        if not 0 <= float(cell["reference_quick_acc"]) <= 1:
            raise ValueError("invalid reference accuracy")
        if not checkpoint_path(cfg, cell).is_file():
            raise FileNotFoundError(checkpoint_path(cfg, cell))
    st = t._settings(cfg)
    if st.quick_every < 1 or st.probe_every < 1:
        raise ValueError("quick/probe cadence must be positive")
    return dev


def _worker(index, cfg, jobs, results, lock):
    sys.stdout = t._LockedLines(sys.stdout, lock)
    sys.stderr = t._LockedLines(sys.stderr, lock)
    t._mark_process(f"{cfg['process_name']}-w{index}")
    t.configure_tf32(False, f"w{index}")
    torch.set_num_threads(2)
    while (cell := jobs.get()) is not None:
        try:
            results.put((cell_id(cell), run_cell(cfg, cell, torch.device(cfg["device"]))))
        except Exception as exc:
            traceback.print_exc()
            error = dict(cell_id=cell_id(cell), error=repr(exc))
            print("CELL_ERROR " + json.dumps(error), flush=True)
            results.put((cell_id(cell), error))


def run(cfg: dict, seed: int = 0) -> dict:
    dev = validate(cfg)
    t._mark_process(cfg["process_name"])
    t.configure_tf32(False, "main")
    workers = int(cfg.get("workers", 1))
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    if workers == 1:
        return {"cells": {cell_id(c): run_cell(cfg, c, dev) for c in cfg["cells"]}}
    return _run_pool(cfg)


def _run_pool(cfg: dict) -> dict:
    """Pool collection; also testable with fake workers without spawning torch."""
    workers = int(cfg["workers"])
    ctx = mp.get_context("spawn")
    jobs, results, lock = ctx.Queue(), ctx.Queue(), ctx.Lock()
    for cell in cfg["cells"]:
        jobs.put(cell)
    for _ in range(workers):
        jobs.put(None)
    processes = [ctx.Process(target=_worker, name=f"{cfg['process_name']}-w{i}",
                 args=(i, cfg, jobs, results, lock)) for i in range(workers)]
    collected = {}
    expired = threading.Event()
    deadline = time.monotonic() + float(cfg.get("max_minutes", 75)) * 60

    def terminate_at_deadline():
        expired.set()
        print(f"POOL_DEADLINE max_minutes={cfg.get('max_minutes', 75)} "
              f"collected={len(collected)} expected={len(cfg['cells'])}", flush=True)
        for process in processes:
            if process.is_alive():
                process.terminate()

    # The watchdog also bounds join() after all cells have reported.
    watchdog = threading.Timer(max(0., deadline - time.monotonic()), terminate_at_deadline)
    watchdog.daemon = True
    try:
        for process in processes:
            process.start()
        watchdog.start()
        while len(collected) < len(cfg["cells"]) and not expired.is_set():
            try:
                ident, result = results.get(timeout=min(5., max(.001, deadline - time.monotonic())))
                collected[ident] = result
            except queue.Empty:
                if not any(p.is_alive() for p in processes):
                    break
        for process in processes:
            process.join()
        # Queue feeder threads can deliver their last result during worker exit.
        while True:
            try:
                ident, result = results.get_nowait()
                collected[ident] = result
            except queue.Empty:
                break
        watchdog.cancel()
        watchdog.join()
        if expired.is_set():
            raise RuntimeError("C1 pool exceeded max_minutes; inspect POOL_DEADLINE")
        if len(collected) != len(cfg["cells"]):
            raise RuntimeError("C1 worker died or failed to return all cells")
        if any("error" in result for result in collected.values()):
            raise RuntimeError("one or more C1 cells failed; inspect CELL_ERROR")
    finally:
        watchdog.cancel()
        if watchdog.ident is not None:
            watchdog.join()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    return {"cells": {cell_id(c): collected[cell_id(c)] for c in cfg["cells"]}}
