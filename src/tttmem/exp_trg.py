"""TRG stage-(i) legacy-state driver. Stage-(ii) primitives are imported separately."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import dense_log as dense
from . import exp_chr as chr
from . import exp_ttt as ttt
from .episodes import COMPLETE, DIVERGED, TRUNCATED, certified_divergence, classify_branch
from .trg_triggers import (ACTIVE_STEP_DOMAIN, SpikeController, WindowController,
                           drop_list_bytes, realised_drop_hash, select_kappa, yoked_sham)

ROOT = Path(__file__).resolve().parents[2]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def config_digest(cfg):
    return digest_json({k:v for k,v in cfg.items() if not k.startswith("_")})


class WallTimeLimit(RuntimeError):
    pass


def _check_deadline(cfg):
    if time.monotonic() >= cfg.get("_deadline", float("inf")):
        raise WallTimeLimit("wall-clock soft deadline reached")


def _hard_stop(output):
    """Daemon watchdog also covers a coordinator/worker hung inside native code."""
    row = dict(complete=False, watchdog=True, reason="hard deadline reached", exit_code=124)
    emit("TRG_STOP", row)
    write_json(Path(output)/"stop.json", row)
    emit("TRG_SUMMARY", row)
    children = mp.active_children()
    for process in children:
        process.terminate()
    for process in children:
        process.join(timeout=.2)
        if process.is_alive():
            process.kill()
    os._exit(124)


def step_stdout(row):
    """Compact registered evidence; full per-group/per-chunk telemetry stays on disk."""
    keys = ("family", "seed", "branch", "step", "loss", "finite", "grad_norm", "clip",
            "clip_scale", "lr", "dropped", "flags", "active", "baseline", "removed_ce_mass",
            "removed_ce_sum", "removed_count", "outer_denominator", "nonfinite_local",
            "nonfinite_cumulative", "continuation_steps", "batch_sha1")
    masked = row["masked_per_sequence"]
    return dict({k:row[k] for k in keys}, masked_min=min(masked) if masked else 0,
                masked_max=max(masked) if masked else 0)


def emit(prefix, row):
    print(prefix+" "+json.dumps(chr._finite(row), allow_nan=False), flush=True)


def write_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(chr._finite(row), indent=2, allow_nan=False))
    temp.replace(path)


def load_fork(path, model, opt, task, batch, *, expected=None):
    """Reject weights-only starts. Restore model, AdamW and clean data RNG exactly."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not {"model", "opt", "rng"} <= state.keys() or not ({"step", "state_step"} & state.keys()):
        raise ValueError("full model+AdamW+data RNG fork required")
    state["step"] = state.get("step", state.get("state_step"))
    if not state["opt"]["state"] or int(state["step"]) <= 0:
        raise ValueError("invalid AdamW fork state")
    model.load_state_dict(state["model"])
    opt.load_state_dict(copy.deepcopy(state["opt"]))
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state["rng"])
    nxt = chr.rng_peek(state["rng"], task, batch)
    if state.get("schema") == 1 and state.get("next_batch_sha1") != nxt:
        raise ValueError("state next-batch SHA1 mismatch")
    if expected and (expected["step"] != state["step"] or expected["next_batch_sha1"] != nxt):
        raise ValueError("CHR fork step / next-batch SHA1 mismatch")
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif "rngs" in state:
        r = state["rngs"]
        torch.set_rng_state(r["torch_cpu"])
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        if torch.cuda.is_available() and r["torch_cuda"]:
            torch.cuda.set_rng_state_all(r["torch_cuda"])
    return rng, int(state["step"]), dict(next_batch_sha1=nxt, sha256=dense.sha256_file(path),
             nonfinite_at_fork=(expected or {}).get("nonfinite_at_fork", state.get("nonfinite", 0)),
             legacy_state="schema" not in state)


def save_state(path, model, opt, rng, step, *, meta=None):
    """Full schema-v1 state for SO reruns; metadata includes detector / set hashes."""
    meta = dict(meta or {}, event_step=step, detected_step=step, state_step=step,
                completed_updates=step, schema=1)
    meta.setdefault("nonfinite", meta.get("nonfinite_at_fork", 0))
    task, batch = meta.pop("task"), meta.pop("batch")
    meta.setdefault("copy_noise", dict(rng="numpy", identity=[meta.get("seed", 0), 4406, step],
                                       stateless=True, next_update=step))
    meta.setdefault("extra", {})
    torch.save(dict(model=chr._to_cpu(copy.deepcopy(model.state_dict())),
                    opt=chr._to_cpu(copy.deepcopy(opt.state_dict())),
                    rng=copy.deepcopy(rng.bit_generator.state), step=step,
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    rngs=dict(python=random.getstate(), numpy=np.random.get_state(),
                              torch_cpu=torch.get_rng_state(),
                              torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                    next_batch_sha1=chr.rng_peek(rng.bit_generator.state, task, batch),
                    **meta), path)


def train_step(model, opt, task, st, arm, seed, step, rng, *, eta=40., outer="none",
               mask_rng=None, controller=None, policy="none", schedule=(), telemetry=True,
               telemetry_instance=None):
    """Flag before clipping / opt.step; skipped updates leave optimiser and LR intact."""
    raw, meta = task.raw(rng, st.batch)
    batch_sha1 = chr._sha1(raw)
    raw, meta = ttt.corrupt_echo(raw, meta, arm, seed, step-1, rf_probability=st.rf_probability)
    idx, y, meta = task.tensors(raw, meta, next(model.parameters()).device)
    keep = ttt.outer_keep(meta, tuple(y.shape), outer, mask_rng, idx.device)
    masked = (1-keep).sum(1) if keep is not None else None
    if masked is not None and bool((masked != task.n_keys).any()):
        raise ValueError("mask miscount: TRUNCATED code fault")
    tel = (telemetry_instance or dense.Telemetry(model)) if telemetry else None
    stats = ttt.GradStats()
    loss = ttt.arm_loss(model, arm, idx, y, eta, st.g_clip, task.chunk, stats, keep=keep,
                        meta=meta, observer=tel.observer if tel else None)
    opt.zero_grad(set_to_none=True)
    if bool(torch.isfinite(loss)):
        loss.backward()
    params = ttt.clipped_params(model)
    norms = [p.grad.detach().norm().square() for p in params if p.grad is not None]
    grad_norm = torch.stack(norms).sum().sqrt() if norms else torch.tensor(float("nan"), device=idx.device)
    rho_finite = model.rho.grad is None or bool(torch.isfinite(model.rho.grad).all())
    finite = bool(torch.isfinite(loss) & torch.isfinite(grad_norm)) and rho_finite
    norm = float(grad_norm) if finite else float("nan")
    flag = controller.observe(step, norm) if controller else dict(flags={}, flag=False, active=False, baseline=None)
    dropped = (policy == "skip" and flag["flag"]) or (policy == "sham" and step in schedule)
    group_norms = tel.before() if tel else {}
    old_lr = [g["lr"] for g in opt.param_groups]
    if finite:
        torch.nn.utils.clip_grad_norm_(params, st.grad_clip)
        if not dropped:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"]*ttt.lr_scale(step-1, ttt.arm_warmup(arm, st),
                                                    st.lr_total or st.steps, st.lr_floor)
            opt.step()
    else:
        opt.zero_grad(set_to_none=True)
    chunks = tel.chunks if tel else {}
    removed = sum(v["ce"]-v["outer_ce"] for v in chunks.values())/(task.T/task.chunk) if keep is not None else 0.
    row = dict(step=step, loss=float(loss.detach()), finite=finite, grad_norm=norm,
               gradient_groups=group_norms, clip=finite and norm > st.grad_clip,
               clip_scale=min(1., st.grad_clip/(norm+1e-12)) if finite else None,
               lr=[g["lr"] for g in opt.param_groups], lr_before=old_lr, dropped=bool(dropped),
               **flag, chunks=chunks, update=tel.after(opt) if tel else {},
               inner=stats.summary(), removed_ce_mass=removed,
               masked_per_sequence=masked.tolist() if masked is not None else [], batch_sha1=batch_sha1)
    row["inner_chunks"] = {str(chunk_id): dict(norm_mean=float(n.detach().mean()),
                                     norm_median=float(n.detach().median()), clip_share=float(cl.float().mean()))
                           for chunk_id, n, cl in zip(stats.indices, stats.norms, stats.clips)}
    row.update(removed_ce_sum=removed*st.batch*task.T,
               removed_count=int(masked.sum()) if masked is not None else 0,
               outer_denominator=st.batch*task.T)
    return row


def gradient_diagnostic(model, probe, arm, eta, st, mask, seed, state_step, step):
    """Independent probe mask stream: never advances training or intervention RNGs."""
    rng = np.random.default_rng([seed, 4242, state_step, step, 25])
    records = []
    for idx, y, meta in probe:
        keep = ttt.outer_keep(meta, tuple(y.shape), mask, rng, idx.device)
        vectors = []
        for k in (None, keep):
            loss = ttt.arm_loss(model, arm, idx, y, eta, st.g_clip, st.chunk, ttt.GradStats(), keep=k, meta=meta)
            gs = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
            vectors.append(torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                                      for p, g in zip(model.parameters(), gs)]))
        a, b = vectors
        records.append(dict(unmasked_norm=float(a.norm()), masked_norm=float(b.norm()),
                            cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0))))
    return records


def _state_meta(cfg, task, st, quick_tokens, probe_tokens, detector, nonfinite, cell):
    hashes = dict(config=config_digest(cfg), source=digest_json(source_hashes()),
                  rules=digest_json({"inner": "ce", "eta": cfg.get("eta", 40)}),
                  quick=chr._sha1(quick_tokens), probe=chr._sha1(probe_tokens),
                  eval=chr._sha1(ttt.fixed_set(task, st, "eval")[0]))
    return dict(task=task, batch=st.batch, config_sha256=config_digest(cfg),
                source_sha256=source_hashes(), rule_sha256=digest_json({"inner": "ce", "eta": cfg.get("eta", 40)}),
                quick_sha1=chr._sha1(quick_tokens), probe_sha1=chr._sha1(probe_tokens),
                nonfinite_at_fork=nonfinite, detector_tail=detector[-20:],
                cell_id=f"echo/s{cell['seed']}/{cell['arm']}/eta{cfg.get('eta',40):g}/st{st.steps}",
                arm=cell["arm"], seed=cell["seed"], eta_base=cfg.get("eta", 40.),
                trigger="t_acq", hashes=hashes)


def run_segment(cfg, cell, *, branch="main", state_file=None, kappa=5, drops=None, drop_sha256=None):
    """Single CPU/CUDA branch. Importable by downstream harnesses, no pool dependency."""
    _check_deadline(cfg)
    st = ttt._settings(cfg)
    task = ttt._task(st, "echo")
    family, seed, arm = cell["family"], cell["seed"], cell["arm"]
    if not 0 <= seed <= 9:
        raise ValueError("stage-(i) guard: legacy seeds0-9 only")
    dev = ttt._device(cfg.get("device", "cpu"))
    model = ttt.build_model(task, st, dev, seed, arm)
    opt = ttt.build_optim(model, st)
    rng, start, state_info = np.random.default_rng([seed, ttt.TRAIN_SEED]), 0, {}
    if state_file:
        rng, start, state_info = load_fork(state_file, model, opt, task, st.batch,
                                          expected=cell.get("fork_expected"))
    eta = cfg.get("eta", 40.)
    qt, qm = ttt.fixed_set(task, st, "quick")
    pt, pm = task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]), st.n_probe)
    quick = ttt.eval_batches(task, qt, qm, dev, st.eval_batch)
    probe = ttt.eval_batches(task, pt, pm, dev, st.eval_batch)
    out = Path(cfg["output_dir"])/f"{family}-s{seed}"/branch
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(family=family, seed=seed, branch=branch)
    logger = dense.DenseLogger(out, identity=identity, emit=cfg.get("emit_dense", True))
    stream = (out/"telemetry.jsonl").open("w")
    at = start if state_file else None
    end = (start+int(cfg.get("horizon", 305))) if state_file and family != "SO" else int(cfg.get("so_end", 660))
    if cfg.get("smoke_only") and cfg.get("smoke_steps"):
        end = start+int(cfg["smoke_steps"]) if state_file and family != "SO" else int(cfg["smoke_steps"])
    window = WindowController(at, family, cap=int(cfg.get("ladder_cap", 300)), so_end=end,
                              so_every=int(cfg.get("so_ladder_every", 5))) if at is not None else None
    controller = SpikeController(start, kappa) if state_file else None
    outer = {"kvm": "kv", "rnd": "rnd"}.get(branch, "none")
    mask_rng = np.random.default_rng([seed, 4242, start])
    schedule = set(yoked_sham(drops, drop_sha256)) if branch == "sham" else set()
    accuracy, dense_steps, detector, telemetry_rows, realised = [], [], [], [], []
    nonfinite, cumulative, t_div, error = 0, state_info.get("nonfinite_at_fork", 0), None, None
    fork_path = state_file
    read_costs = {"ladder": [], "light": []}
    tel = dense.Telemetry(model)

    def read(step):
        ladder = branch == "main" and window.ladder_due(step)
        begin = time.perf_counter()
        r = (dense.ladder_log if ladder else dense.light_read)(model, quick, eta, st.g_clip, task.chunk)
        read_costs["ladder" if ladder else "light"].append(time.perf_counter()-begin)
        logger.append(step, r)
        dense_steps.append(step)
        accuracy.append(r["summary"]["acc"])
        window.observe(step, accuracy[-1])
        return r

    try:
        if at is not None:
            read(at)
        for step in range(start+1, end+1):
            _check_deadline(cfg)
            row = train_step(model, opt, task, st, arm, seed, step, rng, eta=eta, outer=outer,
                             mask_rng=mask_rng, controller=controller,
                             policy=branch if branch in ("skip", "sham") else "none", schedule=schedule,
                             telemetry_instance=tel)
            # Telemetry's reader callback supplied per-chunk CE/accuracy; inner stats remain inherited.
            row.update(identity)
            if outer != "none" and step % 25 == 0:
                row["gradient_diagnostic"] = gradient_diagnostic(model, probe, arm, eta, st, outer,
                                                                    seed, start, step)
            if row["dropped"]:
                realised.append(step)
            if not row["finite"]:
                nonfinite += 1
            continuation_steps = step-start
            cumulative_now = cumulative+nonfinite
            row.update(nonfinite_local=nonfinite, nonfinite_cumulative=cumulative_now,
                       continuation_steps=continuation_steps)
            telemetry_rows.append(row)
            stream.write(json.dumps(chr._finite(row), allow_nan=False)+"\n")
            stream.flush()
            emit("TRG_STEP", step_stdout(row))
            if certified_divergence(nonfinite, continuation_steps):
                t_div = step
                break
            if at is None and step % 5 == 0:
                q = chr.quick_log(model, quick, eta, st.g_clip, task.chunk, chr.CE, (5, 6), geometry_on=False)
                detector.append(dict(step=step, acc=q["acc"]))
                emit("TRG_ACQ", dict(identity, **detector[-1]))
                acquired = step == cfg.get("smoke_fork", -1) if cfg.get("smoke_only") else q["acc"] >= .90
                if acquired:
                    at = step
                    window = WindowController(at, family, so_end=end, so_every=int(cfg.get("so_ladder_every", 5)))
                    controller = SpikeController(at, kappa)
                    fork_path = str(out/"t_acq.pt")
                    save_state(fork_path, model, opt, rng, step,
                               meta=_state_meta(cfg, task, st, qt, pt, detector, cumulative_now, cell))
                    emit("TRG_FORK", dict(identity, step=at, file=fork_path,
                                         next_batch_sha1=chr.rng_peek(rng.bit_generator.state, task, st.batch)))
                    # The inclusive acquisition read is reused for the dense baseline.
                    read(step)
            elif at is not None:
                read(step)
    except Exception as exc:
        error = repr(exc)
        emit("TRG_BRANCH_ERROR", dict(identity, error=error))
    finally:
        stream.close()
        try:
            logger.close()
        except Exception as exc:
            error = error or repr(exc)
            logger.rows.clear()
            if not logger.stream.closed:
                logger.close()
    disposition = (TRUNCATED if error else classify_branch(dense_steps, accuracy, start=at if at is not None else end,
                    end=end, grid=1, t_div=t_div, certified=t_div is not None,
                    continuation_steps=t_div-start if t_div else None, nonfinite=nonfinite,
                    certificate_start=start))
    if at is None:
        disposition = TRUNCATED
    acq_path = out/"acquisitions.jsonl"
    acq_path.write_text("".join(json.dumps(dict(identity, **r))+"\n" for r in detector))
    drop_path = out/"drops.json"
    drop_path.write_bytes(drop_list_bytes(realised))
    result = dict(identity, t_acq=at, start=start, end=end, steps=dense_steps, accuracy=accuracy,
                  t_l1=window.onset if window else None, disposition=disposition,
                  t_div=t_div, certified=t_div is not None, error=error, state_file=fork_path,
                  state_info=state_info, nonfinite_local=nonfinite, nonfinite_cumulative=cumulative+nonfinite,
                  drops=realised, drop_list_sha256=dense.sha256_file(drop_path), drop_list_file=str(drop_path),
                  acquisition_sha256=dense.sha256_file(acq_path),
                  telemetry_sha256=dense.sha256_file(out/"telemetry.jsonl"),
                  telemetry_steps=[r["step"] for r in telemetry_rows],
                  norms=[r["grad_norm"] for r in telemetry_rows], kappa=kappa,
                  output_dir=str(out), shard_manifest_sha256=dense.sha256_file(out/"shards.json"),
                  read_costs={k: dict(n=len(v), median_s=float(np.median(v)) if v else None) for k, v in read_costs.items()},
                  config_sha256=config_digest(cfg), source_sha256=source_hashes(),
                  final_sha256=None)
    final_path = out/"final.pt"
    torch.save(chr._to_cpu(model.state_dict()), final_path)
    result["final_sha256"] = dense.sha256_file(final_path)
    write_json(out/"branch.json", result)
    emit("TRG_BRANCH", result)
    return result


def source_hashes():
    paths = ("exp_trg.py", "dense_log.py", "trg_triggers.py", "trg_jobs.py", "analyze_trg.py",
             "episodes.py", "exp_chr.py", "exp_ttt.py")
    return {**{p: dense.sha256_file(Path(__file__).with_name(p)) for p in paths},
            "scripts/analyze_trg.py": dense.sha256_file(ROOT/"scripts/analyze_trg.py")}


def _cell_worker(payload):
    cfg, cell, phase, kappa = payload
    worker = cell["seed"]
    ttt._mark_process(f"{cfg['process_name']}-{phase}-{cell['family'].lower()}-s{worker}")
    torch.set_num_threads(min(2, int(cfg.get("threads_per_worker", 2))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = Path(cfg["output_dir"])/f"{cell['family']}-s{cell['seed']}"
    marker = folder/f"cell-{phase}.json"
    if cfg.get("resume", True) and marker.exists():
        saved = json.loads(marker.read_text())
        if (saved["config_sha256"] == config_digest(cfg) and saved["source_sha256"] == source_hashes()
                and (phase != "probes" or saved.get("kappa") == kappa)):
            try:
                for r in saved["branches"]:
                    if r["disposition"] == TRUNCATED and (r.get("attempts",0) >= 2 or r["t_acq"] is None):
                        continue  # one retry exhausted: terminal missing, never refill
                    dense.validate_shards(r["output_dir"])
                    evidence_folder = Path(r["output_dir"])
                    if (dense.sha256_file(evidence_folder/"telemetry.jsonl") != r["telemetry_sha256"]
                            or dense.sha256_file(evidence_folder/"final.pt") != r["final_sha256"]
                            or dense.sha256_file(evidence_folder/"acquisitions.jsonl") != r["acquisition_sha256"]
                            or dense.sha256_file(evidence_folder/"drops.json") != r["drop_list_sha256"]):
                        raise ValueError("resume evidence hash mismatch")
                    if r["disposition"] == TRUNCATED:
                        raise ValueError("retry truncated cell")
                emit("TRG_RESUMED", dict(cell=cell, phase=phase))
                return saved
            except ValueError:
                pass
    branches = []
    if phase == "main":
        for attempt in range(2):
            main = run_segment(cfg, cell, state_file=cell.get("state_file"))
            if main["disposition"] != TRUNCATED or main["t_acq"] is None:
                break
        main["attempts"] = attempt+1
        branches.append(main)
        if cell["family"] != "SO" and cfg.get("probes", True):
            for tag in ("kvm", "rnd"):
                for attempt in range(2):
                    r = run_segment(cfg, cell, branch=tag, state_file=main["state_file"])
                    if r["disposition"] != TRUNCATED:
                        break
                r["attempts"] = attempt+1
                branches.append(r)
    else:
        main = json.loads((folder/"main"/"branch.json").read_text())
        if main["t_acq"] is not None:
            for attempt in range(2):
                skip = run_segment(cfg, cell, branch="skip", state_file=main["state_file"], kappa=kappa)
                if skip["disposition"] != TRUNCATED:
                    break
            skip["attempts"] = attempt+1
            branches.append(skip)
            if skip["disposition"] != TRUNCATED:
                for attempt in range(2):
                    sham = run_segment(cfg, cell, branch="sham", state_file=main["state_file"], kappa=kappa,
                                       drops=skip["drops"], drop_sha256=skip["drop_list_sha256"])
                    if sham["disposition"] != TRUNCATED:
                        break
                sham.update(attempts=attempt+1, after=[f"trg-SO-s{cell['seed']}-skip"],
                            source_drop_list_sha256=skip["drop_list_sha256"])
                write_json(Path(sham["output_dir"])/"branch.json", sham)
                branches.append(sham)
    result = dict(cell=cell, phase=phase, branches=branches,
                  kappa=kappa, config_sha256=config_digest(cfg), source_sha256=source_hashes())
    write_json(marker, result)
    return result


def _worker_init(lock):
    # Entire JSON lines exceed PIPE_BUF: preserve the log-only evidence contract.
    sys.stdout = ttt._LockedLines(sys.stdout, lock)
    sys.stderr = ttt._LockedLines(sys.stderr, lock)


def run_trg_cell(cfg, cell, dev=None):
    """Stage-(i) one-cell replay/rerun and FO/noise mask pack; no fresh training."""
    cfg = dict(cfg)
    cfg.setdefault("process_name", "TTM-trg-80")
    if dev is not None:
        cfg["device"] = str(dev)
    return _cell_worker((cfg,cell,"main",5))


def _parallel(cfg, cells, phase, kappa=5):
    payloads = [(cfg, c, phase, kappa) for c in cells]
    if int(cfg.get("workers", 1)) == 1:
        return [_cell_worker(p) for p in payloads]
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=ctx,
                              initializer=_worker_init, initargs=(ctx.Lock(),))
    try:
        result = list(pool.map(_cell_worker, payloads))
    except BaseException:
        for process in list(pool._processes.values()):
            process.terminate()
            process.join(timeout=.2)
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
        return result


def prepare_inputs(cfg, cells):
    """Fail closed on production historical evidence and per-cell CHR_FORK records."""
    smoke = bool(cfg.get("smoke_only"))
    series = cfg.get("historical_series_path")
    if not series or not Path(series).is_file():
        if not smoke:
            raise ValueError("CHR series file is required in non-smoke mode")
        series_hash = None
    else:
        series_hash = dense.sha256_file(series)
    forks = {}
    if cfg.get("chr_log_path"):
        opener = gzip.open if str(cfg["chr_log_path"]).endswith(".gz") else open
        with opener(cfg["chr_log_path"], "rt") as f:
            for line in f:
                if line.startswith("CHR_FORK "):
                    r = json.loads(line[9:])
                    parts = r["cell_id"].split("/")
                    if len(parts) < 3:
                        raise ValueError("invalid CHR_FORK cell_id")
                    forks.setdefault((parts[1], parts[2]), []).append(r)
    for c in cells:
        if c["family"] == "SO":
            continue
        if "state_file" not in c and not (smoke and c.get("rerun")):
            matches = list(Path(cfg["fork_dir"]).glob(f"echo__s{c['seed']}__{c['arm']}__*__fork_state_step*.pt"))
            if len(matches) != 1:
                raise ValueError(f"expected exactly one full fork file for {c}")
            c["state_file"] = str(matches[0])
        found = forks.get((f"s{c['seed']}", c["arm"]), [])
        if not smoke and len(found) != 1:
            raise ValueError(f"expected exactly one CHR_FORK record for {c['family']}/s{c['seed']}")
        if found:
            if len(found) != 1 or Path(found[0]["file"]).name != Path(c["state_file"]).name:
                raise ValueError("CHR_FORK duplicate record or file basename mismatch")
            if not {"step", "next_batch_sha1", "nonfinite_at_fork"} <= found[0].keys():
                raise ValueError("incomplete CHR_FORK record")
            c["fork_expected"] = found[0]
    return series_hash


def _run_stage_i(cfg, seed):
    """Only legacy seeds are trainable here; stage-(ii) specs do not execute."""
    cfg = copy.deepcopy(cfg)
    cells = cfg.get("cells") or ([dict(family="FO", seed=s, arm="e2e1emb") for s in (0,2,3,4,5,6,7,8,9)]
            +[dict(family="noise", seed=s, arm="e2e1emb_cn97") for s in range(6)]
            +[dict(family="SO", seed=s, arm="e2e2emb") for s in range(6)])
    if any(not 0 <= c["seed"] <= 9 for c in cells):
        raise ValueError("stage-(i) guard: legacy seeds0-9 only; fresh seeds belong to sealed pool")
    cfg.setdefault("process_name", "TTM-trg-80")
    ttt._mark_process(cfg["process_name"])
    torch.set_num_threads(min(2, int(cfg.get("threads_per_worker", 2))))
    if cfg.get("require_cuda", False) and not torch.cuda.is_available():
        raise ValueError("production stage-(i) requires CUDA")
    if cfg.get("device") == "mps":
        raise ValueError("MPS training disabled")
    if cfg.get("device") == "cpu" and int(cfg.get("workers",1)) != 1:
        raise ValueError("local CPU validation requires exactly one worker")
    series_hash = prepare_inputs(cfg, cells)
    plan_hash = dense.sha256_file(ROOT/cfg["plan_file"]) if cfg.get("plan_file") else None
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="trg-stage-i-v2", smoke_only=bool(cfg.get("smoke_only")),
                    plan_sha256=plan_hash, source_sha256=source_hashes(),
                    schema_sha256=dense.SCHEMA_SHA256,
                    chr_series_sha256=series_hash,
                    effective_config={k:v for k,v in cfg.items() if not k.startswith("_")}, cells=cells,
                    stage="i", confirmatory=False)
    write_json(output/"manifest.json", manifest)
    emit("TRG_MANIFEST", manifest)
    mains = _parallel(cfg, cells, "main")
    so = [r["branches"][0] for r in mains if r["cell"]["family"] == "SO" and r["branches"][0]["t_acq"] is not None]
    calibration = select_kappa([dict(steps=r["steps"], accuracy=r["accuracy"], t_acq=r["t_acq"],
                               telemetry_steps=r["telemetry_steps"], norms=r["norms"]) for r in so])
    calibration.update(plan_sha256=plan_hash, source_sha256=source_hashes(),
                       schema_sha256=dense.SCHEMA_SHA256, smoke_only=bool(cfg.get("smoke_only")),
                       dated_addendum=time.strftime("%Y-%m-%d"), stage="i")
    write_json(output/"calibration.json", calibration)
    (output/"calibration_addendum.md").write_text(
        f"# TRG κ calibration addendum — {calibration['dated_addendum']}\n\n"
        f"Stage (i), {'smoke only; cannot seal' if cfg.get('smoke_only') else 'legacy seeds; exploratory'}.\n"
        f"κ = {calibration['kappa']}; qualifying units = {calibration['qualifying_units']}; "
        f"no-data default = {calibration['no_data_default']}.\n\n"
        f"Plan sha256: {plan_hash}\n\nSchema sha256: {dense.SCHEMA_SHA256}\n\n"
        f"Median J: {json.dumps(calibration['median_j'])}\n\n"
        f"Near/far step domain: {ACTIVE_STEP_DOMAIN}. Only controller-active rows.\n\n"
        f"Code hashes:\n\n```json\n{json.dumps(source_hashes(),indent=2)}\n```\n")
    emit("TRG_CALIBRATION", calibration)
    if cfg.get("probes", True):
        mains += _parallel(cfg, [c for c in cells if c["family"] == "SO"], "probes", calibration["kappa"])
    records = [b for r in mains for b in r["branches"]]
    write_json(output/"branches.json", records)
    result = dict(protocol="trg-stage-i-v2", stage="i", exploratory=True, complete=True,
                  watchdog=False, plan_sha256=plan_hash,
                  schema_sha256=dense.SCHEMA_SHA256, output_dir=str(output), branches=len(records),
                  dispositions={d: sum(r["disposition"] == d for r in records) for d in (COMPLETE,DIVERGED,TRUNCATED)},
                  calibration=calibration)
    emit("TRG_SUMMARY", result)
    return result


def run(cfg, seed):
    """Soft stop at max_minutes, hard native-code watchdog after the grace period."""
    cfg = copy.deepcopy(cfg)
    minutes = float(cfg.get("max_minutes", 80))
    grace = float(cfg.get("hard_grace_minutes", 10))
    if not np.isfinite(minutes+grace) or minutes <= 0 or grace < 0:
        raise ValueError("invalid wall-clock watchdog budget")
    cfg["_deadline"] = time.monotonic()+60*minutes
    timer = threading.Timer(60*(minutes+grace), _hard_stop, args=(cfg["output_dir"],))
    timer.daemon = True
    timer.start()
    try:
        return _run_stage_i(cfg, seed)
    except WallTimeLimit as exc:
        row = dict(complete=False, watchdog=True, reason=str(exc), output_dir=cfg["output_dir"])
        write_json(Path(cfg["output_dir"])/"stop.json", row)
        emit("TRG_STOP", row)
        emit("TRG_SUMMARY", row)
        return row
    finally:
        timer.cancel()
