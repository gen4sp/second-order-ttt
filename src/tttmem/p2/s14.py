"""S14 one-run-per-machine job coordinator. All torch work runs in sequential subprocesses.

The fixed entry point dispatches mode=s14_pod_job here. No provider API is used.
Workers accept --process-name and inherit PROCESS_NAME. A failed phase never publishes DONE.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from . import matrix
from .data import _set_process_name, open_split, sha256_file
from .s14_data import prepare_data
from .soq_n3b import _oom, warm_page_cache

from .s14_protocol import SO_GRID, FO_GRID, PASSKEY_ETAS, N3B_SO_S, N3B_SO_PILOT_S, N3B_FO_PILOT_S
SO_GRID, FO_GRID = tuple(SO_GRID), tuple(FO_GRID)
PASSKEY_ETAS = {k: tuple(v) for k, v in PASSKEY_ETAS.items()}
ARMDIRS = {"so": "e2e2", "fo-ce": "e2e1-ce", "fo-cm": "e2e1-ce",
           "mini-p": "e2e12-p125", "wsd-1200": "wsd", "wsd-3469": "wsd"}
N_CM = 3469
N3B_T_SO = N3B_SO_S
N3B_T_FO = N3B_FO_PILOT_S[4]
N3B_TARGET = 1200 * N3B_SO_S
GUARD = dict(guard_every=50, guard_first=50, guard_dense_every=0, guard_max_rise=.10,
             guard_consecutive=2, guard_restart_until=200, guard_fo_stop=True,
             train_loss_guard="running_min", train_loss_max_rise=.5, gdiv=True)


def read_name(eta: float) -> str:
    return "ce" if eta == 1 else f"eta{round(100 * eta):03d}"


def read_plan(armdir: str) -> list[str]:
    # G0's V0 contract requires ce first; κ points are included in both grids.
    grid = SO_GRID if armdir == "so" else FO_GRID
    return ["ce"] + [read_name(e) for e in sorted(set(grid) | {1., 2., 4.}) if e != 1]


def passkey_etas_for(cfg, armdir):
    if cfg.get("canary", {}).get("enabled"):
        return cfg["canary"].get("passkey_etas", [1,2])
    return PASSKEY_ETAS["so" if armdir == "so" else "fo"]


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def coordinator_lock(root: Path):
    """The default root gives outputs/s14/.lock; do not modify it after a clash."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another S14 coordinator holds {root / '.lock'}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def require_no_abort(out: Path):
    aborts = sorted(out.glob("*_abort.json"))
    if aborts:
        raise ValueError(f"abort-stopped cell cannot resume or restart: {[p.name for p in aborts]}")


def compute_context(cfg, cell, n_p=None):
    arm = cell["armdir"]
    times = (N3B_T_SO, N3B_T_SO if arm == "so" else N3B_T_FO)
    if arm == "mini-p":
        times = (float(n_p.get("t_so_ms", 0))/1000, float(n_p.get("t_p_ms", 0))/1000)
        if cfg.get("preset", "mini-m") != "tiny" and not cfg.get("canary", {}).get("enabled"):
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError("MINI-P compute audit requires actual positive finite pilot times")
            if round(int(cfg.get("base_steps", 1200))*times[0]/times[1]) != n_p["n_p"]:
                raise ValueError("MINI-P horizon disagrees with its pilot times")
    return dict(arm=arm, armdir=arm, base_steps=int(cfg.get("base_steps", 1200)),
                pilot_source="s14-pilot" if arm == "mini-p" else "n3b-50",
                **({"selected_pilot_micro_batch": n_p["p_micro_batch"],
                    "selected_pilot_so_micro_batch": n_p["p_so_micro_batch"]} if arm == "mini-p" else {}),
                **({"test_only_canary": True, "pilot_measurement": "reference basis only; production pilot unmeasured"}
                   if cfg.get("canary", {}).get("enabled") else {}),
                t_so_pilot_s=times[0], t_arm_pilot_s=times[1])


def compute_record(payload, final, runtime, context=None):
    """Persist a resume-safe audit from the committed P2_TIME prefix, never wall time."""
    out = Path(payload["out_dir"])
    seconds, events = [], []
    for line in (out / "events.jsonl").read_text().splitlines():
        tag, rec = line.split(" ", 1)
        rec = json.loads(rec)
        if tag == "P2_TIME":
            seconds.extend(rec["step_s"])
        if tag in ("P2_GUARD", "P2_ABORT", "P2_GDIV", "P2_GUARD_LABEL"):
            events.append(dict(event=tag, **rec))
    for p in sorted(out.glob("*.json")):
        if p.name.endswith("_abort.json") or p.name == "gdiv.json":
            events.append(dict(event="marker", file=p.name, record=json.loads(p.read_text())))
    context = context or dict(arm=payload["arm"], armdir=payload["arm"], base_steps=1200,
        pilot_source="n3b-50", t_so_pilot_s=N3B_SO_S, t_arm_pilot_s=N3B_T_FO)
    context = {"pilot_source": "s14-pilot" if context["arm"] == "mini-p" else "n3b-50", **context}
    record = dict(**context, seed=payload["init_seed"], steps=final.get("steps", len(seconds)),
                  expected_steps=payload["steps"], micro_batch=payload["micro_batch"],
                  so_micro_batch=payload.get("so_micro_batch", 1) if context["arm"] == "mini-p" else None,
                  pilot_micro_batch=context.get("selected_pilot_micro_batch",payload["micro_batch"]),
                  pilot_so_micro_batch=context.get("selected_pilot_so_micro_batch",payload.get("so_micro_batch",1)) if context["arm"] == "mini-p" else None,
                  train_seconds=math.fsum(seconds), median_step_ms=statistics.median(seconds)*1000
                  if seconds else None, train_seconds_target=payload.get("train_seconds_target", 0),
                  guard_label=final.get("guard_label", final.get("label")), guard_abort_events=events,
                  timing_source="events.jsonl:P2_TIME.step_s", **runtime)
    if record["pilot_source"] == "n3b-50":
        record["t_arm_pilot_s"] = (N3B_SO_PILOT_S.get(payload["micro_batch"],N3B_SO_S) if context["arm"] == "so"
                                   else N3B_FO_PILOT_S.get(payload["micro_batch"],N3B_T_FO))
    previous = out / "compute.json"
    if previous.exists():
        record["peak_gb"] = max(record["peak_gb"], json.loads(previous.read_text())["peak_gb"])
    _json(previous, record)
    return record


def read_cache(item, cfg):
    """Only recorded hashes with the same checkpoint AND read configuration are reusable."""
    out = Path(item["out"])
    names = [f"g0_read_{n}.npz" for n in item["reads"]] + ["g0_hlm.npz"]
    if cfg.get("passkey", False):
        names += [f"passkey_{read_name(e)}.npz" for e in passkey_etas_for(cfg,
            "so" if item["arm"] == "e2e2" else "fo")] + ["passkey_noadapt.npz"]
    config = dict(item=item, passkey_docs=cfg.get("passkey_docs", 256),
                      passkey_seed=cfg.get("passkey_seed", 1014),
                      read_batch=cfg.get("read_batch", 2), passkey_batch=cfg.get("passkey_batch", 1), reads=names)
    record = dict(checkpoint_sha256=sha256_file(item["ckpt"]), config=config,
                  cell_config_sha256=sha256_file(item["cell_config"]),
                  config_sha256=fingerprint(config),
                  read_list=names, npz_sha256={})
    path = out / "READS_DONE.json"
    if path.exists():
        old = json.loads(path.read_text())
        if all(old.get(k) == record[k] for k in ("checkpoint_sha256", "config_sha256", "read_list")):
            record["npz_sha256"] = {n: h for n, h in old.get("npz_sha256", {}).items()
                if n in names and (out/n).is_file() and sha256_file(out/n) == h}
            check = out/'batch_check.json'
            if check.exists() and sha256_file(check) == old.get('batch_check_sha256'):
                record.update(batch_check=json.loads(check.read_text()), batch_check_sha256=sha256_file(check))
            for kind, batch, baseline, prefix in [('frontier',config['read_batch'],2,'g0_'),
                                                  ('passkey',config['passkey_batch'],1,'passkey_')]:
                if batch > baseline:
                    try:
                        from .s14_batch import validate_check
                        validate_check(record['batch_check'][kind],kind,batch,
                                       record['checkpoint_sha256'],record['cell_config_sha256'])
                    except (KeyError,ValueError,TypeError):
                        record['npz_sha256'] = {n:h for n,h in record['npz_sha256'].items() if not n.startswith(prefix)}
    record["complete"] = set(record["npz_sha256"]) == set(names)
    _json(path, record)
    return path, record


def record_read(path, record, name):
    check = Path(path).parent/'batch_check.json'
    if check.exists():
        record.update(batch_check=json.loads(check.read_text()), batch_check_sha256=sha256_file(check))
    record["npz_sha256"][name] = sha256_file(Path(path).parent / name)
    record["complete"] = set(record["npz_sha256"]) == set(record["read_list"])
    _json(path, record)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_n_p(cfg: dict, root: Path) -> dict | None:
    value = cfg.get("n_p")
    path = Path(cfg.get("n_p_file", root / "pilot/n_p.json"))
    rec = json.loads(path.read_text()) if path.exists() else None
    if value is not None:
        value = _positive_int(value, "n_p")
        if rec and rec.get("n_p") != value:
            raise ValueError("n_p disagrees with n_p_file")
        if rec is None and cfg.get('preset', 'mini-m') != 'tiny' and not {
                'p_micro_batch', 'p_so_micro_batch'} <= cfg.keys():
            raise ValueError('config n_p requires pilot-selected p_micro_batch and p_so_micro_batch')
        if rec is None and cfg.get('preset', 'mini-m') != 'tiny' and not {
                't_so_pilot_s', 't_arm_pilot_s'} <= cfg.keys():
            raise ValueError('config n_p requires actual pilot times for the compute audit')
        return rec or {"n_p": value, "source": "launcher config value",
                       "t_so_ms": float(cfg.get("t_so_pilot_s", 0)) * 1000,
                       "t_p_ms": float(cfg.get("t_arm_pilot_s", 0)) * 1000,
                       "train_seconds_target": float(cfg.get("t_so_pilot_s", 0)) * 1200,
                       "p_micro_batch": int(cfg.get("p_micro_batch", 2)),
                       "p_so_micro_batch": int(cfg.get("p_so_micro_batch", 1))}
    if rec:
        _positive_int(rec.get("n_p"), "pilot n_p")
        if rec.get("base_steps") != int(cfg.get("base_steps", 1200)):
            raise ValueError("n_p_file base_steps mismatch")
    return rec


def decide_n_p(pilot: dict, base_steps=1200) -> dict:
    """Fastest completed, finite, exactly timed setting within the 20 GiB headroom."""
    timed = int(pilot["timed_steps"])
    ok = [r for r in pilot["settings"] if r.get("status") == "ok" and
          r.get("n_timed") == timed and r.get("peak_gb") is not None and
          math.isfinite(float(r["peak_gb"])) and 0 <= float(r["peak_gb"]) <= 20.0 and
          math.isfinite(float(r.get("median_ms", 0))) and float(r.get("median_ms", 0)) > 0]
    so = [r for r in ok if r["arm"] == "e2e2"]
    partial = [r for r in ok if r["arm"] == "e2e12-p125"]
    if not so or not partial:
        raise RuntimeError("pilot has no valid SO and MINI-P timing pair")
    so, partial = min(so, key=lambda r: r["median_ms"]), min(partial, key=lambda r: r["median_ms"])
    ratio = so["median_ms"] / partial["median_ms"]
    fo = [r for r in ok if r["arm"] == "e2e1-ce" and r['micro_batch']==4]
    common = None
    if fo:
        fastest = min(fo, key=lambda r: r["median_ms"])
        r = so["median_ms"] / fastest["median_ms"]
        reference = N3B_T_SO / N3B_T_FO
        common = dict(r=r, n3b_r=reference, relative_difference=r/reference-1,
                      differs_over_5pct=abs(r/reference-1) > .05,
                      rho_s14=3469/(base_steps*r),
                      t_so_pilot_s=so["median_ms"]/1000,
                      t_fo_pilot_s=fastest["median_ms"]/1000,
                      fo_micro_batch=fastest["micro_batch"])
    return dict(n_p=round(base_steps * ratio), base_steps=base_steps, r=ratio, common_timing=common,
                t_so_ms=so["median_ms"], t_p_ms=partial["median_ms"],
                so_micro_batch=so["micro_batch"], p_micro_batch=partial["micro_batch"],
                p_so_micro_batch=partial["so_micro_batch"],
                train_seconds_target=base_steps * so["median_ms"] / 1000, pilot=pilot)


def cell_config(cfg: dict, cell: dict, root: Path, n_p: dict | None = None) -> dict:
    armdir, seed = cell["armdir"], int(cell["seed"])
    arm = ARMDIRS[armdir]
    preset = cfg.get("preset", "mini-m")
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    base_arm = "e2e1-ce" if arm == "wsd" else arm
    base = matrix._base_config(cfg, {"preset": preset, "arm": base_arm,
                                    "init_seed": seed, "data_seed": seed}, repo)
    steps = int(cfg.get("base_steps", 1200)) if armdir in ("so", "fo-ce") else N_CM
    if armdir == "mini-p":
        if n_p is None:
            raise ValueError("MINI-P requires n_p from the timing pilot or launcher config")
        steps = _positive_int(n_p["n_p"], "n_p")
    elif armdir.startswith("wsd-"):
        steps = int(armdir.split("-")[1])
    if preset == "tiny":
        steps = int(cell.get("steps", cfg.get("steps", 2)))
        base.update(cfg.get("tiny_overrides", {}))
    canary = cfg.get("canary", {})
    if canary.get("enabled"):
        steps = _positive_int(canary.get("steps", 30), "canary.steps")
    data = Path(cfg.get("data_dir", "data/p2data"))
    out = root / "runs" / preset / armdir / f"s{seed}"
    base.update(arm=arm, steps=steps, init_seed=seed, data_seed=seed,
                data_dir=str(data), out_dir=str(out), resume_dir=str(out),
                process_name=f"{cfg['process_name']}-{armdir}-s{seed}",
                device=cfg.get("device", "cuda"), resume=False, keep_final_state=True,
                save_checkpoints=True, final_model_only=True, save_model_steps=[],
                keep_resume_steps=[], log_train_seconds=True, **GUARD)
    if armdir in ("fo-cm", "mini-p"):
        base.update(warmup=round(.1 * steps), inner_warmup=round(.1 * steps))
        base["micro_batch"] = 4 if armdir == "fo-cm" else int(n_p["p_micro_batch"])
        if armdir == "mini-p":
            base.update(so_fraction=.125, so_micro_batch=int(n_p["p_so_micro_batch"]))
            base["train_seconds_target"] = (int(cfg.get("base_steps", 1200)) *
                                            float(n_p.get("t_so_ms", 0)) / 1000)
    if arm == "wsd":
        base.update(lr_schedule="wsd", warmup=120, inner_warmup=120, wsd_decay=240, micro_batch=4)
    if armdir in ("fo-cm", "wsd-3469"):
        base["train_seconds_target"] = N3B_TARGET
    if preset == "tiny":
        # Tiny smoke uses the same port, but short schedule coordinates and batch sizes.
        base.update(cfg.get("tiny_overrides", {}))
        if arm == "wsd":
            base.update(warmup=1, inner_warmup=1, wsd_decay=1)
    if canary.get("enabled"):
        cadence = _positive_int(canary.get("guard_every",10),"canary.guard_every")
        base.update(warmup=max(1,round(.1*steps)),inner_warmup=max(1,round(.1*steps)),
            wsd_decay=max(1,round(.2*steps)),guard_every=cadence,guard_first=cadence,
            n_eval=canary.get("n_docs",8),ho_docs=canary.get("ho_docs",8),
            ho_trajectory_docs=canary.get("ho_docs",8),trajectory_docs=canary.get("n_docs",8),
            eval_at_init=False,pilot_steps=[],sizing_steps=[],crossover_rules=[],
            full_extra_modes=[],assay_gate=False,pilot_docs=canary.get("ho_docs",8),log_every=10,resume_every=10,
            eval_steps=[steps],full_eval_steps=[steps])
    rows, _ = open_split(data, "A")
    extra = []
    available = len(rows)
    for split in ("B", "C"):
        if steps * int(base["global_batch"]) <= available:
            break
        rows, _ = open_split(data, split)
        available += len(rows)
        extra.append(split)
    if steps * int(base["global_batch"]) > available:
        raise ValueError(f"training data cannot cover {armdir} horizon {steps}")
    base["extra_train_splits"], base["contamination_check"] = extra, True
    base["eval_steps"] = sorted(set(int(s) for s in base.get("eval_steps", []) if 0 < s <= steps) | {steps})
    base["full_eval_steps"] = [steps]
    return base


def passkey_hook(payload: dict) -> dict:
    """Lazy import of ``tttmem.p2.passkey``, which owns the passkey read and its protocol."""
    module = importlib.import_module("tttmem.p2.passkey")
    fn = module.passkey_read
    inspect.signature(fn).bind(payload["model_path"], payload["cell_config_path"],
          payload["eta_mults"], payload["out_dir"], n_docs=payload.get("n_docs", 256),
          seed=payload.get("seed", 1014), device=payload.get("device", "cuda"),
          data_dir=payload.get("data_dir", "data/p2data"), noadapt=payload.get("noadapt", True),
          batch=payload.get("batch", 1))
    if payload.get("preflight"):
        import numpy as np
        data_dir = payload.get('data_dir', 'data/p2data')
        tok = module._tokenizer(data_dir, payload.get('tokenizer_extra', {}))
        # _host_rows always verifies the entire frozen ordering, not just this slice.
        hosts, _, _ = module._host_rows(data_dir, payload.get('n_docs',256), 1014)
        module.candidates(tok, np.random.default_rng(1014))
        return {"passkey_available": True, "verified_hosts": len(hosts)}
    result = fn(payload["model_path"], payload["cell_config_path"], payload["eta_mults"],
                payload["out_dir"], n_docs=payload.get("n_docs", 256),
                seed=payload.get("seed", 1014), device=payload.get("device", "cuda"),
                data_dir=payload.get("data_dir", "data/p2data"), noadapt=payload.get("noadapt", True),
                batch=payload.get("batch", 1))
    expected = [f"passkey_{read_name(e)}.npz" for e in payload["eta_mults"]]
    if payload.get("noadapt", True):
        expected.append("passkey_noadapt.npz")
    for name in expected:
        if not (Path(payload["out_dir"]) / name).is_file():
            raise RuntimeError(f"passkey hook did not produce {name}")
    return result


def execute_worker(phase: str, payload: dict, root: Path, name: str, deadline: float) -> dict:
    """One child at a time, bounded by the whole job's wall deadline (including prep/pulls)."""
    path = root / ".job-configs" / f"{name}.json"
    _json(path, payload)
    log = root / "logs" / f"{name}.log"
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("pod max_hours reached")
    env = dict(os.environ, PROCESS_NAME=name, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    print("S14_PHASE " + json.dumps(dict(phase=phase, process_name=name, log=str(log),
                                        seconds_left=round(left, 1))), flush=True)
    with log.open("w") as f:
        # A process group lets timeouts stop data-prep grandchildren as well as the worker.
        proc = subprocess.Popen([sys.executable, "-m", "tttmem.p2.s14", "--worker", phase,
                "--config-json", str(path), "--process-name", name], stdout=f,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=left)
        except BaseException:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if rc:
        raise RuntimeError(f"{phase} worker failed rc={rc}: {log}")
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if line.startswith("S14_WORKER_DONE "):
            return json.loads(line.split(" ", 1)[1])
    raise RuntimeError(f"{phase} worker missing completion record: {log}")


def timing_pilot(cfg: dict, root: Path, deadline: float) -> dict:
    p = cfg.get("pilot", {})
    warm, timed = int(p.get("warmup_steps", 10)), int(p.get("timed_steps", 50))
    if cfg.get("preset", "mini-m") != "tiny" and (warm, timed) != (10, 50):
        raise ValueError("production timing pilot requires 10 warmup + 50 timed steps")
    cache = warm_page_cache(cfg["data_dir"], ["A", "B"])
    rows = []
    root.joinpath("pilot").mkdir(exist_ok=True)
    settings = [("e2e2", m, 1) for m in p.get("so_micro_batch", [1, 2])]
    settings += [("e2e1-ce", m, 1) for m in p.get("fo_micro_batch", [2, 4])]
    settings += [("e2e12-p125", m, sm) for m in p.get("p_micro_batch", [2, 4])
                 for sm in p.get("p_so_micro_batch", [1, 2])]
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    for arm, micro, somicro in settings:
        label = f"{arm}-mb{micro}-somb{somicro}"
        cdir = root / "pilot" / label
        base = matrix._base_config(cfg, {"preset": cfg.get("preset", "mini-m"), "arm": arm,
                      "init_seed": 0, "data_seed": 0}, repo)
        base.update(cfg.get("tiny_overrides", {}) if cfg.get("preset") == "tiny" else {})
        if base["global_batch"] % micro:
            continue
        base.update(micro_batch=micro, so_micro_batch=somicro, steps=max(warm+timed, int(base["steps"])),
                    stop_after=warm+timed, out_dir=str(cdir), resume_dir=str(cdir), resume=False,
                    data_dir=cfg["data_dir"], device=cfg.get("device", "cuda"),
                    eval_steps=[], full_eval_steps=[], eval_at_init=False, save_checkpoints=False,
                    guard_every=0, gdiv=False, pilot_steps=[], sizing_steps=[], assay_gate=False,
                    resume_every=10**6, keep_final_state=False, max_hours=0, log_train_seconds=True)
        shutil.rmtree(cdir, ignore_errors=True)
        name = f"{cfg['process_name']}-pilot-{label}"
        try:
            done = execute_worker("train", base, root, name, deadline)
        except RuntimeError:
            if not _oom(root / "logs" / f"{name}.log"):
                raise
            rows.append(dict(arm=arm, micro_batch=micro, so_micro_batch=somicro, status="oom"))
            shutil.rmtree(cdir, ignore_errors=True)
            continue
        ms = done["final"]["step_ms"][warm:warm+timed]
        rec = {k:v for k,v in done.get("compute", {}).items()
               if k not in ('t_so_pilot_s','t_arm_pilot_s','pilot_source')}
        rec.update(micro_batch=micro, so_micro_batch=somicro, n_timed=len(ms),
                   median_ms=statistics.median(ms) if ms else 0., peak_gb=done["peak_gb"], status="ok")
        rec["arm"] = arm
        if done['peak_gb'] is None or not math.isfinite(float(done['peak_gb'])):
            rec['status'] = 'missing_peak'
        elif float(done['peak_gb']) > 20.0:
            rec['status'] = 'over_peak'
        rows.append(rec)
        print("S14_PILOT " + json.dumps(rec), flush=True)
        shutil.rmtree(cdir, ignore_errors=True)
    result = decide_n_p(dict(settings=rows, warmup_steps=warm, timed_steps=timed,
                            page_cache=cache), int(cfg.get("base_steps", 1200)))
    if result["common_timing"] is None:
        raise RuntimeError("pilot has no valid FO-CE mb4 timing for the common timing basis")
    _json(root / "pilot/n_p.json", result)
    return result


def read_config(cfg: dict, base: dict, armdir: str, root: Path) -> dict:
    seed, preset = base["init_seed"], base["preset"]
    out = root / "reads" / preset / armdir / f"s{seed}"
    names = read_plan(armdir)
    if cfg.get("canary",{}).get("enabled"):
        names = [read_name(e) for e in cfg["canary"].get("read_etas",[1,2,4])]
    cpath = root / ".cell-configs" / f"{preset}-{armdir}-s{seed}.json"
    return dict(mode="mini_g0", process_name=f"{cfg['process_name']}-{armdir}-s{seed}-reads",
        process_nn=(int(cfg.get("process_nn", 51)) - 51) % 9 + 1,
        data_dir=base["data_dir"], device=base["device"], out_dir=str(out), max_hours=cfg["max_hours"],
        n_docs=base["n_eval"], n_ho_adapt=base["ho_docs"] if cfg.get("canary",{}).get("enabled") or preset=="tiny" else 128,
        read_batch=_positive_int(cfg.get("read_batch", 2), "read_batch"),
        endpoint_parity=True, endpoint_parity_tol=1e-5,
        kappa_reads=["ce", "eta200", "eta400"], kappa_foils=8,
        kappa_boot=10000 if preset != "tiny" else 8, setup_cache=True,
        production_configs={"mini-m/wsd": "configs/mini/e2e1-ce-m.toml"},
        items=[dict(cell=f"{preset}/{armdir}/s{seed}", preset=preset, arm=base["arm"],
               init_seed=seed, data_seed=seed, step=base["steps"], overrides=base,
               cell_config=str(cpath), ckpt=str(Path(base["out_dir"]) / f"model_step{base['steps']:04d}.pt"),
               out=str(out), reads=names, hlm=True)])


def _require_training(base, result):
    fin = result.get("final", {})
    if fin.get("aborted") or fin.get("steps") != base["steps"] or fin.get("guard_label") or fin.get("label"):
        raise RuntimeError(f"training incomplete/guard stopped: {fin}")
    out = Path(base["out_dir"])
    for name in (f"model_step{base['steps']:04d}.pt", f"eval_step{base['steps']:04d}.npz",
                 "eval_meta.npz", "steps.jsonl", "events.jsonl", "compute.json",
                 f"resume_step{base['steps']:04d}.pt"):
        if not (out / name).is_file():
            raise RuntimeError(f"training output missing: {out / name}")
    for name, digest in result.get("output_sha256", {}).items():
        if Path(name).name != name or not (out / name).is_file() or sha256_file(out / name) != digest:
            raise RuntimeError(f"completed training output changed: {out / name}")


def publish_manifest(root: Path, cfg: dict) -> dict:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("resume_") and path.name not in (
                "DONE", "manifest.json", "FAILED.json", ".lock"):
            files.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                              sha256=sha256_file(path)))
    result = dict(schema="s14-pull-v1", config_sha256=fingerprint(cfg), files=files,
                  excludes=["**/resume_*.pt"], cells=cfg["cells"])
    _json(root / "manifest.json", result)
    _json(root / "DONE", dict(status="done", manifest_sha256=sha256_file(root / "manifest.json")))
    return result


def pod_job(cfg: dict) -> dict:
    _set_process_name(cfg["process_name"])
    with coordinator_lock(Path(cfg.get("out_dir", "outputs/s14"))):
        return _pod_job_locked(cfg)


def _pod_job_locked(cfg: dict) -> dict:
    cells = cfg.get("cells", [])
    if not cells or cfg.get("preset", "mini-m") not in ("mini-m", "tiny"):
        raise ValueError("pod job needs mini-m (or tiny smoke) cells")
    seen = set()
    for c in cells:
        key = (c.get("armdir"), c.get("seed"))
        if key[0] not in ARMDIRS or not isinstance(key[1], int) or key[1] < 0 or key in seen:
            raise ValueError(f"invalid/duplicate cell: {c}")
        seen.add(key)
    if float(cfg.get("max_hours", 0)) <= 0:
        raise ValueError("pod job requires a positive max_hours")
    _positive_int(cfg.get("read_batch",2),"read_batch")
    _positive_int(cfg.get("passkey_batch",1),"passkey_batch")
    root = Path(cfg.get("out_dir", "outputs/s14"))
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("logs").mkdir(exist_ok=True)
    if (root / "DONE").exists():
        raise FileExistsError("pod job already DONE; preserve it and use a fresh output root")
    (root / "FAILED.json").unlink(missing_ok=True)
    deadline = time.monotonic() + float(cfg["max_hours"]) * 3600
    phase = "preflight"
    try:
        for cell in cells:
            require_no_abort(root / "runs" / cfg.get("preset", "mini-m") /
                             cell["armdir"] / f"s{cell['seed']}")
        canary = cfg.get("canary", {})
        n_p = (dict(n_p=int(canary.get("steps",30)),p_micro_batch=int(canary.get("micro_batch",4)),
                    p_so_micro_batch=int(canary.get("so_micro_batch",1)),
                    t_so_ms=N3B_T_SO*1000,t_p_ms=N3B_T_FO*1000)
               if canary.get("enabled") else resolve_n_p(cfg, root))
        needs_p = any(c["armdir"] == "mini-p" for c in cells)
        if needs_p and not n_p and not cfg.get("pilot", {}).get("enabled", False):
            raise ValueError("MINI-P requires n_p/n_p_file before this pod starts")
        if needs_p and n_p and not canary.get('enabled') and cfg.get('preset','mini-m') != 'tiny':
            for field in ('p_micro_batch','p_so_micro_batch'):
                _positive_int(n_p.get(field), 'pilot '+field)
            compute_context(cfg,dict(armdir='mini-p'),n_p)
        phase = "data"
        need_c = cfg.get("preset", "mini-m") != "tiny" and (cfg.get("data",{}).get("verify_c",False) or any(c["armdir"] in (
                 "fo-cm", "wsd-3469") for c in cells) or (needs_p and n_p and n_p["n_p"] > 3200))
        execute_worker("data", dict(cfg=cfg, root=str(root), need_c=bool(need_c)), root,
                       f"{cfg['process_name']}-prep", deadline)
        if cfg.get("passkey", False):
            phase = 'passkey_preflight'
            execute_worker("passkey", dict(preflight=True, model_path="", cell_config_path="",
                          eta_mults=[1.], out_dir="", data_dir=cfg['data_dir'],
                          n_docs=cfg.get('passkey_docs',256) if cfg.get('preset')=='tiny' else 256),
                          root, f"{cfg['process_name']}-preflight", deadline)
        phase = "pilot"
        if cfg.get("pilot", {}).get("enabled", False) and n_p is None:
            n_p = timing_pilot(cfg, root, deadline)
        if needs_p and n_p["n_p"] > 3200 and not need_c and cfg.get("preset", "mini-m") != "tiny":
            execute_worker("data", dict(cfg=cfg, root=str(root), need_c=True), root,
                           f"{cfg['process_name']}-prep-c", deadline)
        # Resolve and validate every cell before the first training worker starts.
        bases = [(c, cell_config(cfg, c, root, n_p)) for c in cells]
        for c, base in bases:
            require_no_abort(Path(base["out_dir"]))
            compute_context(cfg, c, n_p)  # reject incomplete compute evidence before training
        plans = []
        for c, base in bases:
            armdir, seed, preset = c["armdir"], c["seed"], base["preset"]
            cpath = root / ".cell-configs" / f"{preset}-{armdir}-s{seed}.json"
            marker = Path(base["out_dir"]) / "MINI_CELL_DONE.json"
            identity = fingerprint(base)
            if cpath.exists() and fingerprint(json.loads(cpath.read_text())) != identity:
                raise ValueError(f"refusing to reuse changed cell config: {cpath}")
            _json(cpath, base)
            phase = f"train:{armdir}:s{seed}"
            if marker.exists():
                result = json.loads(marker.read_text())
                if result.get("config_sha256") != identity:
                    raise ValueError(f"completed cell config changed: {marker}")
            else:
                work = dict(base)
                work["resume"] = bool(list(Path(base["resume_dir"]).glob("resume_step*.pt")))
                work["_compute_context"] = compute_context(cfg, c, n_p)
                result = execute_worker("train", work, root, base["process_name"], deadline)
            _require_training(base, result)
            hashes = {p.name: sha256_file(p) for p in Path(base['out_dir']).iterdir()
                      if p.is_file() and p.name != 'MINI_CELL_DONE.json'}
            _json(marker, {**result, "status": "done", "config_sha256": identity,
                           "output_sha256": hashes})
            plans.append((c, base, cpath))
        for c, base, cpath in plans:
            armdir, seed = c["armdir"], c["seed"]
            phase = f"reads:{armdir}:s{seed}"
            readcfg = read_config(cfg, base, armdir, root)
            item = readcfg["items"][0]
            cache_path, cache = read_cache(item, cfg)
            missing_frontier = [n for n in cache["read_list"] if n.startswith("g0_")
                                and n not in cache["npz_sha256"]]
            if missing_frontier:
                readcfg.update(read_cache_path=str(cache_path), read_cache=cache)
                execute_worker("reads", readcfg, root, readcfg["process_name"], deadline)
                cache = json.loads(cache_path.read_text())
                # Also supports workers without progress callbacks (e.g. test stubs).
                for n in missing_frontier:
                    record_read(cache_path, cache, n)
            if cfg.get("passkey", False):
                phase = f"passkey:{armdir}:s{seed}"
                eta = [e for e in passkey_etas_for(cfg, armdir)
                       if f"passkey_{read_name(e)}.npz" not in cache["npz_sha256"]]
                noadapt = "passkey_noadapt.npz" not in cache["npz_sha256"]
                if eta or noadapt:
                    if cfg.get('passkey_batch',1) > 1:
                        # Check/fallback covers the whole checkpoint, including
                        # conditions cached before an interrupted worker.
                        eta = list(passkey_etas_for(cfg,armdir))
                        noadapt = True
                    execute_worker("passkey", dict(model_path=item["ckpt"], cell_config_path=str(cpath),
                         eta_mults=eta, out_dir=item["out"], n_docs=int(cfg.get("passkey_docs", 256)),
                         seed=int(cfg.get("passkey_seed", 1014)), device=base["device"],
                         data_dir=base["data_dir"], noadapt=noadapt,
                         batch=_positive_int(cfg.get("passkey_batch", 1), "passkey_batch")), root,
                         f"{base['process_name']}-passkey", deadline)
                    for n in [f"passkey_{read_name(e)}.npz" for e in eta] + (
                            ["passkey_noadapt.npz"] if noadapt else []):
                        record_read(cache_path, cache, n)
        if canary.get("enabled"):
            phase = "canary_bench"
            execute_worker("canary_bench",dict(cfg=cfg,root=str(root)),root,
                           f"{cfg['process_name']}-bench",deadline)
            phase = "canary_analysis"
            execute_worker("canary_analysis",dict(cfg=cfg,root=str(root)),root,
                           f"{cfg['process_name']}-analysis",deadline)
        phase = "manifest"
        if time.monotonic() > deadline:
            raise TimeoutError("pod max_hours reached before manifest")
        manifest = publish_manifest(root, cfg)
        return dict(status="done", cells=len(cells), files=len(manifest["files"]), out_dir=str(root))
    except BaseException as exc:
        _json(root / "FAILED.json", dict(phase=phase, error=f"{type(exc).__name__}: {exc}"))
        raise


def worker(phase, payload, name):
    _set_process_name(name)
    if phase == "data":
        return prepare_data(payload["cfg"], Path(payload["root"]), need_c=payload["need_c"])
    if phase == "passkey":
        import torch
        torch.set_num_threads(4)
        return passkey_hook(payload)
    if phase in ("canary_bench", "canary_analysis"):
        from .s14_canary import benchmark, analyze
        return (benchmark if phase == "canary_bench" else analyze)(payload["cfg"],Path(payload["root"]))
    if phase == "train":
        import torch
        from .train import TrainCfg, train
        torch.set_num_threads(4)
        payload["process_name"] = name
        context = payload.pop("_compute_context", None)
        final = train(TrainCfg.from_dict(payload))
        cuda = torch.device(payload["device"]).type == "cuda"
        runtime = dict(peak_gb=final.get("peak_gb", torch.cuda.max_memory_allocated()/2**30 if cuda else 0.),
                       device_name=torch.cuda.get_device_name() if cuda else "cpu",
                       torch_version=torch.__version__, cuda_version=torch.version.cuda,
                       runpod_pod_id=os.environ.get("RUNPOD_POD_ID"))
        compute = compute_record(payload, final, runtime, context)
        result = dict(final=final, compute=compute, **compute)
        print("MINI_CELL_DONE " + json.dumps(result), flush=True)
        return result
    if phase == "reads":
        import numpy as np
        result = frontier_reads(payload)
        for item in payload["items"]:
            out = Path(item["out"])
            selected = None
            for read in item["reads"]:
                with np.load(out / f"g0_read_{read}.npz", allow_pickle=False) as z:
                    if len(z["M_doc"]) != payload["n_docs"] or len(z["L_adapt_sum"]) != payload["n_ho_adapt"]:
                        raise RuntimeError("frontier read did not cover the registered document counts")
                    if selected is not None and not np.array_equal(selected, z["eval_rows"]):
                        raise RuntimeError("frontier reads changed eval rows")
                    selected = z["eval_rows"].copy()
            with np.load(out / "g0_hlm.npz", allow_pickle=False) as z:
                if len(z["heldout_sum"]) != item["overrides"]["ho_docs"]:
                    raise RuntimeError("H_LM did not cover the registered document count")
        return {k: result[k] for k in ("done", "skipped", "errors", "wall_s")}
    raise ValueError(f"unknown S14 worker phase: {phase}")


def frontier_reads(payload):
    """Keep V0 at batch 2; authorize each checkpoint's faster reads separately."""
    from copy import deepcopy
    from .mini import mini_g0
    from .s14_batch import frontier_comparison, save_check

    def run(config):
        result = mini_g0(config)
        if result['errors'] or result['skipped']:
            raise RuntimeError(f"incomplete frontier reads: {result}")
        return result

    requested = payload['read_batch']
    if requested <= 2:
        return run(payload)
    if len(payload['items']) != 1:
        raise ValueError('S14 batch check requires one checkpoint per worker')
    item = payload['items'][0]
    reference = Path(item['out'])/'batch_reference'
    refcfg = deepcopy(payload)
    for key in ('read_cache','read_cache_path'):
        refcfg.pop(key,None)
    refcfg.update(read_batch=2, kappa_reads=['ce'], out_dir=str(reference))
    refcfg['items'][0].update(reads=['ce'],out=str(reference),hlm=True)
    run(refcfg)  # exact, fail-closed V0 against the in-training endpoint
    candidate = deepcopy(refcfg)
    candidate.update(read_batch=requested, endpoint_parity_read=False, out_dir=item['out'])
    candidate['items'][0]['out'] = item['out']
    run(candidate)
    comparison = frontier_comparison(reference,item['out'])
    effective = requested if comparison['status']=='PASS' else 2
    comparison.update(requested_batch=requested,reference_batch=2,effective_batch=effective,
                      checkpoint_sha256=sha256_file(item['ckpt']),
                      cell_config_sha256=sha256_file(item['cell_config']))
    save_check(item['out'],'frontier',comparison)
    print(('BATCH_CHECK' if effective==requested else 'BATCH_FALLBACK')+' '+json.dumps(comparison),flush=True)
    main = deepcopy(payload)
    main.update(read_batch=effective, endpoint_parity_read=False)
    # A failed check discards *all* large-batch/cache outputs for this checkpoint.
    cache = main.get('read_cache',{})
    cache['checkpoint_sha256'] = comparison['checkpoint_sha256']
    cache['npz_sha256'] = {n:h for n,h in cache.get('npz_sha256',{}).items() if not n.startswith('g0_')}
    if effective==requested:
        for filename in ('g0_read_ce.npz','g0_hlm.npz'):
            cache['npz_sha256'][filename] = sha256_file(Path(item['out'])/filename)
    main['read_cache'] = cache
    return run(main)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME"))
    args = ap.parse_args(argv)
    result = worker(args.worker, json.loads(Path(args.config_json).read_text()), args.process_name)
    print("S14_WORKER_DONE " + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
