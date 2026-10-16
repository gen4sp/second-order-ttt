"""SOQ-N3B: FO at equal training GPU-seconds (FO-CM) and the two-sided frontier reads.

Mode ``soq_n3b_fo_cm`` is a torch-free coordinator (every torch process is a matrix worker):

1. pre-flight: the train splits cover the FO-CM horizon, every extra split is present and
   non-empty, and no document of an extension split (C) overlaps eval / ho / calib (sha1 +
   256-token prefix; the registered B's overlaps -- flagged near_dup in P2-1 -- are logged);
2. a full read of the train split files (page cache), then the timing pilot: SO (``e2e2``) at
   micro_batch {1, 2} and FO-CE at {2, 4} (the settings that fit in memory), 10 warm-up + 50
   timed steps each, the median step time; one micro-batch of each arm under torch's FLOP
   counter;
3. r = t(fastest SO) / t(fastest FO), N_CM = round(1200 r) (``n_cm.json``, reused on restart);
4. the FO-CM matrix: FO-CE recipe (same init/data seeds as g43-g45) with the cosine stretched to
   N_CM, outer and inner warmup round(0.1 N_CM), stream A, B then C (C from step
   (|A|+|B|)/16 + 1 on), the fastest FO micro_batch, the running-minimum guard, G-div;
5. cells whose guard asks for a restart run attempt 1 (outer peak 1e-3) under ``attempt1/``;
6. ``n3b_summary.json``: training seconds against the SO target (+-2 %), guard labels.

Mode ``soq_n3b_reads`` resolves ``{N_CM}`` in a ``mini_g0`` config and runs it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

from . import matrix

BASE_STEPS = 1200
R_PRIOR = 2.96
PILOT_SO = "e2e2"
PILOT_FO = "e2e1-ce"


def n_cm_from_r(r: float, base_steps: int = BASE_STEPS) -> int:
    return int(round(base_steps * r))


def fo_cm_overrides(n_cm: int, base: dict, *, fo_micro_batch: int, target_s: float,
                    boundary_step: int, extra: dict | None = None,
                    base_steps: int = BASE_STEPS) -> dict:
    """Cell keys of an FO-CM seed on top of the FO-CE production config ``base``: the schedule
    stretched to ``n_cm`` (warmup and inner warmup keep their fraction of the horizon), evals
    at the FO-CE points plus 2 x base and N_CM (full at base, 2 x base and N_CM), weights kept
    at base and 2 x base, the endpoint state kept, a kept resume state at the A+B/C boundary."""
    frac_w = float(base["warmup"]) / float(base["steps"])
    frac_iw = float(base.get("inner_warmup", base["warmup"])) / float(base["steps"])
    full = sorted({s for s in (base_steps, 2 * base_steps) if s < n_cm} | {n_cm})
    evals = sorted({int(s) for s in base["eval_steps"] if s <= n_cm} | set(full))
    out = dict(steps=int(n_cm), warmup=int(round(frac_w * n_cm)),
               inner_warmup=int(round(frac_iw * n_cm)), eval_steps=evals,
               full_eval_steps=full, save_model_steps=[s for s in full if s != n_cm],
               keep_final_state=True, micro_batch=int(fo_micro_batch),
               log_train_seconds=True, train_seconds_target=float(target_s),
               keep_resume_steps=[int(boundary_step)] if 0 < boundary_step < n_cm else [])
    out.update(extra or {})
    return out


def _rows(data_dir: Path, split: str) -> int:
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def preflight(cfg: dict, base_fo: dict, n_cm: int | None = None) -> dict:
    """Fail fast before any GPU time: split C present / non-empty / uncontaminated (extension
    splits only; B's counts are logged) and, once N_CM is known, the train rows cover it.
    Returns the row counts and boundary step."""
    from .data import contamination_enforced, split_contamination

    data = Path(cfg.get("data_dir", base_fo["data_dir"]))
    train_split = base_fo.get("train_split", "A")
    extra = list(cfg.get("fo_cm", {}).get("extra_train_splits", ["B", "C"]))
    gb = int(base_fo["global_batch"])
    rows = {s: _rows(data, s) for s in [train_split, *extra]}
    for s in extra:
        if rows[s] == 0 or not (data / f"{s}.bin").exists() or (data / f"{s}.bin").stat().st_size == 0:
            raise ValueError(f"split {s!r} is missing or empty under {data}: prepare it first "
                             f"(mode='prepare', extend=true, extend_quotas={{{s} = N}})")
    contamination = []
    if cfg.get("contamination_check", True):
        for s in extra:
            rec = split_contamination(data, s, (base_fo.get("eval_split", "eval"),
                                                base_fo.get("ho_split", "ho"),
                                                base_fo.get("calib_split", "calib")))
            rec["enforced"] = contamination_enforced(data, s)
            print("N3B_CONTAMINATION " + json.dumps(rec, sort_keys=True), flush=True)
            contamination.append(rec)
            if rec["hits"] and rec["enforced"]:
                raise ValueError(f"split {s!r} overlaps eval/ho/calib: {rec}")
    last = extra[-1] if extra else train_split
    before_last = sum(v for k, v in rows.items() if k != last)
    rec = {"rows": rows, "global_batch": gb, "boundary_step": before_last // gb,
           "boundary_split": last, "contamination": contamination}
    if n_cm is not None and sum(rows.values()) < n_cm * gb:
        raise ValueError(f"train splits {rows} hold {sum(rows.values())} rows; N_CM={n_cm} needs "
                         f"{n_cm * gb}: extend split {last} (or add the next split)")
    return rec


def warm_page_cache(data_dir, splits, block: int = 64 << 20) -> dict:
    t0 = time.perf_counter()
    n = 0
    for s in splits:
        path = Path(data_dir) / f"{s}.bin"
        with path.open("rb") as f:
            while chunk := f.read(block):
                n += len(chunk)
    return {"splits": list(splits), "bytes": n, "wall_s": time.perf_counter() - t0}


def _parse_done(log: Path) -> dict | None:
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if line.startswith("MINI_CELL_DONE "):
            return json.loads(line.split(" ", 1)[1])
    return None


def _oom(log: Path) -> bool:
    text = log.read_text(errors="replace")
    return "OutOfMemoryError" in text or "CUDA out of memory" in text


def _run_worker(cell_cfg: dict, path: Path, log: Path, name: str, *, flops: bool = False) -> int:
    import subprocess
    import sys

    path.write_text(json.dumps(cell_cfg, sort_keys=True) + "\n")
    env = dict(os.environ, PROCESS_NAME=name, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    cmd = ([sys.executable, "-m", "tttmem.p2.soq_n3b", "--flops", str(path)] if flops else
           [sys.executable, "-m", "tttmem.p2.matrix", "--cell-json", str(path)])
    with log.open("a") as f:
        return subprocess.run(cmd + ["--process-name", name], stdout=f, stderr=subprocess.STDOUT,
                              env=env).returncode


def pilot_settings(cfg: dict) -> list[tuple[str, int]]:
    p = cfg.get("pilot", {})
    return ([(PILOT_SO, int(m)) for m in p.get("so_micro_batch", [1, 2])]
            + [(PILOT_FO, int(m)) for m in p.get("fo_micro_batch", [2, 4])])


def timing_pilot(cfg: dict, repo: Path, out: Path, name_of) -> dict:
    """Every pilot setting in its own worker; a setting that runs out of memory is recorded
    and skipped.  The median of the timed steps is the step time."""
    p = cfg.get("pilot", {})
    warm, timed = int(p.get("warmup_steps", 10)), int(p.get("timed_steps", 50))
    root = out / "pilot"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (arm, mb) in enumerate(pilot_settings(cfg), start=1):
        cell = {"preset": cfg.get("preset", "mini-m"), "arm": arm, "init_seed": 0, "data_seed": 0,
                "overrides": dict(cfg.get("pilot_overrides", {}))}
        base = matrix._base_config(cfg, cell, repo)
        if int(base["global_batch"]) % mb:
            rows.append({"arm": arm, "micro_batch": mb, "status": "skip_global_batch"})
            continue
        cdir = root / f"{arm}-mb{mb}"
        base.update(out_dir=str(cdir), resume=False, resume_dir=str(cdir / ".resume"),
                    micro_batch=mb, stop_after=warm + timed, eval_steps=[], full_eval_steps=[],
                    eval_at_init=False, save_checkpoints=False, guard_every=0, gdiv=False,
                    pilot_steps=[], sizing_steps=[], assay_gate=False, resume_every=10 ** 6,
                    max_hours=0, log_train_seconds=True)
        for key in ("data_dir", "code_data_dir", "device"):
            if key in cfg:
                base[key] = cfg[key]
        name = name_of(f"pilot{k}")
        log = root / f"{arm}-mb{mb}.log"
        import shutil
        shutil.rmtree(cdir, ignore_errors=True)  # restart-safe: a half-done setting reruns fresh
        log.unlink(missing_ok=True)              # and an old OOM line cannot mislabel a new error
        t0 = time.perf_counter()
        rc = _run_worker(base, root / f"{arm}-mb{mb}.json", log, name)
        rec = {"arm": arm, "micro_batch": mb, "wall_s": time.perf_counter() - t0}
        done = _parse_done(log) if rc == 0 else None
        if done is None:
            rec["status"] = "oom" if _oom(log) else f"error_rc{rc}"
            if rec["status"] != "oom":
                raise RuntimeError(f"timing pilot {arm} mb{mb} failed (rc={rc}); see {log}")
        else:
            ms = list(done["final"].get("step_ms", []))[warm:warm + timed]
            rec.update(status="ok", median_ms=statistics.median(ms), mean_ms=statistics.fmean(ms),
                       n_timed=len(ms), peak_gb=done.get("peak_gb"))
            cap = p.get("max_peak_gb")  # "fits" = pilot peak <= cap (headroom for evals / probes)
            if cap is not None and rec.get("peak_gb") is not None and float(rec["peak_gb"]) > float(cap):
                rec["status"] = "over_peak"
        print("N3B_PILOT " + json.dumps(rec, sort_keys=True), flush=True)
        rows.append(rec)
        if not cfg.get("keep_pilot_dirs", False):
            import shutil
            shutil.rmtree(cdir, ignore_errors=True)
    flops = {}
    if p.get("count_flops", True):
        for arm in (PILOT_SO, PILOT_FO):
            cell = {"preset": cfg.get("preset", "mini-m"), "arm": arm, "init_seed": 0,
                    "data_seed": 0, "overrides": dict(cfg.get("pilot_overrides", {}))}
            base = matrix._base_config(cfg, cell, repo)
            for key in ("data_dir", "code_data_dir", "device"):
                if key in cfg:
                    base[key] = cfg[key]
            base.update(micro_batch=1, out_dir=str(root / f"{arm}-flops"))
            log = root / f"{arm}-flops.log"
            rc = _run_worker(base, root / f"{arm}-flops.json", log, name_of(f"flops-{arm}"), flops=True)
            line = next((ln for ln in reversed(log.read_text(errors="replace").splitlines())
                         if ln.startswith("N3B_FLOPS ")), None)
            flops[arm] = json.loads(line.split(" ", 1)[1]) if rc == 0 and line else {"error": rc}
    return {"settings": rows, "flops": flops, "warmup_steps": warm, "timed_steps": timed}


def decide_n_cm(pilot: dict, base_steps: int = BASE_STEPS, r_prior: float = R_PRIOR,
                r_flag_tol: float = 0.10) -> dict:
    ok = [r for r in pilot["settings"] if r.get("status") == "ok"]
    so = [r for r in ok if r["arm"] == PILOT_SO]
    fo = [r for r in ok if r["arm"] == PILOT_FO]
    if not so or not fo:
        raise RuntimeError(f"timing pilot has no usable SO or FO setting: {pilot['settings']}")
    best_so = min(so, key=lambda r: r["median_ms"])
    best_fo = min(fo, key=lambda r: r["median_ms"])
    r = best_so["median_ms"] / best_fo["median_ms"]
    fl = pilot.get("flops", {})
    ratio = None
    if all("flops_per_seq" in fl.get(a, {}) for a in (PILOT_SO, PILOT_FO)):
        ratio = fl[PILOT_SO]["flops_per_seq"] / fl[PILOT_FO]["flops_per_seq"]
    return {"r": r, "r_prior": r_prior, "r_flag": abs(r / r_prior - 1.0) > r_flag_tol,
            "r_flag_tol": r_flag_tol,
            "n_cm": n_cm_from_r(r, base_steps), "base_steps": base_steps,
            "so_micro_batch": best_so["micro_batch"], "fo_micro_batch": best_fo["micro_batch"],
            "t_so_ms": best_so["median_ms"], "t_fo_ms": best_fo["median_ms"],
            "train_seconds_target": best_so["median_ms"] / 1000.0 * base_steps,
            "flop_ratio_counted": ratio, "pilot": pilot}


def _cell_results(root: Path, cells: list[dict]) -> list[dict]:
    res = []
    for c in cells:
        marker = matrix._cell_out(root, c) / "MINI_CELL_DONE.json"
        rec = json.loads(marker.read_text()) if marker.exists() else {"status": "missing"}
        fin = rec.get("final", {}) or {}
        res.append({"cell": f"s{c['init_seed']}", "status": rec.get("status"),
                    "aborted": fin.get("aborted"), "label": fin.get("label"),
                    "guard_label": fin.get("guard_label"), "steps": fin.get("steps"),
                    "train_seconds": fin.get("train_seconds"), "final_state": fin.get("final_state")})
    return res


def soq_n3b_fo_cm(cfg: dict) -> dict:
    from .data import _set_process_name

    nn = int(cfg.get("process_nn", 2))
    prefix = cfg.get("name_prefix", "TTM-n3b")
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", f"{prefix}-{nn:02d}")
    _set_process_name(name)
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    base_steps = int(cfg.get("base_steps", BASE_STEPS))
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    fo_cell = {"preset": cfg.get("preset", "mini-m"), "arm": PILOT_FO, "init_seed": seeds[0],
               "data_seed": seeds[0], "overrides": dict(cfg.get("pilot_overrides", {}))}
    base_fo = matrix._base_config(cfg, fo_cell, repo)
    for key in ("data_dir", "code_data_dir", "device"):
        if key in cfg:
            base_fo[key] = cfg[key]
    pre = preflight(cfg, base_fo)
    print("N3B_PREFLIGHT " + json.dumps(pre, sort_keys=True), flush=True)
    n_cm_path = out / "n_cm.json"
    if not n_cm_path.exists() and cfg.get("n_cm_file"):
        # Registered layout: one pilot job, one FO-CM job per seed with the pilot's N_CM.
        n_cm_path.write_text(Path(cfg["n_cm_file"]).read_text())
    if n_cm_path.exists():
        dec = json.loads(n_cm_path.read_text())
        print("N3B_N_CM_REUSED " + json.dumps({k: dec[k] for k in ("r", "n_cm")}), flush=True)
    else:
        warm = warm_page_cache(base_fo["data_dir"], [base_fo.get("train_split", "A"),
                               *cfg.get("fo_cm", {}).get("extra_train_splits", ["B", "C"])])
        print("N3B_PAGE_CACHE " + json.dumps(warm, sort_keys=True), flush=True)
        pilot = timing_pilot(cfg, repo, out, lambda s: f"{prefix}-{nn:02d}-{s}")
        dec = decide_n_cm(pilot, base_steps, float(cfg.get("r_prior", R_PRIOR)),
                          float(cfg.get("r_flag_tol", 0.10)))
        dec["page_cache"] = warm
        n_cm_path.write_text(json.dumps(dec, indent=2, sort_keys=True) + "\n")
    print("N3B_N_CM " + json.dumps({k: v for k, v in dec.items() if k != "pilot"}, sort_keys=True),
          flush=True)
    if dec["r_flag"]:
        print("N3B_R_FLAG " + json.dumps({"r": dec["r"], "r_prior": dec["r_prior"],
              "tol": dec.get("r_flag_tol"), "note": "|r/r_prior - 1| > tol; the pilot r is used"}),
              flush=True)
    if cfg.get("pilot_only", False):
        return {"pilot_only": True, **{k: v for k, v in dec.items() if k != "pilot"}}
    n_cm = int(dec["n_cm"])
    pre = preflight({**cfg, "contamination_check": False}, base_fo, n_cm)
    over = fo_cm_overrides(n_cm, base_fo, fo_micro_batch=dec["fo_micro_batch"],
                           target_s=dec["train_seconds_target"],
                           boundary_step=pre["boundary_step"], extra=cfg.get("fo_cm", {}),
                           base_steps=base_steps)
    hours = (n_cm * dec["t_fo_ms"] / 3.6e6) * float(cfg.get("hours_margin", 1.25))
    mcfg = {k: v for k, v in cfg.items() if k not in ("mode", "pilot", "fo_cm", "seeds", "items")}
    mcfg.update(out_dir=str(out), workers=1, process_nn=nn, name_prefix=prefix,
                registered_root=str(out), analyse=False,
                podsmoke_hours={f"{fo_cell['preset']}/{PILOT_FO}": hours},
                cells=[{"preset": fo_cell["preset"], "arm": PILOT_FO, "init_seed": s,
                        "data_seed": s, "steps": n_cm, "resume_every": int(over.get("resume_every", 100)),
                        "overrides": {**cfg.get("pilot_overrides", {}), **over}} for s in seeds])
    print("N3B_FO_CM_PLAN " + json.dumps({"n_cm": n_cm, "overrides": over, "est_hours_per_seed": hours},
                                         sort_keys=True), flush=True)
    matrix.run(mcfg, repo_root=repo)
    cells = _cell_results(out, mcfg["cells"])
    restart = [c for c, rec in zip(mcfg["cells"], cells) if rec.get("label") == "RESTART"]
    attempt1 = []
    if restart and cfg.get("auto_attempt1", True):
        a1 = dict(mcfg, out_dir=str(out / "attempt1"), registered_root=str(out / "attempt1"),
                  cells=[dict(c, overrides=dict(c["overrides"], attempt=1, lr_peak=1e-3))
                         for c in restart])
        matrix.run(a1, repo_root=repo)
        attempt1 = _cell_results(out / "attempt1", a1["cells"])
    summary = {"n_cm": n_cm, "r": dec["r"], "r_flag": dec["r_flag"], "cells": cells,
               "attempt1": attempt1, "so_attempt1_required": bool(restart),
               "within_2pct": all((c.get("train_seconds") or {}).get("within_2pct") is True
                                  for c in (attempt1 or cells) if not c.get("aborted"))}
    (out / "n3b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("N3B_DONE " + json.dumps({k: summary[k] for k in ("n_cm", "r", "within_2pct",
                                                             "so_attempt1_required")}), flush=True)
    return summary


def _resolve(value, n_cm: int):
    if isinstance(value, str):
        if value == "{N_CM}":
            return n_cm
        return value.replace("{N_CM}", str(n_cm)).replace("{N_CM:04d}", f"{n_cm:04d}")
    if isinstance(value, list):
        return [_resolve(v, n_cm) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, n_cm) for k, v in value.items()}
    return value


def resolve_reads(cfg: dict, n_cm: int | None) -> dict:
    """Items of the ``run_groups`` (default: all) with ``{N_CM}`` / ``{N_CM:04d}`` resolved; an
    item with ``drop_if_n_cm_at_most`` >= N_CM is dropped (no checkpoint at its step)."""
    groups = cfg.get("run_groups")
    chosen = [it for it in cfg.get("items", []) if groups is None or it.get("group") in groups]
    if n_cm is None:
        if "{N_CM" in json.dumps(chosen):
            raise ValueError("these read items need N_CM: set n_cm_file (the FO-CM n_cm.json) or n_cm")
        n_cm = -1
    out = _resolve({k: v for k, v in cfg.items() if k not in ("items", "run_groups")}, n_cm)
    items = []
    for it in chosen:
        lim = it.get("drop_if_n_cm_at_most")
        if lim is not None and n_cm <= int(lim):
            print("N3B_READ_DROP " + json.dumps({"cell": it.get("cell"), "n_cm": n_cm}), flush=True)
            continue
        items.append({k: v for k, v in _resolve(it, n_cm).items()
                      if k not in ("drop_if_n_cm_at_most", "group")})
    out["items"] = items
    return out


def soq_n3b_reads(cfg: dict) -> dict:
    from .mini import mini_g0

    if cfg.get("n_cm") is not None:
        n_cm = int(cfg["n_cm"])
    elif cfg.get("n_cm_file") and Path(cfg["n_cm_file"]).exists():
        n_cm = int(json.loads(Path(cfg["n_cm_file"]).read_text())["n_cm"])
    else:
        n_cm = None
    resolved = resolve_reads({k: v for k, v in cfg.items() if k not in ("n_cm_file", "n_cm")}, n_cm)
    print("N3B_READS " + json.dumps({"n_cm": n_cm, "items": len(resolved["items"])}), flush=True)
    return mini_g0(resolved)


def count_flops(cell_json: str) -> dict:
    """FLOPs of one outer micro-batch (forward, inner loop and backward) under torch's FLOP
    counter, at the cell's micro_batch on its first stream rows."""
    import numpy as np
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from .data import open_split
    from .meta import InnerCfg
    from .train import ARMS, TrainCfg, build, outer_microstep

    torch.set_num_threads(4)
    d = json.loads(Path(cell_json).read_text())
    cfg = TrainCfg.from_dict({k: v for k, v in d.items() if k != "process_name"})
    model, _ = build(cfg)
    model.train()
    kind, rule = ARMS[cfg.arm]
    arr, _ = open_split(cfg.data_dir, cfg.train_split)
    rows = torch.from_numpy(np.asarray(arr[:cfg.micro_batch, :cfg.seq_len + 1]).astype(np.int64))
    rows = rows.to(cfg.device)
    inner = InnerCfg(rule=rule, lr=cfg.rule_lr(rule), clip=cfg.inner_clip, r_ref=cfg.r_ref,
                     seed=cfg.data_seed)
    inner.row_ids = torch.arange(cfg.micro_batch, device=rows.device)
    with FlopCounterMode(display=False) as fc:
        outer_microstep(model, rows, cfg, kind, inner)
    total = int(fc.get_total_flops())
    rec = {"arm": cfg.arm, "micro_batch": cfg.micro_batch, "seq_len": cfg.seq_len,
           "flops": total, "flops_per_seq": total / cfg.micro_batch}
    print("N3B_FLOPS " + json.dumps(rec, sort_keys=True), flush=True)
    return rec


def main(argv=None):
    import argparse

    from .data import _set_process_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--flops")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-n3b-flops"))
    args = ap.parse_args(argv)
    _set_process_name(args.process_name)
    if not args.flops:
        raise SystemExit("--flops <cell.json> required")
    count_flops(args.flops)


if __name__ == "__main__":
    main()
