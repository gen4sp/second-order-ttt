"""Single-process CUDA/CPU MINI measurement harness.

The harness only reads production configs and writes under its configured MINI root.
It is intentionally usable on CPU with a synthetic P2-1-layout store for CI.
"""
from __future__ import annotations

import json
import gc
import os
import statistics
import subprocess
import sys
import time
import shutil
from pathlib import Path
import tomllib

import torch

from .meta import inner_logit_cotangent, nc_control_mask

from .train import TrainCfg, train

CORE_ARMS = ("e2e2", "e2e1-ce", "e2e1-rp", "e2e1-nc")
MIX_ARMS = ("e2e2-mix", "e2e1-ce-mix")


def _head_benchmark(cfg):
    """Head-dominated CPU probe used to catch RP/NC regressions (V=32768)."""
    if not cfg.get("head_benchmark", False):
        return []
    torch.set_num_threads(4)
    B, C, V = 2, 1024, 32768
    z = torch.randn(B, C, V)
    y = torch.randint(V, (B, C))
    w = torch.ones(B, C)
    classes = torch.ones(B, C, dtype=torch.long)
    active = torch.rand(B, C) < .2
    rows = []
    for ref in (.03, .99):
        for rule in ("ce", "rp", "nc"):
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                if rule == "nc":
                    nc_control_mask(active, classes, seed=3, step=2, chunk=0)
                    inner_logit_cotangent("rp", z, y, w, r_ref=ref)
                else:
                    inner_logit_cotangent(rule, z, y, w, r_ref=ref)
                times.append((time.perf_counter() - t0) * 1000)
            rec = {"shape": [B, C, V], "r_ref": ref, "rule": rule,
                   "ms_eval": float(statistics.median(times)), "ms_step": float(statistics.median(times))}
            rows.append(rec)
            print("MINI_PODSMOKE_HEAD " + json.dumps(rec, sort_keys=True), flush=True)
    return rows


def _config_path(root: Path, preset: str, arm: str) -> Path:
    name = arm + ("-m" if preset == "mini-m" and not arm.endswith("-mix") else "") + ".toml"
    return root / "configs" / "mini" / name


def _load(path: Path) -> dict:
    with path.open("rb") as f:
        doc = tomllib.load(f)
    return dict(doc.get("p2", {}))


def _memory(device):
    if torch.device(device).type != "cuda":
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30)


def _run_arm(base: dict, arm: str, preset: str, out: Path, cfg: dict) -> dict:
    d = dict(base)
    configured_micro_batch = int(d.get("micro_batch", 1))
    calib_path = Path(d.get("data_dir", "")) / f"{d.get('calib_split', 'calib')}.jsonl"
    calib_n = sum(1 for _ in calib_path.open()) if calib_path.exists() else int(base.get("pilot_docs", 64))
    d.update({"arm": arm.removesuffix("-mix"), "preset": preset,
              "steps": int(cfg.get("steps", 20)), "stop_after": int(cfg.get("steps", 20)),
              "eval_steps": [int(cfg.get("steps", 20))],
              "full_eval_steps": [int(cfg.get("steps", 20))], "sizing_steps": [],
              "eval_at_init": bool(cfg.get("eval_at_init", True)),
              "init_eval_docs": min(32, int(base.get("n_eval", 1024))),
              "init_ho_docs": min(32, int(base.get("ho_docs", 256))),
              "pilot_steps": [], "assay_gate": False, "save_checkpoints": False,
              "n_eval": min(32, int(base.get("n_eval", 1024))),
              "ho_docs": min(32, int(base.get("ho_docs", 256))),
              "ho_trajectory_docs": min(32, int(base.get("ho_trajectory_docs", 128))),
              "pilot_docs": min(32, calib_n),
              "trajectory_docs": min(32, int(base.get("trajectory_docs", 256))),
              "min_probes": int(cfg.get("min_probes", 1 if preset == "tiny" else base.get("min_probes", 5))),
              "eval_batch": min(2, int(base.get("eval_batch", 2))),
              "out_dir": str(out), "process_name": f"TTM-min-{int(cfg.get('process_nn', 31)):02d}-podsmoke2-{preset}-{arm}",
              "measure_timing": True})
    # Podsmoke is a disposable measurement root.  Clearing one arm's prior
    # artifacts makes a re-run deterministic instead of tripping train's
    # fresh-output guard.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if torch.device(d.get("device", "cpu")).type == "cuda":
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    retry = False
    try:
        result = train(TrainCfg.from_dict(d))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or int(d["micro_batch"]) <= 1:
            raise
        retry = True
        d["micro_batch"] = max(1, int(d["micro_batch"]) // 2)
        d["out_dir"] = str(out.with_name(out.name + "-retry"))
        result = train(TrainCfg.from_dict(d))
    alloc, reserved = _memory(d.get("device", "cpu"))
    steps = result.get("step_ms", [])
    mid = steps[4:20] if len(steps) >= 20 else steps
    eval_ms = result.get("eval_ms", [])
    eval_total_ms = sum((x[1] if isinstance(x, (tuple, list)) else x) for x in eval_ms)
    measured_init = next((x[1] for x in eval_ms if isinstance(x, (tuple, list)) and x[0] == 0), 0.0)
    measured_last = next((x[1] for x in reversed(eval_ms) if isinstance(x, (tuple, list))), 0.0)
    # The smoke reads 32 trajectory and 32 held-out documents.  Scale those
    # measured calls to MINI's registered diet: SO has the shared init read;
    # all arms have two trajectory and one endpoint read, and mix adds one
    # 512-document code read. RP/NC's endpoint crossover makes this a safe
    # upper estimate for their trajectory calls.
    if d.get("eval_at_init", True):
        eval_s = (5.0 * measured_init + 30.0 * measured_last) / 1000.0
    else:
        eval_s = 30.0 * measured_last / 1000.0
    if arm.endswith("-mix"):
        eval_s += 16.0 * measured_last / 1000.0
    rec = {"preset": preset, "arm": arm, "configured_micro_batch": configured_micro_batch,
           "micro_batch": d["micro_batch"], "micro_batch_fits": not retry, "oom_retry": retry,
           "median_ms_step_5_20": float(statistics.median(mid)) if mid else float(result.get("wall_s", 0) * 1000 / max(1, d["steps"])),
           "max_memory_allocated_gb": alloc, "max_memory_reserved_gb": reserved,
           "wall_s": time.perf_counter() - started,
           "eval_ms": eval_ms, "final": result,
           "production_eval_steps": [300, 600, 1200],
           # t0/t20 are measured on 32 endpoint + 32 ho docs.  MINI's
           # production schedule is init + three trajectory/endpoint reads;
           # the registered conservative extrapolation scales the whole call
           # by 32 (the ho component then has slack).
           "eval_extrapolated_s": eval_s}
    print("MINI_PODSMOKE " + json.dumps(rec, sort_keys=True, allow_nan=True), flush=True)
    return rec


def _concurrency_cell(path: str, process_name: str) -> dict:
    """Child entry point for one short CUDA concurrency measurement."""
    with Path(path).open() as f:
        d = json.load(f)
    d["process_name"] = process_name
    if torch.device(d.get("device", "cpu")).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = train(TrainCfg.from_dict(d))
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.device(d.get("device", "cpu")).type == "cuda" else 0.0)
    rec = {"tokens_per_s": float(result.get("tok_per_s", 0.0)),
           "peak_memory_gb": peak, "steps": result.get("steps", 0)}
    print("MINI_CONCURRENCY_CELL " + json.dumps(rec, sort_keys=True), flush=True)
    return rec


def _run_concurrency_group(base: dict, preset: str, arm: str, k: int,
                           out: Path, nn_start: int) -> dict:
    """Spawn k identical 15-step cells and aggregate their measured throughput."""
    out.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for i in range(k):
        d = dict(base)
        d.update({"steps": 15, "stop_after": 15, "eval_steps": [], "full_eval_steps": [],
                  "sizing_steps": [], "pilot_steps": [], "assay_gate": False,
                  "n_eval": min(32, int(base.get("n_eval", 1024))),
                  "ho_docs": min(32, int(base.get("ho_docs", 256))),
                  "ho_trajectory_docs": min(32, int(base.get("ho_trajectory_docs", 128))),
                  "pilot_docs": min(32, int(base.get("pilot_docs", 64))),
                  "trajectory_docs": min(32, int(base.get("trajectory_docs", 256))),
                  "eval_batch": min(2, int(base.get("eval_batch", 2))), "save_checkpoints": False,
                  "out_dir": str(out / f"cell{i}"), "measure_timing": True})
        config = out / f"cell{i}.json"
        config.write_text(json.dumps(d, sort_keys=True) + "\n")
        log_path = out / f"cell{i}.log"
        log = log_path.open("w")
        name = f"TTM-min-{nn_start + i:02d}-conc-{preset}-{arm}-{i}"
        env = os.environ.copy()
        env.update({"PROCESS_NAME": name, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        p = subprocess.Popen([sys.executable, "-m", "tttmem.p2.podsmoke", "--concurrency-json",
                              str(config), "--process-name", name], stdout=log,
                             stderr=subprocess.STDOUT, env=env)
        procs.append(p); logs.append((log, log_path))
    results = []
    for p, (log, log_path) in zip(procs, logs):
        p.wait(); log.close()
        for line in reversed(log_path.read_text().splitlines()):
            if line.startswith("MINI_CONCURRENCY_CELL "):
                try:
                    results.append(json.loads(line.split(" ", 1)[1]))
                except json.JSONDecodeError:
                    pass
                break
    payload = {"preset": preset, "arm": arm, "k": k, "steps": 15,
               "aggregate_tokens_s": sum(x["tokens_per_s"] for x in results),
               "peak_memory_gb_per_process": [x["peak_memory_gb"] for x in results],
               "returncodes": [p.returncode for p in procs]}
    print("MINI_PODSMOKE_CONCURRENCY " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run(cfg: dict) -> dict:
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    out = Path(cfg.get("out_dir", "outputs/mini/podsmoke"))
    out.mkdir(parents=True, exist_ok=True)
    presets = cfg.get("presets", ["mini-s", "mini-m"])
    arms = cfg.get("arms", list(CORE_ARMS))
    config_dir = Path(cfg.get("config_dir", repo / "configs" / "mini"))
    if not config_dir.is_absolute():
        config_dir = repo / config_dir
    if cfg.get("include_mix", True):
        code_root = Path(cfg.get("code_data_dir", "data/mini-code"))
        if (code_root / "code_train.bin").exists() and (code_root / "code_eval.bin").exists():
            arms += [a for a in MIX_ARMS if a not in arms]
        else:
            print("MINI_PODSMOKE_SKIP mix code store missing", flush=True)
    head = _head_benchmark(cfg) if torch.device(cfg.get("device", "cpu")).type == "cpu" else []
    records = []
    for preset in presets:
        for arm in arms:
            if arm.endswith("-mix") and preset != cfg.get("mix_preset", "mini-s"):
                continue
            path = config_dir / (
                arm + ("-m" if preset == "mini-m" and not arm.endswith("-mix") else "") + ".toml")
            if preset == "tiny":
                # The CPU rehearsal uses the same P2-1 row store as
                # mini_smoke, while keeping the production runner's path.
                base = {"arm": arm.removesuffix("-mix"), "preset": "tiny", "dtype": "fp32",
                        "device": cfg.get("device", "cpu"), "seq_len": 64,
                        "global_batch": 16 if arm.endswith("-mix") else 2,
                        "micro_batch": 4 if arm.endswith("-mix") else 1,
                        "steps": 12, "warmup": 2, "lr_peak": 1e-2,
                        "barriers": [2], "segment_len": 16, "fused_attn": False,
                        "n_eval": 4, "ho_docs": 2, "ho_trajectory_docs": 2,
                        "trajectory_docs": 4, "pilot_docs": 2, "min_probes": 1,
                        "eval_batch": 2, "data_dir": cfg.get("data_dir", ""),
                        "code_data_dir": cfg.get("data_dir", ""), "mix": arm.endswith("-mix"),
                        "code_train_split": "code_train", "code_eval_split": "code_eval",
                        "code_eval_steps": [12], "eval_steps": [12], "full_eval_steps": [12],
                        "crossover_rules": [], "assay_gate": False, "pilot_steps": [],
                        "sizing_steps": [], "save_checkpoints": False}
                path = None
            if path is not None and not path.exists():
                if arm.endswith("-mix"):
                    continue
                raise FileNotFoundError(path)
            if path is not None:
                base = _load(path)
            if "device" in cfg:
                base["device"] = cfg["device"]
            if "data_dir" in cfg:
                base["data_dir"] = cfg["data_dir"]
            if "code_data_dir" in cfg:
                base["code_data_dir"] = cfg["code_data_dir"]
            try:
                records.append(_run_arm(base, arm, preset, out / preset / arm, cfg))
            except RuntimeError as exc:
                rec = {"preset": preset, "arm": arm, "error": str(exc)}
                print("MINI_PODSMOKE " + json.dumps(rec, sort_keys=True), flush=True)
                records.append(rec)
    concurrency = []
    if torch.device(cfg.get("device", "cpu")).type == "cuda":
        gc.collect(); torch.cuda.empty_cache()
        # Run real short cells. Each group is waited before the next group, so
        # at most three CUDA training processes coexist.
        conc_nn = int(cfg.get("process_nn_start", 20))
        for preset, ks in (("mini-s", (1, 2, 3)), ("mini-m", (1, 2))):
            if preset not in presets:
                continue
            for arm in ("e2e2", "e2e1-ce", "e2e1-rp", "e2e1-nc"):
                path = config_dir / (
                    arm + ("-m" if preset == "mini-m" else "") + ".toml")
                if not path.exists():
                    continue
                base = _load(path)
                base.update({"device": cfg.get("device", base.get("device", "cuda")),
                             "data_dir": cfg.get("data_dir", base.get("data_dir")),
                             "process_name": f"TTM-min-{int(cfg.get('process_nn', 31)):02d}-podsmoke2-concurrency"})
                for k in ks:
                    concurrency.append(_run_concurrency_group(
                        base, preset, arm, k,
                        out / "concurrency" / preset / arm / f"k{k}",
                        conc_nn))
                    conc_nn += k
    else:
        print("MINI_PODSMOKE_CONCURRENCY_SKIP cpu", flush=True)
    # Micro-batch probe is sequential and therefore does not affect the
    # production records.  ``micro_batch_probe`` enables it; it is cheap on CPU/tiny.
    micro = []
    if cfg.get("micro_batch_probe", False):
        for preset in presets:
            for arm in arms:
                if arm.endswith("-mix") and preset != cfg.get("mix_preset", "mini-s"):
                    continue
                path = config_dir / (
                    arm + ("-m" if preset == "mini-m" and not arm.endswith("-mix") else "") + ".toml")
                if not path.exists(): continue
                base = _load(path); configured = int(base.get("micro_batch", 1))
                choices = [1, 2, 4] if arm.startswith("e2e2") else [2, 4]
                for mb in choices:
                    d = dict(base, device=cfg.get("device", base.get("device", "cpu")),
                             data_dir=cfg.get("data_dir", base.get("data_dir")), micro_batch=mb)
                    d.update(steps=12, stop_after=12, eval_at_init=False, eval_steps=[], full_eval_steps=[],
                             sizing_steps=[], pilot_steps=[], assay_gate=False, save_checkpoints=False,
                             n_eval=min(32, int(base.get("n_eval", 1024))), ho_docs=min(32, int(base.get("ho_docs", 256))),
                             ho_trajectory_docs=min(32, int(base.get("ho_trajectory_docs", 64))),
                             trajectory_docs=min(32, int(base.get("trajectory_docs", 256))), eval_batch=2,
                             out_dir=str(out / "micro" / preset / arm / f"mb{mb}"),
                             process_name=f"TTM-min-{int(cfg.get('process_nn',31)):02d}-podsmoke2-micro-{preset}-{arm}-mb{mb}")
                    try:
                        rec = _run_arm(d, arm, preset, Path(d["out_dir"]), {"steps": 12})
                        rec.update(micro_batch=mb, configured_micro_batch=configured, fits=rec["max_memory_allocated_gb"] <= 20)
                    except RuntimeError as exc:
                        rec = dict(preset=preset, arm=arm, micro_batch=mb, oom="out of memory" in str(exc).lower(), error=str(exc))
                    micro.append(rec)
                    print("MINI_PODSMOKE_MICRO " + json.dumps(rec, sort_keys=True, allow_nan=True), flush=True)

    if cfg.get("p23_probe", False):
        if torch.device(cfg.get("device", "cpu")).type != "cuda":
            print("MINI_PODSMOKE_P23_SKIP cpu", flush=True)
        elif not Path(cfg.get("data_dir", "data/p2data")).joinpath("A.bin").exists():
            print("MINI_PODSMOKE_P23_SKIP p2data-missing", flush=True)
        else:
            print("MINI_PODSMOKE_P23_SKIP disabled-in-shared-core", flush=True)

    best = min((r.get("median_ms_step_5_20", float("inf")) for r in records if "error" not in r), default=float("inf"))
    hours = {f"{r.get('preset')}/{r.get('arm')}":
             (r.get("median_ms_step_5_20", 0) * 1200 / 1000.0 + r.get("eval_extrapolated_s", 0)) / 3600.0
              for r in records if "error" not in r}
    counts = {"core": 8 * len([p for p in presets if p in {"mini-s", "mini-m"}]),
              "mix": 4 if any(a.endswith("-mix") for a in arms) else 0}
    total_h = sum(hours.values())
    by_key = {(x["preset"], x["arm"], x["k"]): x for x in concurrency
              if all(rc == 0 for rc in x.get("returncodes", []))}
    baseline = {k[:2]: v["aggregate_tokens_s"] for k, v in by_key.items() if k[2] == 1}
    choices = []
    for k, v in by_key.items():
        if k[:2] in baseline and baseline[k[:2]] > 0:
            choices.append((v["aggregate_tokens_s"] / baseline[k[:2]], k[2]))
    best_concurrency = int(max(choices, default=(1., 1.))[1])
    matrix_h = 0.0
    matrix_parts = {}
    for preset in presets:
        if preset in {"mini-s", "mini-m"}:
            subtotal = 0.0
            for arm, count in (("e2e2", 3), ("e2e1-ce", 3), ("e2e1-rp", 1), ("e2e1-nc", 1)):
                subtotal += hours.get(f"{preset}/{arm}", 0.0) * count
            matrix_parts[f"core-{preset[-1]}"] = subtotal
            matrix_h += subtotal
    if any(a.endswith("-mix") for a in arms):
        mix_preset = cfg.get("mix_preset", "mini-s")
        mix_total = 0.0
        for arm in MIX_ARMS:
            mix_total += hours.get(f"{mix_preset}/{arm}", 0.0) * 2
        matrix_parts["mix"] = mix_total
        matrix_h += mix_total
    matrix_h /= max(1., max(choices, default=(1., 1.))[0])
    cost = {"gpu_hours": hours, "matrix_cells": counts, "total_gpu_hours": total_h,
            "usd_per_hour": float(cfg.get("usd_per_hour", .74)),
            "matrix_gpu_hours": matrix_h,
            "total_usd": matrix_h * float(cfg.get("usd_per_hour", .74)),
            "best_concurrency": best_concurrency,
            "speedup": max(choices, default=(1., 1.))[0],
            "matrix_totals": {"core-s": sum(hours.get(f"mini-s/{a}", 0.) * n for a, n in (("e2e2", 3), ("e2e1-ce", 3), ("e2e1-rp", 1), ("e2e1-nc", 1))),
                              "core-m": sum(hours.get(f"mini-m/{a}", 0.) * n for a, n in (("e2e2", 3), ("e2e1-ce", 3), ("e2e1-rp", 1), ("e2e1-nc", 1))),
                              "mix": matrix_parts.get("mix", 0.)},
            "matrix_totals_k1": matrix_parts,
            "matrix_totals_best": {k: v / max(1., max(choices, default=(1., 1.))[0]) for k, v in matrix_parts.items()},
            "concurrency": concurrency}
    print("MINI_PODSMOKE_COST " + json.dumps(cost, sort_keys=True), flush=True)
    payload = {"records": records, "micro_batch": micro, "head_benchmark": head, "cost": cost}
    (out / "podsmoke.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    return payload


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency-json")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-min-05-podsmoke"))
    ap.add_argument("config", nargs="?")
    args = ap.parse_args(argv)
    from .data import _set_process_name
    _set_process_name(args.process_name)
    if args.concurrency_json:
        _concurrency_cell(args.concurrency_json, args.process_name)
        return
    if not args.config:
        raise SystemExit("podsmoke config required")
    with open(args.config, "rb") as f:
        doc = tomllib.load(f)
    run(doc.get("p2", doc))


if __name__ == "__main__":
    main()
