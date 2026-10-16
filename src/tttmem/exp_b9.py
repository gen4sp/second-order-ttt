"""Conditional pre-sealed B9 child; starts from scratch only after the FRS-B gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import exp_frsb as f
from .frsb_analysis import analyze as analyze_frsb, write_scores
from .b9_analysis import analyze, matching


def smoke_profile(payload, baseline):
    """Outcome-free pre-seal timing and dose feasibility readout, seed 0 only."""
    import statistics
    from .aux_b9 import B9_BASE
    by_parent = {(r["arm"], r["seed"]): r for r in baseline["cells"].values() if r.get("block") == "core"}
    timings = {}
    for r in payload["cells"].values():
        parent = by_parent.get((B9_BASE[r["arm"]], r["seed"]), {})
        ms, base_ms = r.get("ms_step"), parent.get("ms_step")
        timings[r["arm"]] = dict(ms_step=ms, base_ms_step=base_ms,
                                  ratio=ms/base_ms if isinstance(ms, (int, float)) and isinstance(base_ms, (int, float)) and base_ms > 0 else None,
                                  peak_cuda_mb=r.get("peak_cuda_mb"))
    rows = list(payload["cells"].values())
    match = matching([r for r in rows if r["arm"] == "e2e1_lin_b9"],
                     [r for r in rows if r["arm"] == "e2e1_lin_b9f"], steps=payload["manifest"]["config"]["steps"])
    ratios = [r["ratio"] for r in timings.values() if r["ratio"] is not None]
    return dict(timings=timings, median_step_ratio=statistics.median(ratios) if ratios else None,
                sham_dose_medians=match["medians"], smoke_only=True,
                note="pre-seal profile; CUDA production rates still require GPU measurement")


def parent_check(cfg):
    root = Path(__file__).resolve().parents[2]
    path = root / cfg["frsb_result_path"]
    if not path.exists():
        raise ValueError("B9 needs the completed FRS-B result.json")
    payload = json.loads(path.read_text())
    manifest = payload["manifest"]
    smoke = cfg.get("smoke", False)
    if bool(manifest["smoke"]) != smoke:
        raise ValueError("B9 and FRS-B must use the same phase")
    if manifest["seeds"] != cfg["seeds"]:
        raise ValueError("B9 and FRS-B seed cohorts differ")
    if cfg.get("require_seal", False):
        plan_hash = hashlib.sha256((root / cfg["frsb_plan_path"]).read_bytes()).hexdigest()
        if manifest["plan_sha256"] != plan_hash:
            raise ValueError("FRS-B registration checksum mismatch")
        if not smoke and manifest["source_sha256"] != cfg["frsb_source_sha256"]:
            raise ValueError("FRS-B implementation checksum mismatch")
        if not smoke and manifest["config_sha256"] != cfg["frsb_config_sha256"]:
            raise ValueError("FRS-B protocol checksum mismatch")
    # Re-score, rather than trusting an unauthenticated standalone gate file.
    scores = analyze_frsb(payload, smoke_only=smoke)
    return payload, scores["b9_gate"]


def run(cfg, seed=0):
    f.validate(cfg, b9=True)
    f.execution_guard(cfg, b9=True)
    baseline, gate = parent_check(cfg)
    if not cfg.get("smoke", False) and not gate["run_b9"]:
        result = dict(skipped=True, reason=gate["reason"], run_b9=False)
        print("B9_SKIPPED " + json.dumps(result), flush=True)
        return result
    parent_sha = hashlib.sha256((Path(__file__).resolve().parents[2] / cfg["frsb_result_path"]).read_bytes()).hexdigest()
    cfg = {**cfg, "_frsb_result_sha256": parent_sha}
    manifest, folder = f.prepare(cfg, b9=True)
    manifest["frsb_result_sha256"] = parent_sha
    f.atomic_json(folder/"manifest.json", manifest)
    out = dict(manifest=manifest, cells=f.execute(cfg, manifest, folder))
    if cfg.get("smoke", False):
        out["smoke_profile"] = smoke_profile(out, baseline)
        print("B9_SMOKE_PROFILE " + json.dumps(out["smoke_profile"]), flush=True)
    f.atomic_json(folder/"result.json", out)
    scores = analyze(out, baseline, smoke_only=cfg.get("smoke", False))
    write_scores(folder/"scores", scores, prefix="b9")
    result = dict(cells=len(out["cells"]), complete=sum(bool(r.get("trained")) for r in out["cells"].values()),
                  smoke=manifest["smoke"], output_dir=str(folder), plan_sha256=manifest["plan_sha256"],
                  parent_gate=gate)
    print("B9_SUMMARY " + json.dumps(result), flush=True)
    return result
