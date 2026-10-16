"""CPU-safe MINI rehearsal helpers and lightweight profiling."""
from __future__ import annotations

import json
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from .data import BOS, SEQ, _set_process_name
from .model import PRESETS, build_model, param_count
from .train import ARMS, TrainCfg, prepare_eval_setup, train
from .meta import InnerCfg, stream_align

MINI_ARMS = ("e2e2", "e2e1-ce", "e2e1-rp", "e2e1-nc", "e2e2-mix", "e2e1-ce-mix")


def _write_tiny_data(root: Path, *, vocab: int = 97, rows: int = 16, seed: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, count in (("A", rows), ("code_train", rows), ("code_eval", 4),
                         ("eval", 4), ("ho", 2), ("calib", 2)):
        arr = np.empty((count, SEQ), dtype=np.uint32)
        arr[:, 0] = BOS if vocab > BOS else 1
        arr[:, 1:] = rng.integers(3, vocab, size=(count, SEQ - 1), dtype=np.uint32)
        # Plant repeated 4-grams in the probe region so the probe path is exercised.
        for i in range(count):
            arr[i, 3:7] = [11, 12, 13, 14]
            arr[i, 35:39] = [11, 12, 13, 14]
        arr.tofile(root / f"{split}.bin")
        with (root / f"{split}.jsonl").open("w") as f:
            for i in range(count):
                rec = {"row": i, "sha1": f"{split}-{i}",
                       "domain": f"{split}{i}.example"}
                if split == "code_eval":
                    rec["pool_row"] = i
                f.write(json.dumps(rec) + "\n")
    np.save(root / "excluded_vocab.npy", np.zeros(vocab, dtype=bool))
    meta = {"quotas": {"A": rows, "code_train": rows, "code_eval": 4,
                        "eval": 4, "ho": 2, "calib": 2}, "seq": SEQ}
    (root / "meta.json").write_text(json.dumps(meta) + "\n")


def mini_profile(cfg: dict) -> dict:
    _set_process_name(cfg.get("process_name") or os.environ.get("PROCESS_NAME", "TTM-min-30-profile"))
    out = {}
    dtype = getattr(torch, cfg.get("dtype", "float32")) if cfg.get("dtype") else torch.float32
    for preset in cfg.get("presets", ["mini-s", "mini-m"]):
        t0 = time.perf_counter()
        m = build_model(preset, compute_dtype=dtype)
        out[preset] = {"params": param_count(m), "construct_s": time.perf_counter() - t0}
        print("MINI_PROFILE " + json.dumps({"preset": preset, **out[preset]}), flush=True)
    return out


def mini_smoke(cfg: dict) -> dict:
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", "TTM-min-30-smoke")
    _set_process_name(name)
    root = Path(cfg.get("out_dir", "outputs/mini-smoke"))
    if root.exists():
        import shutil
        shutil.rmtree(root)
    data = Path(cfg.get("data_dir", root / "data"))
    _write_tiny_data(data, seed=int(cfg.get("data_seed", 0)))
    device = cfg.get("device", "cpu")
    steps = int(cfg.get("steps", 2))
    outputs = {}
    for arm in MINI_ARMS:
        arm_root = root / arm
        mixed = arm.endswith("-mix")
        tc = TrainCfg(arm=arm.removesuffix("-mix"), preset="tiny", dtype="fp32", device=device,
                      seq_len=64, global_batch=16 if mixed else 2,
                      micro_batch=4 if mixed else 1, steps=steps,
                      warmup=max(1, min(steps, 2)), lr_peak=1e-2, inner_lr=1.0,
                      data_dir=str(data), out_dir=str(arm_root), n_eval=4,
                      min_probes=1, eval_steps=[0, steps], full_eval_steps=[0, steps],
                      sizing_steps=[], trajectory_docs=4, ho_docs=2,
                      ho_trajectory_docs=2, pilot_docs=2, eval_batch=2,
                      pilot_steps=[], assay_gate=False, crossover_rules=[],
                      log_every=1, resume_every=steps, save_checkpoints=False,
                      barriers=[2], segment_len=16, fused_attn=False, mix=mixed,
                      code_train_split="code_train", code_eval_split="code_eval",
                      code_eval_steps=[steps],
                      process_name=f"TTM-min-30-{arm}")
        t0 = time.perf_counter()
        result = train(tc)
        result["ms_per_step"] = 1000 * result.get("wall_s", time.perf_counter() - t0) / steps
        result["peak_gb"] = 0.0
        outputs[arm] = result
        print("MINI_SMOKE_ARM " + json.dumps({"arm": arm, **result}), flush=True)
    from .mini_analysis import write
    analysis = write(root, root / "mini_analysis.json", root / "mini_analysis.md")
    summary = {"smoke": True, "device": device, "arms": outputs,
               "analysis": str(root / "mini_analysis.json")}
    (root / "smoke_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("MINI_SMOKE " + json.dumps(summary, allow_nan=True), flush=True)
    return summary


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


_IDENTITY_FIELDS = (
    "preset", "arm", "init_seed", "data_seed", "data_dir", "steps",
    "n_eval", "ho_docs", "eval_batch", "inner_lr", "inner_clip", "r_ref",
    "rule_lr_scale", "barriers", "segment_len", "fused_attn", "min_probes",
    "eval_split", "ho_split", "calib_split",
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same_value(left, right):
    left, right = _json_value(left), _json_value(right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return list(left) == list(right)
    if isinstance(left, dict) or isinstance(right, dict):
        return dict(left or {}) == dict(right or {})
    return left == right


def _cell_config_path(path: Path, item: dict, cfg: TrainCfg) -> Path | None:
    explicit = item.get("cell_config")
    if explicit:
        return Path(explicit)
    # matrix.py writes this beside the cell's ``runs`` directory.
    if len(path.parents) > 4:
        candidate = (path.parents[4] / ".cell-configs" /
                     f"{cfg.preset}-{cfg.arm}-s{int(cfg.init_seed)}.json")
        if candidate.exists():
            return candidate
    return None


def _checkpoint_skip(path: Path, item: dict, cfg: TrainCfg, *, setup=None,
                     payload=None, mini_e_only: bool = True):
    """Return a closed-world identity/input skip reason, or ``None``.

    Loading a present checkpoint is deliberately outside this closed-world list:
    corrupt pickles and strict state mismatches are evaluation errors.
    """
    if not path.exists():
        return "missing_checkpoint"
    expected = (item.get("ckpt_sha256") or item.get("checkpoint_sha256") or
                item.get("sha256"))
    actual = _sha256(path)
    if expected and str(expected) != actual:
        return f"checkpoint_hash_mismatch expected={expected} actual={actual}"
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    bare_state = isinstance(payload, dict) and payload and all(
        isinstance(v, torch.Tensor) for v in payload.values())
    if not isinstance(payload, dict) or ("model" not in payload and not bare_state):
        return "checkpoint_config_mismatch:missing_model_payload"
    expected_step = int(item.get("step", cfg.steps))
    if mini_e_only and cfg.preset in {"mini-s", "mini-m", "125m"} and cfg.arm != "e2e1-ce":
        return "checkpoint_config_mismatch:MINI-E requires arm=e2e1-ce"
    if not bare_state and payload.get("step") != expected_step:
        return f"checkpoint_config_mismatch:step={payload.get('step')}"
    if not bare_state and payload.get("arm") != cfg.arm:
        return f"checkpoint_config_mismatch:arm={payload.get('arm')}"
    if not bare_state and "init_label" not in payload:
        return "checkpoint_config_mismatch:missing_init_label"
    if not bare_state and payload.get("init_label") != cfg.init_label:
        return f"checkpoint_config_mismatch:init_label={payload.get('init_label')}"
    if not bare_state and payload.get("preset") not in (None, cfg.preset):
        return f"checkpoint_config_mismatch:preset={payload.get('preset')}"
    # A bare state dict has no provenance. It is safe only when the caller
    # supplied an item-level content hash and a matrix cell config is present.
    cell_config = _cell_config_path(path, item, cfg)
    if bare_state and (not expected or cell_config is None):
        return "checkpoint_config_mismatch:bare_state_without_provenance"
    if cfg.preset in {"mini-s", "mini-m", "125m"} and cell_config is None:
        return "missing_cell_config"
    if cell_config is not None:
        try:
            saved = json.loads(cell_config.read_text())
        except FileNotFoundError:
            return "missing_cell_config"
        except json.JSONDecodeError:
            return "checkpoint_config_mismatch:invalid_cell_config"
        if not isinstance(saved, dict):
            return "checkpoint_config_mismatch:invalid_cell_config"
        for key in _IDENTITY_FIELDS:
            expected_value = getattr(cfg, key)
            saved_value = saved.get(key, getattr(TrainCfg(), key))
            if not _same_value(saved_value, expected_value):
                return f"checkpoint_config_mismatch:{key}"
    if setup is not None:
        meta_path = path.parent / "eval_meta.npz"
        if not meta_path.exists():
            return "missing_eval_meta"
        try:
            with np.load(meta_path, allow_pickle=False) as meta:
                for key in ("eval_rows", "eval_sha1", "ho_sha1"):
                    if key not in meta or not np.array_equal(meta[key], setup.eval_meta[key]):
                        return f"eval_meta_mismatch:{key}"
        except FileNotFoundError:
            return "missing_eval_meta"
        step_path = path.parent / f"eval_step{expected_step:04d}.npz"
        if not step_path.exists():
            return "missing_eval_output"
        try:
            with np.load(step_path, allow_pickle=False) as logged:
                for key in ("eval_rows", "eval_sha1", "ho_sha1"):
                    if key not in logged or not np.array_equal(logged[key], setup.eval_meta[key]):
                        return f"eval_output_mismatch:{key}"
        except FileNotFoundError:
            return "missing_eval_output"
    if not bare_state:
        for key in ("init_seed", "data_seed"):
            if key in payload and int(payload[key]) != int(getattr(cfg, key)):
                return f"checkpoint_config_mismatch:{key}"
    return None


def _load_checkpoint(path: Path, cfg: TrainCfg, *, payload=None, model=None):
    from .train import build

    if model is None:
        model, _ = build(cfg)
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _save_ckpt_eval(out: Path, rule: str, r: dict, setup, cfg: TrainCfg,
                    n_docs: int, n_ho: int, extra: dict) -> Path:
    payload = {f"doc_{m}": v for m, v in r["doc"].items()}
    payload.update({f"tok_{m}": np.concatenate(v).astype(np.float32)
                    for m, v in r["tok"].items()
                    if v and all(x is not None for x in v)})
    payload.update({f"src_nll_{m}": v for m, v in r["src_nll"].items()})
    for key in ("src_tok", "src_cot"):
        payload.update({f"{key}_{m}": np.concatenate(v).astype(np.float32)
                        for m, v in r[key].items() if v})
    payload.update({
        "eval_rows": np.asarray(setup.elig[:n_docs]),
        "eval_sha1": np.asarray([setup.eval_man[i]["sha1"] for i in setup.elig[:n_docs]]),
        "eval_domain": np.asarray([setup.eval_man[i]["domain"] for i in setup.elig[:n_docs]]),
        "ho_sha1": np.asarray([m["sha1"] for m in setup.ho_man[:n_ho]]),
        "ho_domain": np.asarray([m["domain"] for m in setup.ho_man[:n_ho]]),
        "rule": np.asarray(rule), "preset": np.asarray(cfg.preset),
        "arm": np.asarray(cfg.arm), "init_seed": np.asarray(cfg.init_seed),
        "data_seed": np.asarray(cfg.data_seed), "init_label": np.asarray(cfg.init_label),
    })
    payload.update(extra)
    path = out / f"eval_ckpt_{rule}.npz"
    np.savez_compressed(path, **payload)
    return path


def _minie_item(item: dict, root_cfg: dict, repo: Path, root_name: str,
                ordinal: int) -> dict:
    from .data import repeat_classes
    from .evaluate import eval_docs
    from .matrix import _base_config
    from .train import build

    preset, arm = item["preset"], item["arm"]
    seed = int(item.get("init_seed", 0))
    data_seed = int(item.get("data_seed", seed))
    cell = {"preset": preset, "arm": arm, "init_seed": seed, "data_seed": data_seed}
    base = _base_config(root_cfg, cell, repo)
    for key in ("data_dir", "code_data_dir", "device"):
        if key in root_cfg:
            base[key] = root_cfg[key]
    out = Path(item.get("out", Path(root_cfg.get("out_dir", ".")) /
                               preset / arm / f"s{seed}"))
    base.update(out_dir=str(out), process_name=root_name,
                resume=False, save_checkpoints=False)
    tc = TrainCfg.from_dict(base)
    path = Path(item["ckpt"])
    cell_name = item.get("cell", f"{preset}/{arm}/s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    item_name = f"TTM-min-{int(root_cfg.get('process_nn', 60)):02d}-{preset}-{arm}-s{seed}"
    _set_process_name(item_name)
    model = None
    try:
        if not path.exists():
            rec = {"cell": cell_name, "reason": "missing_checkpoint", "ckpt": str(path)}
            print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
            return {**rec, "status": "skip"}
        excluded_path = Path(tc.data_dir) / "excluded_vocab.npy"
        if not excluded_path.exists():
            rec = {"cell": cell_name, "reason": "missing_input:excluded_vocab.npy",
                   "ckpt": str(path)}
            print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
            return {**rec, "status": "skip"}
        excluded = np.load(excluded_path, allow_pickle=False)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model, _ = build(tc)
        setup = prepare_eval_setup(tc, model, excluded, write_meta=False)
        reason = _checkpoint_skip(path, item, tc, setup=setup, payload=payload)
        if reason:
            rec = {"cell": cell_name, "reason": reason, "ckpt": str(path)}
            print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
            return {**rec, "status": "skip"}
        model = _load_checkpoint(path, tc, payload=payload, model=model)
        n_docs = min(len(setup.eval_docs_arr), int(tc.n_eval), 1024)
        n_ho = min(len(setup.ho_docs), int(tc.ho_docs), 256)
        docs = setup.eval_docs_arr[:n_docs]
        probes = setup.probes[:n_docs]
        eval_cls = repeat_classes(docs, tc.segment_len)
        ho_cls = repeat_classes(setup.ho_docs[:n_ho], tc.segment_len)
        rules = list(dict.fromkeys(item.get("rules", ["ce"])))
        results = []
        skipped_rules = []
        had_error = False
        meta_sha256 = _sha256(path.parent / "eval_meta.npz")
        checkpoint_sha256 = _sha256(path)
        for rule in rules:
            if rule not in ("ce", "rp", "nc"):
                rec = {"cell": cell_name, "rule": rule,
                       "reason": f"config_mismatch:unknown_rule:{rule}"}
                print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
                skipped_rules.append(rec)
                continue
            rule_t0 = time.perf_counter()
            try:
                if rule == "ce":
                    modes = list(dict.fromkeys(tc.eval_modes + tc.full_extra_modes))
                else:
                    modes = ["fP_focal", "fP_nonfocal"]
                kw = dict(batch=tc.eval_batch, inner_lr=tc.rule_lr(rule),
                          inner_clip=tc.inner_clip, device=tc.device,
                          fused_attn=tc.fused_attn, barriers=tc.barriers,
                          r_ref=tc.r_ref, data_seed=tc.data_seed)
                r = eval_docs(model, docs, probes, rule, modes=modes,
                              classes=eval_cls, **kw)
                rh = eval_docs(model, setup.ho_docs[:n_ho], [None] * n_ho, rule,
                               modes=["heldout"], classes=ho_cls, **kw)
                r["doc"]["heldout"] = rh["doc"]["heldout"]
                r["doc"]["heldout_noadapt"] = rh["doc"]["heldout_noadapt"]
                M = float(np.nanmean(r["doc"]["fP_focal"] - r["doc"]["fP_nonfocal"]))
                extra = {f"x_{rule}_M": r["doc"]["fP_focal"] - r["doc"]["fP_nonfocal"],
                         f"x_{rule}_heldout": r["doc"]["heldout"],
                         "n_docs": np.asarray(n_docs), "n_ho": np.asarray(n_ho),
                         "inner_lr": np.asarray(tc.rule_lr(rule)),
                         "r_ref": np.asarray(tc.r_ref),
                         "checkpoint_sha256": np.asarray(checkpoint_sha256),
                         "eval_meta_sha256": np.asarray(meta_sha256)}
                v0 = None
                if rule == "ce":
                    step = int(item.get("step", tc.steps))
                    logged = path.parent / f"eval_step{step:04d}.npz"
                    with np.load(logged, allow_pickle=False) as z:
                        if "x_ce_M" in z:
                            logged_m = float(np.nanmean(z["x_ce_M"]))
                        else:
                            logged_m = float(np.nanmean(z["doc_fP_focal"] - z["doc_fP_nonfocal"]))
                    v0 = {"cell": cell_name, "M_logged": logged_m,
                          "M_recomputed": M, "abs_diff": abs(logged_m - M),
                          "pass": bool(abs(logged_m - M) <= 1e-3)}
                    print("MINIE_V0 " + json.dumps(v0, sort_keys=True), flush=True)
                    extra["v0_pass"] = np.asarray(v0["pass"])
                    extra["v0_abs_diff"] = np.asarray(v0.get("abs_diff", np.nan))
                    extra["M_logged"] = np.asarray(v0["M_logged"])
                saved = _save_ckpt_eval(out, rule, r, setup, tc, n_docs, n_ho, extra)
                G = float(np.nanmean(r["doc"].get("noadapt", np.full(n_docs, np.nan)) -
                                    r["doc"].get("adapt", np.full(n_docs, np.nan))))
                read = {"cell": cell_name, "rule": rule, "status": "done", "M": M, "G": G,
                        "n_docs": n_docs, "wall_s": time.perf_counter() - rule_t0,
                        "path": str(saved)}
                print("MINIE_READ " + json.dumps(read, sort_keys=True), flush=True)
                results.append({**read, "v0": v0})
            except Exception as exc:
                had_error = True
                error = f"{type(exc).__name__}: {exc}"
                rec = {"cell": cell_name, "rule": rule, "error": error}
                print("MINIE_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
                results.append({"cell": cell_name, "rule": rule, "status": "error",
                                "error": error})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if had_error:
            return {"cell": cell_name, "status": "error", "results": results,
                    "skipped_rules": skipped_rules}
        if not results:
            reason = "config_mismatch:no_registered_rules"
            rec = {"cell": cell_name, "reason": reason, "ckpt": str(path)}
            print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
            return {**rec, "status": "skip", "results": [], "skipped_rules": skipped_rules}
        return {"cell": cell_name, "status": "done", "results": results,
                "skipped_rules": skipped_rules}
    except FileNotFoundError as exc:
        rec = {"cell": cell_name, "reason": f"missing_input:{Path(exc.filename or str(exc)).name}",
               "ckpt": str(path)}
        print("MINIE_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rec = {"cell": cell_name, "rule": "setup", "error": error}
        print("MINIE_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
        return {"cell": cell_name, "status": "error", "results": [rec]}
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mini_ckpt_eval(cfg: dict) -> dict:
    """Run the registered MINI-E checkpoint read without changing checkpoints."""
    nn = int(cfg.get("process_nn", 60))
    if not 60 <= nn <= 69:
        raise ValueError("MINI-E process_nn must be in 60..69")
    torch.set_num_threads(4)
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", f"TTM-min-{nn:02d}-minie")
    _set_process_name(name)
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    items = list(cfg.get("items", []))
    if not items:
        raise ValueError("mini_ckpt_eval requires p2.items")
    estimates = []
    for i, item in enumerate(items):
        preset = item.get("preset", "mini-s")
        default = 900.0 if preset == "mini-s" else 1800.0
        est = float(item.get("estimate_wall_s", cfg.get("estimate_wall_s", default)))
        estimates.append(est)
        print("MINIE_ESTIMATE " + json.dumps({"item": i, "cell": item.get("cell"),
              "preset": preset, "est_wall_s": est}, sort_keys=True), flush=True)
    print("MINIE_ESTIMATE_TOTAL " + json.dumps({"items": len(items),
          "wall_s": sum(estimates), "wall_h": sum(estimates) / 3600.0}, sort_keys=True), flush=True)
    started = time.perf_counter()
    results = []
    for i, item in enumerate(items):
        try:
            results.append(_minie_item(item, cfg, repo, name, i))
        except Exception as exc:  # keep the matrix fail-closed and continue items
            cell = item.get("cell", f"{item.get('preset', 'unknown')}/{item.get('arm', 'unknown')}/s{item.get('init_seed', 0)}")
            error = f"{type(exc).__name__}: {exc}"
            rec = {"cell": cell, "rule": "setup", "error": error}
            print("MINIE_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
            results.append({"cell": cell, "status": "error", "results": [rec]})
    summary_root = Path(cfg.get("out_dir") or
                        (items[0].get("out") if items and items[0].get("out") else "."))
    summary_root.mkdir(parents=True, exist_ok=True)
    errors = [r for r in results if r.get("status") == "error"]
    done = [r for r in results if r.get("status") == "done"]
    summary = {"items": results, "estimated_wall_s": sum(estimates),
               "wall_s": time.perf_counter() - started,
               "done": len(done), "skipped": sum(r.get("status") == "skip" for r in results),
               "errors": len(errors), "nothing_evaluated": not done}
    (summary_root / "minie_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n")
    if errors:
        raise RuntimeError(f"MINI-E had {len(errors)} item error(s); see minie_summary.json")
    if not done:
        raise RuntimeError("MINI-E nothing evaluated; see minie_summary.json")
    return {"items": results, "estimated_wall_s": sum(estimates),
            "out_dir": cfg.get("out_dir"), "summary": str(summary_root / "minie_summary.json")}


def _minia_item(item: dict, root_cfg: dict, repo: Path, root_name: str) -> dict:
    """Evaluate one checkpoint with write/read gradient projections."""
    from .data import repeat_classes
    from .evaluate import eval_docs
    from .matrix import _base_config
    from .train import build

    started = time.perf_counter()
    preset, arm = item["preset"], item["arm"]
    seed = int(item.get("init_seed", 0))
    data_seed = int(item.get("data_seed", seed))
    cell = {"preset": preset, "arm": arm, "init_seed": seed, "data_seed": data_seed}
    base = _base_config(root_cfg, cell, repo)
    for key in ("data_dir", "code_data_dir", "device"):
        if key in root_cfg:
            base[key] = root_cfg[key]
    out = Path(item.get("out", Path(root_cfg.get("out_dir", ".")) / preset / arm / f"s{seed}"))
    base.update(out_dir=str(out), process_name=root_name, resume=False, save_checkpoints=False)
    tc = TrainCfg.from_dict(base)
    path = Path(item["ckpt"])
    cell_name = item.get("cell", f"{preset}/{arm}/s{seed}")
    if not path.exists():
        rec = {"cell": cell_name, "reason": "missing_checkpoint", "ckpt": str(path)}
        print("MINIA_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    excluded_path = Path(tc.data_dir) / "excluded_vocab.npy"
    if not excluded_path.exists():
        rec = {"cell": cell_name, "reason": "missing_input:excluded_vocab.npy", "ckpt": str(path)}
        print("MINIA_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model, _ = build(tc)
    setup = prepare_eval_setup(tc, model, np.load(excluded_path, allow_pickle=False), write_meta=False)
    reason = _checkpoint_skip(path, item, tc, setup=setup, payload=payload, mini_e_only=False)
    if reason:
        rec = {"cell": cell_name, "reason": reason, "ckpt": str(path)}
        print("MINIA_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    model = _load_checkpoint(path, tc, payload=payload, model=model)
    out.mkdir(parents=True, exist_ok=True)
    n_docs = min(int(item.get("n_docs", 256)), len(setup.eval_docs_arr), int(tc.n_eval))
    docs, probes = setup.eval_docs_arr[:n_docs], setup.probes[:n_docs]
    eval_cls = repeat_classes(docs, tc.segment_len)
    # Reuse the production evaluation path for the two measured suppression values.
    kw = dict(batch=tc.eval_batch, inner_lr=tc.rule_lr("ce"), inner_clip=tc.inner_clip,
              device=tc.device, fused_attn=tc.fused_attn, barriers=tc.barriers,
              r_ref=tc.r_ref, data_seed=tc.data_seed)
    measured = eval_docs(model, docs, probes, "ce", modes=("fP_focal", "fP_nonfocal"),
                         classes=eval_cls, **kw)
    vf, vn, cf, cn, bf, bn, gr, scales = [], [], [], [], [], [], [], []
    batch_size = max(1, int(tc.eval_batch))
    for b0 in range(0, n_docs, batch_size):
        batch_docs = docs[b0:b0 + batch_size]
        batch_probes = probes[b0:b0 + batch_size]
        ids = torch.from_numpy(batch_docs[:, :-1].astype(np.int64)).to(tc.device)
        tgt = torch.from_numpy(batch_docs[:, 1:].astype(np.int64)).to(tc.device)
        valid = tgt != model.cfg.bos_id
        f = torch.from_numpy(np.stack([p.focal for p in batch_probes])).to(tc.device)
        nf = torch.from_numpy(np.stack([p.nonfocal for p in batch_probes])).to(tc.device)
        pt = torch.zeros_like(valid)
        for j, probe in enumerate(batch_probes):
            if len(probe.pos):
                pt[j, torch.from_numpy(probe.pos).to(tc.device)] = True
        a = stream_align(model, ids, tgt, valid, f, nf,
                         InnerCfg(rule="ce", lr=tc.rule_lr("ce"), clip=tc.inner_clip),
                         barriers=tc.barriers, fused_attn=tc.fused_attn,
                         probe_targets=pt)
        vf.extend(a["vf"].cpu().numpy()); vn.extend(a["vn"].cpu().numpy())
        cf.extend(a["cosf"].cpu().numpy()); cn.extend(a["cosn"].cpu().numpy())
        bf.extend(a["bf"].cpu().numpy()); bn.extend(a["bn"].cpu().numpy()); gr.extend(a["gr"].cpu().numpy())
        scales.extend(a["scales"].cpu().numpy().transpose(1, 0))
    vf, vn, cf, cn = map(lambda x: np.asarray(x, dtype=np.float32), (vf, vn, cf, cn))
    n_probes = np.asarray([len(p.pos) for p in probes[:n_docs]], dtype=np.int64)
    measured_m = np.asarray(measured["doc"]["fP_focal"] - measured["doc"]["fP_nonfocal"], np.float64)

    # MINI-A V0: the production evaluation at this checkpoint must reproduce
    # the logged per-document focal/non-focal difference before alignment data
    # are consumed.  Keep this diagnostic in the artifact and in the log.
    step = int(item.get("step", tc.steps))
    logged_path = path.parent / f"eval_step{step:04d}.npz"
    v0 = {"cell": cell_name, "path": str(logged_path), "pass": False,
          "abs_diff": np.nan, "reason": "missing_logged_doc_values"}
    if logged_path.exists():
        with np.load(logged_path, allow_pickle=False) as logged:
            if "doc_fP_focal" in logged and "doc_fP_nonfocal" in logged:
                logged_m = np.asarray(logged["doc_fP_focal"][:n_docs] -
                                      logged["doc_fP_nonfocal"][:n_docs], np.float64)
                if logged_m.shape == measured_m.shape:
                    diff = np.abs(logged_m - measured_m)
                    v0.update({"abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
                               "mean_abs_diff": float(np.nanmean(diff)) if diff.size else 0.0,
                               "pass": bool(np.allclose(logged_m, measured_m,
                                                         rtol=1e-5, atol=1e-6)),
                               "reason": ""})
    print("MINIA_V0 " + json.dumps(v0, sort_keys=True, allow_nan=True), flush=True)

    payload_out = {
        "normVF": vf, "normVN": vn, "normGr": np.asarray(gr, np.float32),
        "cosF": cf, "cosN": cn, "bF": np.asarray(bf, np.float32), "bN": np.asarray(bn, np.float32),
        "clip_scale": np.asarray(scales, np.float32),
        "fP_focal": np.asarray(measured["doc"]["fP_focal"], np.float32),
        "fP_nonfocal": np.asarray(measured["doc"]["fP_nonfocal"], np.float32),
        "M_doc": measured_m.astype(np.float32), "n_probes": n_probes,
        "eval_rows": np.asarray(setup.elig[:n_docs]), "eval_sha1": np.asarray([setup.eval_man[i]["sha1"] for i in setup.elig[:n_docs]]),
        "preset": np.asarray(preset), "arm": np.asarray(arm), "init_seed": np.asarray(seed),
        "data_seed": np.asarray(data_seed), "n_docs": np.asarray(n_docs),
        "v0_pass": np.asarray(v0["pass"]), "v0_abs_diff": np.asarray(v0["abs_diff"]),
    }
    np.savez_compressed(out / "align.npz", **payload_out)
    bdiff = np.asarray(bf) - np.asarray(bn)
    mdoc = payload_out["M_doc"]
    def rank(x):
        x = np.asarray(x, dtype=float)
        o = np.argsort(x, kind="stable"); r = np.empty(len(x), float)
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and x[o[j]] == x[o[i]]:
                j += 1
            r[o[i:j]] = (i + j - 1) / 2
            i = j
        return r
    per_probe = np.divide(bdiff, n_probes, out=np.full_like(bdiff, np.nan, dtype=float),
                          where=n_probes > 0)
    finite = np.isfinite(per_probe) & np.isfinite(mdoc)
    finite_raw = np.isfinite(bdiff) & np.isfinite(mdoc)
    spearman = (float(np.corrcoef(rank(per_probe[finite]), rank(mdoc[finite]))[0, 1])
                if finite.sum() > 1 and np.std(per_probe[finite]) and np.std(mdoc[finite]) else np.nan)
    spearman_raw = (float(np.corrcoef(rank(bdiff[finite_raw]), rank(mdoc[finite_raw]))[0, 1])
                    if finite_raw.sum() > 1 and np.std(bdiff[finite_raw]) and np.std(mdoc[finite_raw]) else np.nan)
    wmean_f = np.divide((cf * vf).sum(1), vf.sum(1), out=np.full(n_docs, np.nan), where=vf.sum(1) > 0)
    wmean_n = np.divide((cn * vn).sum(1), vn.sum(1), out=np.full(n_docs, np.nan), where=vn.sum(1) > 0)
    rec = {"cell": cell_name, "n_docs": n_docs,
           "mean_cosF": float(np.nanmean(wmean_f)), "mean_cosN": float(np.nanmean(wmean_n)),
           "mean_normVF": float(np.nanmean(vf.sum(1))),
           "mean_normVF_per_chunk": float(np.nanmean(vf)), "mean_normGr": float(np.nanmean(gr)),
           "mean_bF": float(np.nanmean(bf)), "mean_bN": float(np.nanmean(bn)),
           "spearman_bdiff_M": spearman, "spearman_bdiff_raw_M": spearman_raw,
           "v0_pass": bool(v0["pass"]), "v0_abs_diff": float(v0["abs_diff"]),
           "wall_s": time.perf_counter() - started}
    print("MINIA_READ " + json.dumps(rec, sort_keys=True), flush=True)
    return {**rec, "status": "done", "path": str(out / "align.npz")}


def mini_align(cfg: dict) -> dict:
    """Run registered MINI-A checkpoint alignment probes."""
    nn = int(cfg.get("process_nn", 80))
    if nn != 80:
        raise ValueError("MINI-A process_nn must be 80")
    torch.set_num_threads(4)
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", "TTM-min-80-minia")
    _set_process_name(name)
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    items = list(cfg.get("items", []))
    if not items:
        raise ValueError("mini_align requires p2.items")
    estimates = []
    for i, item in enumerate(items):
        est = float(item.get("estimate_wall_s", cfg.get("estimate_wall_s", 900.0)))
        estimates.append(est)
        print("MINIA_ESTIMATE " + json.dumps({"item": i, "cell": item.get("cell"), "wall_s": est}, sort_keys=True), flush=True)
    print("MINIA_ESTIMATE_TOTAL " + json.dumps({"items": len(items), "wall_s": sum(estimates), "wall_h": sum(estimates) / 3600}, sort_keys=True), flush=True)
    root = Path(cfg.get("out_dir") or (items[0].get("out", ".")))
    results = []
    started = time.perf_counter()
    for item in items:
        try:
            results.append(_minia_item(item, cfg, repo, name))
        except Exception as exc:
            cell = item.get("cell", f"{item.get('preset')}/{item.get('arm')}/s{item.get('init_seed', 0)}")
            rec = {"cell": cell, "error": f"{type(exc).__name__}: {exc}"}
            print("MINIA_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
            results.append({**rec, "status": "error"})
    root.mkdir(parents=True, exist_ok=True)
    summary = {"items": results, "estimated_wall_s": sum(estimates), "wall_s": time.perf_counter() - started,
               "done": sum(r.get("status") == "done" for r in results), "skipped": sum(r.get("status") == "skip" for r in results),
               "errors": sum(r.get("status") == "error" for r in results)}
    (root / "minia_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("MINIA_DONE " + json.dumps({k: summary[k] for k in ("done", "skipped", "errors")},
                                     sort_keys=True), flush=True)
    if summary["errors"]:
        raise RuntimeError(f"MINI-A had {summary['errors']} item error(s); see minia_summary.json")
    if summary["skipped"]:
        suffix = "; nothing evaluated" if not summary["done"] else ""
        raise RuntimeError(f"MINI-A skipped {summary['skipped']} item(s); see minia_summary.json{suffix}")
    if not summary["done"]:
        raise RuntimeError("MINI-A nothing evaluated; see minia_summary.json")
    return summary


# --------------------------------------------------------------------------- G0
G0_READS = {
    # name: (eta multiplier, token multiplier, calibration key of const_mult)
    "ce": (1.0, "none", None),
    "eta050": (0.5, "none", None),
    "eta075": (0.75, "none", None),
    "eta125": (1.25, "none", None),
    "eta150": (1.5, "none", None),
    "eta200": (2.0, "none", None),
    "eta250": (2.5, "none", None),
    "eta300": (3.0, "none", None),
    "eta400": (4.0, "none", None),
    # SOQ-FS: mini-s matched-LM frontier extension.
    "eta500": (5.0, "none", None),
    "eta600": (6.0, "none", None),
    "sig4": (1.0, "sigma", None),
    "sig4shuf": (1.0, "sigma_shuf", None),
    "const4": (1.0, "const", "const_mult"),
    # G0b: the inverse rule m = min(4, sigma_ref / sigma) and its controls.
    "isig4": (1.0, "sigma_inv", None),
    "isig4shuf": (1.0, "sigma_inv_shuf", None),
    "iconst4": (1.0, "const", "const_mult_inv"),
    # G0c: the inverse rule at larger eta (does it lift the M-L frontier beyond eta alone?).
    "isig4_eta150": (1.5, "sigma_inv", None),
    "isig4_eta200": (2.0, "sigma_inv", None),
    "isig4_eta250": (2.5, "sigma_inv", None),
    "isig4_eta300": (3.0, "sigma_inv", None),
    "isig4_eta400": (4.0, "sigma_inv", None),
}


def g0_read_spec(name: str):
    """(eta multiplier, token multiplier, const key) of a read: a registered name above, else
    the generic ``etaNNN`` spec (plain CE at eta x NNN/100: eta160 = x1.6, eta1600 = x16), so
    new grid points need no code edit.  ``None`` if the name is neither."""
    if name in G0_READS:
        return G0_READS[name]
    m = re.fullmatch(r"eta(\d{3,4})", name)
    if m and int(m.group(1)) > 0:
        return (int(m.group(1)) / 100.0, "none", None)
    return None


# Fields that define prepare_eval_setup's selection (one job reads several checkpoints of one
# selection; SOQ-N3B caches it across items).
_SETUP_FIELDS = ("preset", "data_dir", "code_data_dir", "eval_split", "ho_split", "calib_split",
                 "seq_len", "n_eval", "ho_docs", "pilot_docs", "barriers", "min_probes",
                 "init_label")
G0_OPTIONAL_MODES = ("frozenP", "fP_focal_refclip")


def _g0_parity(name: str, M_doc: np.ndarray, Ls: np.ndarray, L_tok: float, ref_path,
               rows: np.ndarray, n_ho: int, tol_M: float, tol_L: float) -> dict:
    """SOQ-N3B hard parity of a read against the stored G0 read of the same checkpoint:
    same documents, |ΔM| <= tol_M (pooled doc mean) and |ΔL| <= tol_L (pooled L_adapt)."""
    with np.load(ref_path, allow_pickle=False) as z:
        ref_M = np.asarray(z["M_doc"], np.float64)
        ref_L = float(z["L_adapt"])
        same = bool(len(ref_M) == len(M_doc) and int(z["n_ho"]) == int(n_ho)
                    and np.array_equal(np.asarray(z["eval_rows"]), np.asarray(rows)))
        ref_Ld = np.asarray(z["L_adapt_sum"], np.float64) / np.maximum(1.0, z["L_adapt_count"])
    Ld = Ls[:, 0] / np.maximum(1.0, Ls[:, 1])
    dM = abs(float(np.nanmean(M_doc)) - float(np.nanmean(ref_M))) if same else float("nan")
    dL = abs(L_tok - ref_L) if same else float("nan")
    rec = {"read": name, "ref": str(ref_path), "same_documents": same, "dM": dM, "dL": dL,
           "tol_M": tol_M, "tol_L": tol_L,
           "doc_dM_max": float(np.nanmax(np.abs(M_doc - ref_M))) if same else float("nan"),
           "doc_dL_max": float(np.max(np.abs(Ld - ref_Ld))) if same and len(Ld) == len(ref_Ld)
           else float("nan")}
    rec["pass"] = bool(same and dM <= tol_M and dL <= tol_L)
    return rec


def _g0_kappa(name: str, r: dict, ksets: list, n_boot: int, seed: int) -> tuple[dict, dict]:
    """κ summaries (centred logits and log-probs) and the arrays saved in the read npz."""
    from .probe import kappa_stats

    out, summ = {}, {}
    for tag, key in (("kz", "kappa_z"), ("klp", "kappa_lp")):
        sm, arr = kappa_stats(ksets, r[key]["fP_nonfocal"], r[key]["fP_focal"],
                              n_boot=n_boot, seed=seed)
        summ[tag] = sm
        for k, v in arr.items():
            v = np.asarray(v)
            if v.dtype == np.float64 and not k.startswith("doc_"):
                v = v.astype(np.float32)
            out[f"{tag}_{k}"] = v
        out[f"{tag}_summary"] = np.asarray(json.dumps(sm, sort_keys=True, allow_nan=True))
    return summ, out


def _g0_sigma_calibration(model, setup, tc: TrainCfg, cap: float, batch: int) -> dict:
    """sigma_ref = median final-norm sigma over the frozen-P write tokens (chunks 0-3) of the
    calibration split under the plain CE stream; const = mean of min(cap, sigma/sigma_ref) and
    const_inv = mean of min(cap, sigma_ref/sigma)."""
    from .meta import stream_eval
    from .probe import PROBE_START, boundary_valid

    docs = np.asarray(setup.calib_docs)
    C = model.cfg.chunk
    T = docs.shape[1] - 1
    bvalid = torch.from_numpy(boundary_valid(T, C, list(tc.barriers))).to(tc.device)
    sig = []
    for b0 in range(0, len(docs), batch):
        arr = torch.from_numpy(docs[b0:b0 + batch].astype(np.int64)).to(tc.device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        bmask = (tgt != model.cfg.bos_id) & bvalid
        im = torch.ones_like(bmask)
        im[:, PROBE_START * C:] = False
        r = stream_eval(model, ids, tgt, bmask, InnerCfg(rule="ce", lr=tc.rule_lr("ce"),
                        clip=tc.inner_clip), barriers=list(tc.barriers), inner_mask=im,
                        fused_attn=tc.fused_attn, return_sigma=True,
                        nll_positions=torch.zeros_like(bmask))
        sel = bmask.clone()
        sel[:, PROBE_START * C:] = False
        sig.append(r["sigma"][sel].double().cpu().numpy())
    s = np.concatenate(sig)
    ref = float(np.median(s))
    const = float(np.mean(np.minimum(cap, s / ref)))
    const_inv = float(np.mean(np.minimum(cap, ref / s)))
    return {"sigma_ref": ref, "const_mult": const, "const_mult_inv": const_inv,
            "n_tokens": int(s.size),
            "sigma_q": [float(np.quantile(s, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
            "frac_capped": float(np.mean(s / ref >= cap)),
            "frac_capped_inv": float(np.mean(ref / s >= cap))}


def _g0_sample(probe, C: int, k: int, seed: int, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Up to k focal and k count-matched non-focal source positions in the write chunks 0-3."""
    from .probe import PROBE_START

    rng = np.random.default_rng(int(seed))
    lim = PROBE_START * C
    out = []
    for m in (probe.focal, probe.nonfocal):
        idx = np.flatnonzero(np.asarray(m[:lim]) & valid[:lim])
        if len(idx) > k:
            idx = np.sort(rng.choice(idx, size=k, replace=False))
        out.append(idx)
    return out[0], out[1]


def _g0_ho_adapt(model, docs: np.ndarray, inner: InnerCfg, tc: TrainCfg, batch: int) -> np.ndarray:
    """Per-document NLL sum and token count with the rule writing everywhere (TTT LM loss),
    chunks >= 1; L_adapt is the pooled per-token mean."""
    from .meta import stream_eval
    from .probe import boundary_valid

    C = model.cfg.chunk
    T = docs.shape[1] - 1
    bvalid = torch.from_numpy(boundary_valid(T, C, list(tc.barriers))).to(tc.device)
    out = []
    for b0 in range(0, len(docs), batch):
        arr = torch.from_numpy(docs[b0:b0 + batch].astype(np.int64)).to(tc.device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        bmask = (tgt != model.cfg.bos_id) & bvalid
        inner.row_ids = torch.arange(10_000 + b0, 10_000 + b0 + ids.shape[0], device=ids.device)
        r = stream_eval(model, ids, tgt, bmask, inner, barriers=list(tc.barriers),
                        fused_attn=tc.fused_attn)
        sel = bmask.clone()
        sel[:, :C] = False
        bind = (r["clip_scale"] < 1).double().mean(0)  # SOQ-N3B: per-doc clip-binding fraction
        out.append(torch.stack([(r["nll"] * sel).sum(1).double(), sel.sum(1).double(), bind],
                               1).cpu().numpy())
    return np.concatenate(out)  # [n, 3]: per-doc NLL sum, token count, clip-binding fraction


def _g0_ho_noadapt(model, docs: np.ndarray, tc: TrainCfg, batch: int) -> np.ndarray:
    """Per-document no-adaptation LM NLL on the held-out documents.

    This uses the same production ``heldout`` path as ``eval_step1200.npz``;
    it is independent of the eval-time inner-learning-rate multiplier.
    """
    from .evaluate import eval_docs
    from .data import repeat_classes

    if len(docs) == 0:
        return np.empty((0,), dtype=np.float64)
    classes = repeat_classes(docs, tc.segment_len)
    r = eval_docs(model, docs, [None] * len(docs), "ce", modes=("heldout",),
                  classes=classes, batch=batch, inner_lr=tc.rule_lr("ce"),
                  inner_clip=tc.inner_clip, device=tc.device,
                  fused_attn=tc.fused_attn, barriers=tc.barriers,
                  r_ref=tc.r_ref, data_seed=tc.data_seed)
    return np.asarray(r["doc"]["heldout"], dtype=np.float64)


def _g0_split(model, setup, tc: TrainCfg, root_cfg: dict, out: Path, cell_name: str,
              meta: dict) -> dict:
    """Gain split on the first eval documents; fatal analytic-vs-autograd identity gate."""
    from .meta import stream_gain_split
    from .probe import boundary_valid

    C = model.cfg.chunk
    t0 = time.perf_counter()
    n_split = min(int(root_cfg.get("n_split_docs", 256)), len(setup.eval_docs_arr))
    k = int(root_cfg.get("split_k", 32))
    sb = max(1, int(root_cfg.get("split_batch", 4)))
    docs = setup.eval_docs_arr[:n_split]
    T = docs.shape[1] - 1
    bvalid = boundary_valid(T, C, list(tc.barriers))
    parts = []
    for b0 in range(0, n_split, sb):
        arr = torch.from_numpy(docs[b0:b0 + sb].astype(np.int64)).to(tc.device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        valid = (tgt != model.cfg.bos_id)
        vnp = valid.cpu().numpy() & bvalid[None]
        sf = torch.zeros_like(valid)
        sn = torch.zeros_like(valid)
        for j in range(ids.shape[0]):
            fi, ni = _g0_sample(setup.probes[b0 + j], C, k, setup.elig[b0 + j], vnp[j])
            sf[j, torch.from_numpy(fi).to(tc.device)] = True
            sn[j, torch.from_numpy(ni).to(tc.device)] = True
        r = stream_gain_split(model, ids, tgt, valid, sf, sn,
                              InnerCfg(rule="ce", lr=tc.rule_lr("ce"), clip=tc.inner_clip),
                              barriers=list(tc.barriers), fused_attn=tc.fused_attn)
        r["doc"] = r["row"] + b0
        parts.append(r)
    cat = {key: torch.cat([p[key] for p in parts]).numpy() for key in parts[0]}
    err = cat["ch_err"]
    foc = cat["focal"].astype(bool)
    finite = all(bool(np.isfinite(v).all()) for v in cat.values() if v.dtype.kind == "f")
    logs = {f: np.log(cat[f]) for f in ("e", "u", "gam", "proj", "s")}
    logs["R"] = np.log(cat["gnorm"]) - sum(logs.values())
    summ = {"cell": cell_name, "n_tok": int(len(err)), "n_focal": int(foc.sum()),
            "finite": finite,
            "ch_err_median": float(np.median(err)) if len(err) else float("nan"),
            "ch_err_p99": float(np.quantile(err, 0.99)) if len(err) else float("nan"),
            "mean_log_g_focal": float(np.mean(np.log(cat["gnorm"][foc]))) if foc.any() else float("nan"),
            "mean_log_g_nonfocal": float(np.mean(np.log(cat["gnorm"][~foc]))) if (~foc).any() else float("nan"),
            "mean_log_focal": {f: float(np.mean(v[foc])) for f, v in logs.items()} if foc.any() else {},
            "wall_s": time.perf_counter() - t0}
    print("G0_SPLIT " + json.dumps(summ, sort_keys=True, allow_nan=True), flush=True)
    gate_med = float(root_cfg.get("ch_err_median_max", 0.02))
    gate_p99 = float(root_cfg.get("ch_err_p99_max", 0.10))
    ok = (len(err) > 0 and finite and summ["ch_err_median"] <= gate_med
          and summ["ch_err_p99"] <= gate_p99)
    # A failed split keeps its data under a name the analysis never reads.
    np.savez_compressed(out / ("gain_split.npz" if ok else "gain_split_FAILED.npz"), **cat,
                        **{key: np.asarray(v) for key, v in meta.items()},
                        n_docs=np.asarray(n_split), split_k=np.asarray(k),
                        eval_rows=np.asarray(setup.elig[:n_split]))
    if not ok:
        raise RuntimeError(f"G0 split identity gate failed: {summ}")
    return summ


def _g0_item(item: dict, root_cfg: dict, repo: Path, root_name: str, cache: dict | None = None) -> dict:
    """One checkpoint: the registered eval-time reads, then the optional gain split.

    SOQ-N3B keys (all optional; absent = the G0/G0b/G0c behaviour): ``overrides`` (cell keys on
    top of the production config, e.g. the FO-CM ``steps``), ``kappa_reads`` ("all" or read
    names: κ gathered inside the frozen-P forwards), ``read_modes`` ({read: ["frozenP",
    "fP_focal_refclip"]}), ``g0_ref`` ({read: stored G0 read npz}: hard parity) and ``hlm``
    (the no-adapt H_LM read of the training-layout held-out documents)."""
    from .data import repeat_classes
    from .evaluate import eval_docs
    from .matrix import _base_config
    from .probe import kappa_set
    from .train import build

    started = time.perf_counter()
    preset, arm = item["preset"], item["arm"]
    seed = int(item.get("init_seed", 0))
    data_seed = int(item.get("data_seed", seed))
    cell = {"preset": preset, "arm": arm, "init_seed": seed, "data_seed": data_seed}
    if item.get("overrides"):
        cell["overrides"] = dict(item["overrides"])
    base = _base_config(root_cfg, cell, repo)
    for key in ("data_dir", "code_data_dir", "device"):
        if key in root_cfg:
            base[key] = root_cfg[key]
    out = Path(item.get("out", Path(root_cfg.get("out_dir", ".")) / preset / arm / f"s{seed}"))
    base.update(out_dir=str(out), process_name=root_name, resume=False, save_checkpoints=False)
    tc = TrainCfg.from_dict(base)
    path = Path(item["ckpt"])
    cell_name = item.get("cell", f"{preset}/{arm}/s{seed}")
    if not path.exists():
        rec = {"cell": cell_name, "reason": "missing_checkpoint", "ckpt": str(path)}
        print("G0_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    excluded_path = Path(tc.data_dir) / "excluded_vocab.npy"
    if not excluded_path.exists():
        rec = {"cell": cell_name, "reason": "missing_input:excluded_vocab.npy", "ckpt": str(path)}
        print("G0_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model, _ = build(tc)
    skey = json.dumps({k: _json_value(getattr(tc, k)) for k in _SETUP_FIELDS}, sort_keys=True)
    if cache is not None and skey in cache:
        setup = cache[skey]
    else:
        setup = prepare_eval_setup(tc, model, np.load(excluded_path, allow_pickle=False),
                                   write_meta=False)
        if cache is not None:
            cache.clear()  # one selection at a time bounds the host memory
            cache[skey] = setup
    reason = _checkpoint_skip(path, item, tc, setup=setup, payload=payload, mini_e_only=False)
    if reason:
        rec = {"cell": cell_name, "reason": reason, "ckpt": str(path)}
        print("G0_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return {**rec, "status": "skip"}
    model = _load_checkpoint(path, tc, payload=payload, model=model)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_sha = _sha256(path)
    batch = max(1, int(root_cfg.get("read_batch", tc.eval_batch)))
    C = model.cfg.chunk
    result = {"cell": cell_name, "ckpt_sha256": ckpt_sha, "reads": []}

    reads = list(item.get("reads", root_cfg.get("reads", [])))
    read_cache = root_cfg.get("read_cache", {})
    if read_cache and read_cache.get("checkpoint_sha256") != ckpt_sha:
        raise ValueError("read cache checkpoint changed")
    verified_reads = read_cache.get("npz_sha256", {})
    unknown = [r for r in reads if g0_read_spec(r) is None]
    if unknown:
        raise ValueError(f"unknown G0 reads {unknown}")
    if reads and reads[0] != "ce":
        raise ValueError("G0 reads must start with the V0 'ce' read")
    kreads = item.get("kappa_reads", root_cfg.get("kappa_reads", []))
    kreads = set(reads) if kreads == "all" else set(kreads)
    read_modes = {**root_cfg.get("read_modes", {}), **item.get("read_modes", {})}
    refs = dict(item.get("g0_ref", {}))
    for label, names in (("kappa_reads", kreads), ("read_modes", read_modes), ("g0_ref", refs)):
        if set(names) - set(reads):
            raise ValueError(f"{label} names reads not in this item: {sorted(set(names) - set(reads))}")
    bad_modes = {m for v in read_modes.values() for m in v} - set(G0_OPTIONAL_MODES)
    if bad_modes:
        raise ValueError(f"read_modes allows only {G0_OPTIONAL_MODES}, got {sorted(bad_modes)}")
    for name, ref in refs.items():
        if not Path(ref).exists():
            raise FileNotFoundError(f"g0_ref for {name}: {ref}")
    tol_M = float(root_cfg.get("parity_tol_M", 1e-3))
    tol_L = float(root_cfg.get("parity_tol_L", 2e-4))
    cap = float(root_cfg.get("sigma_cap", 4.0))
    calib = None
    if any(g0_read_spec(r)[1] != "none" for r in reads):
        calib = _g0_sigma_calibration(model, setup, tc, cap, batch)
        calib["cell"] = cell_name
        print("G0_CALIB " + json.dumps(calib, sort_keys=True), flush=True)
        (out / "g0_calib.json").write_text(json.dumps(calib, indent=2, sort_keys=True) + "\n")
        result["calib"] = calib
    n_docs = min(int(item.get("n_docs", root_cfg.get("n_docs", 512))), len(setup.eval_docs_arr))
    docs, probes = setup.eval_docs_arr[:n_docs], setup.probes[:n_docs]
    eval_cls = repeat_classes(docs, tc.segment_len)
    n_ho = min(int(root_cfg.get("n_ho_adapt", 128)), len(setup.ho_docs))
    if root_cfg.get("read_cache_path") and (
            n_docs != int(root_cfg["n_docs"]) or n_ho != int(root_cfg["n_ho_adapt"])
            or len(setup.ho_docs) != tc.ho_docs):
        raise RuntimeError("S14 reads cannot cover the registered document counts")
    ho = np.asarray(setup.ho_docs[:n_ho])
    fs = bool(root_cfg.get("fs_protocol", False))
    if fs:  # SOQ-FS: the no-adapt held-out loss, read once per checkpoint
        L_noadapt_doc = _g0_ho_noadapt(model, ho, tc, batch)
        L_noadapt = float(np.nanmean(L_noadapt_doc)) if len(L_noadapt_doc) else float("nan")
    ksets = None
    if kreads:
        kkey = ("kappa", skey, n_docs, int(root_cfg.get("kappa_foils", 8)))
        if cache is not None and kkey in cache:
            ksets = cache[kkey]
        else:
            ksets = [kappa_set(d[:-1], d[1:], p, C, valid=d[1:] != model.cfg.bos_id,
                               barriers=tc.barriers, n_foils=int(root_cfg.get("kappa_foils", 8)),
                               seed=int(row))
                     for d, p, row in zip(docs, probes, setup.elig[:n_docs])]
            if cache is not None:
                cache[kkey] = ksets
    for name in reads:
        filename = f"g0_read_{name}.npz"
        if filename in verified_reads:
            if _sha256(out / filename) != verified_reads[filename]:
                raise ValueError(f"verified read changed: {filename}")
            result["reads"].append({"read": name, "status": "verified_skip"})
            continue
        t0 = time.perf_counter()
        eta, mult, const_key = g0_read_spec(name)
        extra = {"tok_mult": mult, "sigma_cap": cap}
        if calib is not None:
            extra.update(sigma_ref=calib["sigma_ref"],
                         const_mult=calib[const_key or "const_mult"])
        lr = tc.rule_lr("ce") * eta
        extra_modes = tuple(read_modes.get(name, ()))
        r = eval_docs(model, docs, probes, "ce", modes=("fP_focal", "fP_nonfocal") + extra_modes,
                      classes=eval_cls, batch=batch, inner_lr=lr, inner_clip=tc.inner_clip,
                      device=tc.device, fused_attn=tc.fused_attn, barriers=tc.barriers,
                      r_ref=tc.r_ref, data_seed=tc.data_seed, inner_extra=extra,
                      kappa=ksets if name in kreads else None)
        M_doc = np.asarray(r["doc"]["fP_focal"] - r["doc"]["fP_nonfocal"], np.float64)
        v0 = None
        if name == "ce" and root_cfg.get("endpoint_parity_read", True):
            step = int(item.get("step", tc.steps))
            with np.load(path.parent / f"eval_step{step:04d}.npz", allow_pickle=False) as z:
                logged = np.asarray(z["doc_fP_focal"][:n_docs] - z["doc_fP_nonfocal"][:n_docs],
                                    np.float64)
            diff = np.abs(logged - M_doc)
            tolerance = float(root_cfg.get("endpoint_parity_tol", 1e-5))
            v0 = {"cell": cell_name, "abs_diff_max": float(np.nanmax(diff)),
                  "pass": bool(logged.shape == M_doc.shape and
                               np.isfinite(diff).all() and
                               (np.max(diff) <= tolerance if root_cfg.get("endpoint_parity") else
                                np.allclose(logged, M_doc, rtol=1e-5, atol=1e-6))),
                  "atol": tolerance, "rtol": 0 if root_cfg.get("endpoint_parity") else 1e-5}
            if fs:
                # SOQ-FS gate: mean absolute difference <= 0.01 nats/document for M and
                # for the no-adapt held-out loss.
                m_existing_pass = bool(logged.shape == M_doc.shape and
                                       np.allclose(logged, M_doc, rtol=1e-5, atol=1e-6))
                m_mean_abs = float(np.nanmean(diff)) if diff.size else float("nan")
                l_mean_abs = float("nan")
                l_logged = path.parent / f"eval_step{step:04d}.npz"
                with np.load(l_logged, allow_pickle=False) as z:
                    if "doc_heldout" in z and len(z["doc_heldout"]) >= n_ho:
                        ldiff = np.abs(np.asarray(z["doc_heldout"][:n_ho], np.float64) - L_noadapt_doc)
                        l_mean_abs = float(np.nanmean(ldiff)) if ldiff.size else float("nan")
                fs_pass = bool(m_mean_abs <= 0.01 and np.isfinite(l_mean_abs) and l_mean_abs <= 0.01)
                v0 = {"cell": cell_name, "abs_diff_max": float(np.nanmax(diff)),
                      "m_mean_abs": m_mean_abs, "l_mean_abs": l_mean_abs,
                      "m_existing_allclose_pass": m_existing_pass,
                      "criterion": "FS mean abs <= 0.01 nats/document for M and no-adapt L",
                      "pass": fs_pass}
            print("G0_V0 " + json.dumps(v0, sort_keys=True), flush=True)
            if not v0["pass"]:
                raise RuntimeError(f"G0 V0 failed: {v0}")
        inner = InnerCfg(rule="ce", lr=lr, clip=tc.inner_clip, seed=tc.data_seed, **extra)
        Ls = _g0_ho_adapt(model, ho, inner, tc, batch)
        L_tok = float(Ls[:, 0].sum() / max(1.0, Ls[:, 1].sum()))
        st = r.get("rule_stats", {})
        bad = [k for k, v in (("M_doc", M_doc), ("fP_focal", r["doc"]["fP_focal"]),
                              ("fP_nonfocal", r["doc"]["fP_nonfocal"]), ("L_adapt", Ls[:, :2]))
               if not np.isfinite(np.asarray(v, np.float64)).all()]
        n3b = {m: np.asarray(r["doc"][m], np.float64) for m in extra_modes}
        n3b.update({f"clip_bind_{m}": v for m, v in r.get("clip_bind_doc", {}).items()})
        n3b["L_clip_bind_doc"] = Ls[:, 2]
        ksumm = None
        if name in kreads:
            ksumm, karr = _g0_kappa(name, r, ksets, int(root_cfg.get("kappa_boot", 10000)),
                                    int(root_cfg.get("kappa_boot_seed", 0)))
            n3b.update(karr)
            n3b.update(kappa_foils=np.asarray(int(root_cfg.get("kappa_foils", 8))))
        if fs:
            n3b.update(L_noadapt_doc=L_noadapt_doc, L_noadapt=np.asarray(L_noadapt))
        np.savez_compressed(out / f"g0_read_{name}.npz", **n3b, M_doc=M_doc,
                            fP_focal=np.asarray(r["doc"]["fP_focal"], np.float64),
                            fP_nonfocal=np.asarray(r["doc"]["fP_nonfocal"], np.float64),
                            L_adapt_doc=Ls[:, 0] / np.maximum(1.0, Ls[:, 1]),
                            L_adapt_sum=Ls[:, 0], L_adapt_count=Ls[:, 1],
                            L_adapt=np.asarray(L_tok), eta=np.asarray(eta), lr=np.asarray(lr),
                            tok_mult=np.asarray(mult), n_docs=np.asarray(n_docs),
                            n_ho=np.asarray(n_ho), ckpt_sha256=np.asarray(ckpt_sha),
                            stats=np.asarray(json.dumps(st, sort_keys=True)),
                            eval_rows=np.asarray(setup.elig[:n_docs]))
        rec = {"cell": cell_name, "read": name, "eta": eta, "tok_mult": mult,
               "M": float(np.nanmean(M_doc)), "L_adapt": L_tok,
               "stats": st, "n_docs": n_docs, "n_ho": n_ho,
               "wall_s": time.perf_counter() - t0}
        if fs:
            rec["L_noadapt"] = L_noadapt
        if extra_modes:
            rec["extra_modes"] = {m: float(np.nanmean(v)) for m, v in n3b.items() if m in extra_modes}
        print("G0_READ " + json.dumps(rec, sort_keys=True, allow_nan=True), flush=True)
        if ksumm is not None:
            rec["kappa"] = ksumm
            print("G0_KAPPA " + json.dumps({"cell": cell_name, "read": name, **ksumm},
                                           sort_keys=True, allow_nan=True), flush=True)
        if bad:
            raise RuntimeError(f"G0 read {name} has non-finite {bad}")
        if name in refs:
            par = _g0_parity(name, M_doc, Ls, L_tok, refs[name], setup.elig[:n_docs], n_ho,
                             tol_M, tol_L)
            par["cell"] = cell_name
            rec["parity"] = par
            print("G0_PARITY " + json.dumps(par, sort_keys=True, allow_nan=True), flush=True)
            if not par["pass"]:
                raise RuntimeError(f"G0 parity failed: {par}")
        result["reads"].append({**rec, "v0": v0})
        if root_cfg.get("read_cache_path"):
            from .s14 import record_read
            record_read(root_cfg["read_cache_path"], read_cache, filename)
    if bool(item.get("hlm", root_cfg.get("hlm", False))):
        if "g0_hlm.npz" in verified_reads:
            if _sha256(out / "g0_hlm.npz") != verified_reads["g0_hlm.npz"]:
                raise ValueError("verified H_LM changed")
            result["hlm"] = {"status": "verified_skip"}
        else:
            result["hlm"] = _g0_hlm(model, setup, tc, item, path, out, cell_name, ckpt_sha, batch)
            if root_cfg.get("read_cache_path"):
                from .s14 import record_read
                record_read(root_cfg["read_cache_path"], read_cache, "g0_hlm.npz")
    # The registered reads come first; the descriptive split cannot cost them.
    if bool(item.get("split", False)):
        result["split"] = _g0_split(model, setup, tc, root_cfg, out, cell_name,
                                    {"preset": preset, "arm": arm, "init_seed": seed,
                                     "ckpt_sha256": ckpt_sha})
    return {**result, "status": "done", "wall_s": time.perf_counter() - started}


def _g0_hlm(model, setup, tc: TrainCfg, item: dict, path: Path, out: Path, cell_name: str,
            ckpt_sha: str, batch: int) -> dict:
    """SOQ-N3B H_LM: the no-adaptation loss of the training-layout held-out documents (the
    ``heldout`` mode of the in-training eval, η-independent), per-document sums and counts;
    a non-fatal parity against the logged ``doc_heldout_noadapt`` of the checkpoint step."""
    from .data import repeat_classes
    from .evaluate import eval_docs

    t0 = time.perf_counter()
    rule = ARMS[tc.arm][1]
    hd = np.asarray(setup.ho_docs)
    n = len(hd)
    rh = eval_docs(model, hd, [None] * n, rule, modes=["heldout"],
                   classes=repeat_classes(hd, tc.segment_len), batch=batch,
                   inner_lr=tc.rule_lr(rule), inner_clip=tc.inner_clip, device=tc.device,
                   fused_attn=tc.fused_attn, barriers=tc.barriers, r_ref=tc.r_ref,
                   data_seed=tc.data_seed)
    hs, hc = rh["heldout_sum"], rh["heldout_count"]
    doc = np.asarray(rh["doc"]["heldout_noadapt"], np.float64)
    step = int(item.get("step", tc.steps))
    logged_path = path.parent / f"eval_step{step:04d}.npz"
    diff = None
    if logged_path.exists():
        with np.load(logged_path, allow_pickle=False) as z:
            if "doc_heldout_noadapt" in z.files and len(z["doc_heldout_noadapt"]) == n:
                diff = float(np.nanmax(np.abs(z["doc_heldout_noadapt"] - doc)))
    rec = {"cell": cell_name, "n_docs": n, "H_LM_tok": float(hs.sum() / max(1.0, hc.sum())),
           "H_LM_doc": float(np.nanmean(doc)), "logged_abs_diff_max": diff,
           "wall_s": time.perf_counter() - t0}
    np.savez_compressed(out / "g0_hlm.npz", heldout_sum=hs, heldout_count=hc, heldout_doc=doc,
                        H_LM_tok=np.asarray(rec["H_LM_tok"]), ho_sha1=setup.eval_meta["ho_sha1"],
                        ckpt_sha256=np.asarray(ckpt_sha), step=np.asarray(step))
    print("G0_HLM " + json.dumps(rec, sort_keys=True, allow_nan=True), flush=True)
    if not (np.isfinite(hs).all() and np.isfinite(hc).all()):
        raise RuntimeError(f"G0 H_LM has non-finite values: {rec}")
    return rec


def mini_g0(cfg: dict) -> dict:
    """Run the registered G0 reads (gain split + eval-time gain titration)."""
    nn = int(cfg.get("process_nn", 1))
    if not (1 <= nn <= 9 or 90 <= nn <= 99):
        raise ValueError("G0 process_nn must be in 1..9 (jobs) or 90..99 (smokes)")
    torch.set_num_threads(4)
    fs = bool(cfg.get("fs_protocol", False))
    if fs and str(cfg.get("device", "cpu")) == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(float(cfg.get("mps_memory_fraction", 0.25)))
        else:
            cfg["device"] = "cpu"
            print("G0_DEVICE_FALLBACK " + json.dumps({"requested": "mps", "device": "cpu"}), flush=True)
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", f"TTM-g0-{nn:02d}")
    if fs:
        os.environ["PROCESS_NAME"] = name
    _set_process_name(name)
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    items = list(cfg.get("items", []))
    if not items:
        raise ValueError("mini_g0 requires p2.items")
    print("G0_ESTIMATE_TOTAL " + json.dumps({"items": len(items), "wall_s": sum(
        float(i.get("estimate_wall_s", cfg.get("estimate_wall_s", 600))) for i in items)},
        sort_keys=True), flush=True)
    root = Path(cfg.get("out_dir") or (items[0].get("out", ".")))
    deadline = time.time() + 3600.0 * float(cfg.get("max_hours", 3.0))
    results = []
    started = time.perf_counter()
    cache = {} if cfg.get("setup_cache", False) else None
    for item_index, item in enumerate(items):
        cell = item.get("cell", f"{item.get('preset')}/{item.get('arm')}/s{item.get('init_seed', 0)}")
        if time.time() > deadline:
            rec = {"cell": cell, "reason": "max_hours"}
            print("G0_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
            results.append({**rec, "status": "skip"})
            continue
        try:
            results.append(_g0_item(item, cfg, repo, name, cache))
        except Exception as exc:
            rec = {"cell": cell, "error": f"{type(exc).__name__}: {exc}"}
            print("G0_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
            results.append({**rec, "status": "error"})
            # SOQ-FS: the first read is the V0 parity gate; a failed gate invalidates
            # every later read, so stop immediately.
            if fs and (item_index == 0 or "G0 V0 failed" in str(exc)):
                print("ABORT " + json.dumps({"reason": "V0 parity failure", **rec},
                                             sort_keys=True), flush=True)
                raise RuntimeError(f"ABORT: V0 parity failure (error in {cell}): {exc}") from exc
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    root.mkdir(parents=True, exist_ok=True)
    summary = {"items": results, "wall_s": time.perf_counter() - started,
               "done": sum(r.get("status") == "done" for r in results),
               "skipped": sum(r.get("status") == "skip" for r in results),
               "errors": sum(r.get("status") == "error" for r in results)}
    (root / "g0_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("G0_DONE " + json.dumps({k: summary[k] for k in ("done", "skipped", "errors")},
                                  sort_keys=True), flush=True)
    if summary["errors"]:
        raise RuntimeError(f"G0 had {summary['errors']} item error(s); see g0_summary.json")
    if summary["skipped"]:
        raise RuntimeError(f"G0 skipped {summary['skipped']} item(s); see g0_summary.json")
    return summary


def _minil_item(item: dict, cfg: dict, repo: Path, name: str) -> dict:
    from .matrix import _base_config, _resume_dir, _stale_cell
    from .train import build, training_order
    from .data import open_split

    preset, init_arm, arm = item["preset"], item["init_arm"], item["arm"]
    seed = int(item["init_seed"])
    data_seed = int(item["data_seed"])
    label = ("L-REF" if init_arm == "e2e2" and arm == "e2e2"
             else "L-SO" if arm == "e2e2" else "L-FO")
    cell = item.get("cell", f"{preset}/{label}/s{seed}")
    out = Path(item["out"])
    path = Path(item["ckpt"])
    if out.parts[-2:] != (label, f"s{seed}"):
        raise ValueError(f"MINI-L output path does not match cell label: {out}")
    identity = dict(preset=preset, arm=init_arm, init_seed=seed, data_seed=data_seed)
    if not path.exists():
        rec = dict(cell=cell, status="skip", reason="missing_checkpoint", ckpt=str(path))
        print("MINIL_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return rec

    # MINI's production checkpoints carry no provenance fields.  Build the
    # source-cell config and eval setup from the matrix config, then let the
    # shared checker validate the checkpoint payload plus the source eval metadata.
    source_base = _base_config(cfg, identity, repo)
    for key in ("data_dir", "device"):
        if key in cfg:
            source_base[key] = cfg[key]
    source_base.update(out_dir=str(path.parent), process_name=name,
                       init_seed=seed, data_seed=data_seed)
    source_cfg = TrainCfg.from_dict(source_base)
    excluded_path = Path(source_cfg.data_dir) / "excluded_vocab.npy"
    if not excluded_path.exists():
        rec = dict(cell=cell, status="skip", reason="missing_input:excluded_vocab.npy",
                   ckpt=str(path))
        print("MINIL_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return rec
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source_model, _ = build(source_cfg)
    source_setup = prepare_eval_setup(source_cfg, source_model, np.load(excluded_path, allow_pickle=False),
                                      write_meta=False)
    source_item = dict(item, preset=preset, arm=init_arm, init_seed=seed,
                       data_seed=data_seed, step=1200)
    reason = _checkpoint_skip(path, source_item, source_cfg, setup=source_setup,
                              payload=payload, mini_e_only=False)
    if reason:
        rec = dict(cell=cell, status="skip", reason=reason, ckpt=str(path))
        print("MINIL_SKIP " + json.dumps(rec, sort_keys=True), flush=True)
        return rec
    del payload, source_model, source_setup
    base = _base_config(cfg, {**identity, "arm": arm}, repo)
    for key in ("data_dir", "device"):
        if key in cfg:
            base[key] = cfg[key]
    base.update(init_seed=seed, data_seed=data_seed, init=str(path), steps=300,
                warmup=30, lr_peak=1e-3, lr_floor=1e-5, inner_warmup=1,
                data_start_step=1200, eval_steps=[150, 300], full_eval_steps=[300],
                n_eval=1024, trajectory_docs=256, ho_docs=256, eval_at_init=False,
                save_checkpoints=True, resume_every=75, out_dir=str(out), process_name=name)
    root = Path(cfg.get("out_dir", out.parents[2]))
    state_root = _resume_dir(out, item.get("resume_dir", ""), root, cfg.get("resume_root"))
    base.update(resume_dir=str(state_root), resume=any(state_root.glob("resume_step*.pt")))
    marker = out / "MINIL_CELL_DONE.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        if done.get("train") != {k: v for k, v in base.items() if k not in ("resume", "process_name")} \
                or done.get("init_sha256") != _sha256(path):
            raise ValueError("completed MINI-L config or init checkpoint changed")
        print("MINIL_CELL_DONE " + json.dumps(done, sort_keys=True), flush=True)
        return done
    stale = _stale_cell(out, state_root)
    if stale:
        print("MINIL_STALE " + json.dumps(dict(cell=cell, path=str(stale))), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(base, sort_keys=True) + "\n")
    t0 = time.perf_counter()
    result = train(TrainCfg.from_dict(base))
    if result.get("aborted") or result.get("steps") != 300:
        raise RuntimeError("MINI-L continuation did not complete")
    metrics = {}
    n_docs_300 = None
    for step in (150, 300):
        with np.load(out / f"eval_step{step:04d}.npz", allow_pickle=False) as z:
            metrics[f"M_{step}"] = float(np.nanmean(z["doc_fP_focal"] - z["doc_fP_nonfocal"]))
            n_docs = len(np.asarray(z["doc_fP_focal"]))
            if step == 300:
                metrics["G_300"] = float(np.nanmean(z["doc_noadapt"] - z["doc_adapt"]))
                n_docs_300 = n_docs
    if not all(np.isfinite(x) for x in metrics.values()):
        raise ValueError("MINI-L endpoint contains nonfinite metrics")
    train_arr, _ = open_split(base["data_dir"], base.get("train_split", "A"))
    order = training_order(len(train_arr), steps=300, batch=int(base["global_batch"]),
                           data_seed=data_seed, data_start_step=1200)
    done = dict(cell=cell, init_arm=init_arm, arm=arm, status="done", **metrics,
                n_docs=n_docs_300,
                order_sha1=hashlib.sha1(order.tobytes()).hexdigest(),
                wall_s=time.perf_counter() - t0, init_sha256=_sha256(path),
                train={k: v for k, v in base.items() if k not in ("resume", "process_name")})
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(done, sort_keys=True) + "\n")
    tmp.replace(marker)
    print("MINIL_CELL_DONE " + json.dumps(done, sort_keys=True), flush=True)
    return done


def mini_late(cfg: dict) -> dict:
    """MINI-L: sequential, fresh-optimizer SO/FO continuations on one GPU."""
    nn = int(cfg.get("process_nn", 81))
    if not 81 <= nn <= 83:
        raise ValueError("MINI-L process_nn must be in 81..83")
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", f"TTM-min-{nn:02d}-minil")
    _set_process_name(name)
    torch.set_num_threads(4)
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    items = list(cfg.get("items", []))
    if not items:
        raise ValueError("mini_late requires p2.items")
    estimates = []
    for i, item in enumerate(items):
        default = (2.4 if item["arm"] == "e2e1-ce" else 7.2) * 300 + 450
        est = float(item.get("estimate_wall_s", default))
        estimates.append(est)
        print("MINIL_ESTIMATE " + json.dumps(dict(item=i, out=item["out"], est_wall_s=est)), flush=True)
    started = time.perf_counter()
    results = []
    for item, estimate in zip(items, estimates):
        if time.perf_counter() - started + estimate > float(cfg.get("max_hours", 2.5)) * 3600:
            rec = dict(cell=item["out"], status="deferred", reason="max_hours")
            print("MINIL_DEFER " + json.dumps(rec), flush=True)
            results.append(rec)
            continue
        try:
            results.append(_minil_item(item, cfg, repo, name))
        except Exception as exc:
            rec = dict(cell=item["out"], status="error", error=f"{type(exc).__name__}: {exc}")
            print("MINIL_ERROR " + json.dumps(rec, sort_keys=True), flush=True)
            results.append(rec)
        finally:
            _set_process_name(name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    root = Path(cfg.get("out_dir", Path(items[0]["out"]).parents[2]))
    root.mkdir(parents=True, exist_ok=True)
    summary = dict(items=results, estimated_wall_s=sum(estimates), wall_s=time.perf_counter() - started,
                   done=sum(r["status"] == "done" for r in results),
                   skipped=sum(r["status"] == "skip" for r in results),
                   errors=sum(r["status"] == "error" for r in results),
                   deferred=sum(r["status"] == "deferred" for r in results))
    path = root / f"minil_summary_g{nn}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MINIL_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["skipped"] or summary["errors"] or summary["deferred"]:
        raise RuntimeError(f"MINI-L incomplete; see {path}")
    return summary
