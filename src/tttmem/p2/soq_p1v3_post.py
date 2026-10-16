"""Post-hoc SOQ-P1 v3 echo reads on saved model weights.

This mode is evaluation-only.  It constructs the registered TEST templates and
paired TRAIN-pool templates, loads each checkpoint strictly, and writes all output
under ``post_out_dir``.  Training configuration keys (the same as in the SOQ-P1-v3x config) are accepted
but never passed to ``train``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import soq_p1v3 as api
from .model import build_model, load_state
from .train import ARMS, TrainCfg


def _sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _post_cfg(raw: dict) -> TrainCfg:
    """Make the normal model/eval config while dropping post-only keys."""
    ignored = {"mode", "post_checkpoints", "post_out_dir", "post_parent_npz"}
    return TrainCfg.from_dict({k: v for k, v in raw.items() if k not in ignored})


def _parent_npz(checkpoint: Path, step: int, raw: dict) -> Path | None:
    explicit = raw.get("post_parent_npz")
    candidates = ([Path(explicit)] if explicit else []) + [
        checkpoint.parent / f"eval_step{int(step):04d}.npz",
        checkpoint.parent.parent / f"eval_step{int(step):04d}.npz",
    ]
    return next((p for p in candidates if p.is_file()), None)


def _assert_parent_docs(es, checkpoint: Path, step: int, raw: dict) -> str | None:
    p = _parent_npz(checkpoint, step, raw)
    if p is None:
        return None
    with np.load(p, allow_pickle=False) as z:
        if "echo_docs_sha256" not in z:
            return None
        expected = str(np.asarray(z["echo_docs_sha256"]).item())
        # A two-document CPU smoke intentionally uses a prefix of the registered
        # 256-document set; its byte hash cannot equal the parent's full-set hash.
        parent_n = int(len(z["echo_available"])) if "echo_available" in z else len(es.available)
        if parent_n != len(es.available):
            return str(p) + f" (full-set hash retained; smoke docs={len(es.available)})"
    if expected != es.docs_sha256:
        raise ValueError(f"TEST docs_sha256 mismatch for step {step}: {es.docs_sha256} != {expected}")
    return str(p)


def _read_specs(set_name: str, es, fwd: str):
    if set_name == "test":
        if fwd in ("P", "P-A"):
            return es.docs_p, es.positions_p, ("source", "cross_paraphrase")
        return es.docs, es.positions, tuple(r for r in ("source", "cross_verbatim", "lookup_verbatim")
                                            if r in es.positions)
    # TRAIN eval-layout N/A/Z and exact training-layout T/T-A are cross-only.
    return es.docs, es.positions, ("source", "cross_verbatim")


def _run_forward(model, es, fwd: str, *, cfg: TrainCfg, device: str, batch: int,
                 set_name: str):
    docs, positions, reads = _read_specs(set_name, es, fwd)
    return api._read_forward(model, docs, positions, es.write_masks[fwd], reads, es,
                             rule=ARMS[cfg.arm][1], inner_lr=cfg.rule_lr(ARMS[cfg.arm][1]),
                             inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=device,
                             fused_attn=cfg.fused_attn, batch=batch, r_ref=cfg.r_ref,
                             data_seed=cfg.data_seed)


def _store_results(arrays: dict, es, results: dict, *, set_name: str, include_meta=True):
    if include_meta:
        arrays[f"echo_{set_name}_available"] = es.available
        arrays[f"echo_{set_name}_row_ids"] = es.row_ids
        arrays[f"echo_{set_name}_docs_sha256"] = np.asarray(es.docs_sha256)
        arrays[f"echo_{set_name}_docs_p_sha256"] = np.asarray(es.docs_p_sha256)
    for fwd, rr in results.items():
        for read in rr["em"]:
            p = f"echo_{set_name}_{fwd}_{read}"
            arrays[f"{p}_R"] = rr["em"][read]
            arrays[f"{p}_value_nll"] = rr["nll"][read]
            arrays[f"{p}_first_nll"] = rr["first_nll"][read]
            arrays[f"{p}_tokens23_nll"] = rr["tokens23_nll"][read]
            arrays[f"{p}_tokens_2_3_nll"] = rr["tokens23_nll"][read]
            arrays[f"{p}_greedy"] = rr["greedy_pos"][read]
            arrays[f"{p}_per_position_greedy"] = rr["greedy_pos"][read]
            arrays[f"{p}_cand_logp"] = rr["cand"][read]
            arrays[f"{p}_first_rank_16"] = rr["rank"][read].astype(np.int16)
            arrays[f"{p}_first_token_rank"] = rr["rank"][read].astype(np.int16)
            arrays[f"{p}_top1_16"] = rr["top1"][read]
            arrays[f"{p}_within_stratum_rank_8"] = rr["within_rank"][read].astype(np.int16)
            arrays[f"{p}_within_rank_8"] = rr["within_rank"][read].astype(np.int16)
            arrays[f"{p}_top1_8"] = rr["within_top1"][read]
            arrays[f"{p}_within_stratum_top1_8"] = rr["within_top1"][read]
            seg = (es.segment or {}).get(read)
            if seg is None:
                seg = np.ones_like(rr["em"][read], dtype=np.int8)
            arrays[f"{p}_segment"] = seg
            arrays[f"{p}_stratum"] = np.where(es.available, "copy_available", "copy_blocked")
            order = (es.read_order or {}).get(read)
            if order is None:
                order = np.broadcast_to(np.arange(es.available.shape[1], dtype=np.int16),
                                        es.available.shape).copy()
            arrays[f"{p}_read_order"] = order
            arrays[f"{p}_within_segment_read_order"] = order
            slot = (es.slot_index or {}).get(read)
            if slot is None:
                slot = np.broadcast_to(np.arange(es.available.shape[1], dtype=np.int16),
                                       es.available.shape).copy()
            arrays[f"{p}_binding_slot"] = slot
            arrays[f"{p}_binding_slot_index"] = slot


def _summary(es, results: dict) -> dict:
    """Compact means for each forward/read and registered stratum/segment/order slices."""
    out = {}
    for fwd, rr in results.items():
        for read in rr["em"]:
            n, slots = rr["em"][read].shape
            masks = {"pooled": np.ones((n, slots), dtype=bool),
                     "copy_available": np.broadcast_to(es.available, (n, slots)),
                     "copy_blocked": ~np.broadcast_to(es.available, (n, slots))}
            seg = (es.segment or {}).get(read)
            if seg is not None:
                for s in (2, 3, 4): masks[f"segment{s}"] = seg == s
            order = (es.read_order or {}).get(read)
            if order is not None: masks["order0"] = order == 0
            for label, mask in masks.items():
                def mean(x):
                    v = np.asarray(x)[mask]
                    return float(np.nanmean(v)) if v.size else None
                out[f"{fwd}/{read}/{label}"] = {
                    "R": mean(rr["em"][read]), "top1_16": mean(rr["top1"][read]),
                    "top1_8": mean(rr["within_top1"][read]),
                    "first_nll": mean(rr["first_nll"][read]),
                    "tokens23_nll": mean(rr["tokens23_nll"][read]),
                    "value_nll": mean(rr["nll"][read]),
                }
    return out


def _checkpoint_records(raw: dict):
    cps = raw.get("post_checkpoints", [])
    if not isinstance(cps, list) or not cps:
        raise ValueError("soq_p1v3_post requires a non-empty post_checkpoints list")
    out = []
    for c in cps:
        if not isinstance(c, dict) or "step" not in c or "path" not in c:
            raise ValueError("each post checkpoint needs {step, path}")
        out.append((int(c["step"]), Path(c["path"])))
    return out


def run(cfg_raw: dict) -> dict:
    """Run all post-hoc forwards; no training entry point is reachable here."""
    raw = dict(cfg_raw)
    cfg = _post_cfg(raw)
    out_dir = Path(raw.get("post_out_dir", "post_echo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(raw.get("device", cfg.device))
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[str(raw.get("dtype", cfg.dtype))]
    torch.set_num_threads(4)
    torch.manual_seed(cfg.init_seed)
    excl = np.load(Path(cfg.data_dir) / "excluded_vocab.npy", allow_pickle=False)
    test = api.build_echo_eval_set(cfg, excl, forward_masks=True)
    train_eval = api.build_echo_train_templates(cfg, excl, controls=True)
    train_layout = api.build_echo_train_templates(cfg, excl, controls=False)
    parent_checks = {}
    outputs = []
    for step, path in _checkpoint_records(raw):
        if not path.is_file():
            raise FileNotFoundError(path)
        model = build_model(cfg.preset, compute_dtype=dtype, device="cpu", seed=cfg.init_seed)
        if dtype == torch.float64: model = model.double()
        meta = load_state(model, str(path))
        model = model.to(device).eval()
        parent_checks[str(step)] = _assert_parent_docs(test, path, step, raw)
        results = {}
        for fwd in ("N", "A", "Z", "P", "P-A"):
            results[("test", fwd)] = _run_forward(model, test, fwd, cfg=cfg, device=device,
                                                   batch=int(raw.get("eval_batch", cfg.eval_batch)), set_name="test")
        for fwd in ("N", "A", "Z"):
            results[("train", fwd)] = _run_forward(model, train_eval, fwd, cfg=cfg, device=device,
                                                    batch=int(raw.get("eval_batch", cfg.eval_batch)), set_name="train")
        for fwd in ("T", "T-A"):
            results[("train", fwd)] = _run_forward(model, train_layout, fwd, cfg=cfg, device=device,
                                                    batch=int(raw.get("eval_batch", cfg.eval_batch)), set_name="train")
        checkpoint_sha = _sha256(path)
        arrays = {"echo_version": np.asarray(3), "checkpoint_step": np.asarray(step),
                  "checkpoint_path": np.asarray(str(path)), "checkpoint_sha256": np.asarray(checkpoint_sha),
                  "parent_npz": np.asarray(parent_checks[str(step)] or "")}
        summaries = {}
        # Keys are grouped by set in the saved npz while the summary keeps the same compact names.
        for idx, ((set_name, fwd), rr) in enumerate(results.items()):
            es_for = test if set_name == "test" else (train_layout if fwd in ("T", "T-A") else train_eval)
            _store_results(arrays, es_for, {fwd: rr}, set_name=set_name,
                           include_meta=not any(k.startswith(f"echo_{set_name}_available") for k in arrays))
            summaries.update({f"{set_name}/{k}": v for k, v in _summary(es_for, {fwd: rr}).items()})
        arrays["echo_train_layout_docs_sha256"] = np.asarray(train_layout.docs_sha256)
        out_path = out_dir / f"post_step{step:04d}.npz"
        np.savez_compressed(out_path, **arrays)
        rec = {"step": step, "path": str(path), "sha256": checkpoint_sha,
               "out": str(out_path), "summary": summaries}
        print("P2_ECHO3_POST " + json.dumps(rec, sort_keys=True, separators=(",", ":")), flush=True)
        outputs.append(rec)
        del model
    return {"post_out_dir": str(out_dir), "checkpoints": outputs,
            "test_docs_sha256": test.docs_sha256, "train_docs_sha256": train_eval.docs_sha256}


__all__ = ["run", "_summary"]
