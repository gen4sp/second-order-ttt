"""CUDA V1 validity read for the text echo templates.

The job performs one production-read parity check before touching injected CAL
documents, then reads all registered MINI checkpoints in sequence.  It emits
small ``v1_item.npz`` files and never writes a checkpoint.

Generator v2 (natural wrappers): the control sentences (no-barrier, no-binding,
query-entity) never write in these reads (``echo_text.write_mask`` is the inner
mask), and the value-mask ablation acts in chunk 0, where every mention and binding
lies (the layout validates this; the mask builder refuses anything else).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import _set_process_name, open_split
from .echo_text import make_injector, write_mask
from .meta import InnerCfg, stream_eval
from .train import TrainCfg, build, prepare_eval_setup
from .evaluate import eval_docs


def _cfg(root: dict, item: dict, out: Path) -> TrainCfg:
    preset = str(item.get("preset", "mini-m"))
    arm = str(item["arm"])
    seed = int(item.get("init_seed", item.get("seed", 0)))
    return TrainCfg.from_dict({
        "preset": preset, "arm": arm, "init_seed": seed, "data_seed": seed,
        "dtype": root.get("dtype", "bf16"), "device": root.get("device", "cuda"),
        "seq_len": int(root.get("T", 8192)), "global_batch": 16, "micro_batch": 1,
        "steps": 1200, "warmup": 120, "data_dir": root["data_dir"],
        "out_dir": str(out), "eval_split": "eval", "calib_split": "calib", "ho_split": "ho",
        "n_eval": max(256, int(root.get("parity_docs", 16))),
        "pilot_docs": max(256, int(root.get("n_docs", 256))), "ho_docs": 2,
        "min_probes": 1, "barriers": list(root.get("barriers", [2, 4, 6])),
        "segment_len": int(root.get("S", 2048)), "eval_batch": int(root.get("eval_batch", 2)),
        "fused_attn": False, "save_checkpoints": False, "process_name": root["process_name"],
    })


def _load(path: Path, cfg: TrainCfg):
    model, _ = build(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def _parity(item: dict, root: dict, cfg: TrainCfg, path: Path, excluded: np.ndarray) -> dict:
    """Recompute the logged production adapted probe NLL on the first 16 docs."""
    from .data import open_split
    if not (path.parent / "eval_step1200.npz").exists():
        raise RuntimeError(f"ABORT parity missing eval_step1200.npz: {path}")
    model, _ = _load(path, cfg)
    setup = prepare_eval_setup(cfg, model, excluded, write_meta=False)
    n = min(int(root.get("parity_docs", 16)), len(setup.eval_docs_arr))
    if n < 16:
        raise RuntimeError(f"ABORT parity needs >=16 eligible docs, got {n}")
    kw = dict(batch=cfg.eval_batch, inner_lr=cfg.inner_lr, inner_clip=cfg.inner_clip,
              device=cfg.device, fused_attn=False, barriers=cfg.barriers,
              r_ref=cfg.r_ref, data_seed=cfg.data_seed)
    measured = eval_docs(model, setup.eval_docs_arr[:n], setup.probes[:n], "ce",
                         modes=("adapt",), **kw)["doc"]["adapt"]
    with np.load(path.parent / "eval_step1200.npz", allow_pickle=False) as z:
        if "doc_adapt" not in z:
            raise RuntimeError("ABORT parity logged file lacks doc_adapt")
        logged = np.asarray(z["doc_adapt"][:n], dtype=np.float64)
    diff = np.abs(np.asarray(measured, dtype=np.float64) - logged)
    rec = {"cell": item.get("cell", f"{cfg.preset}/{cfg.arm}/s{cfg.init_seed}"),
           "n_docs": n, "mean_abs_diff": float(np.nanmean(diff)),
           "max_abs_diff": float(np.nanmax(diff)), "tolerance": .01,
           "pass": bool(np.isfinite(diff).all() and np.nanmean(diff) <= .01)}
    print("V1_PARITY " + json.dumps(rec, sort_keys=True), flush=True)
    if not rec["pass"]:
        raise RuntimeError("ABORT parity failed: " + json.dumps(rec, sort_keys=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def _mask_for_source(records: list[dict], T: int, C: int, device) -> torch.Tensor:
    """Math-attention key mask for c=0: hide the mention value tokens from every
    source value read.  All mentions and source reads must lie in chunk 0."""
    b = len(records)
    km = torch.ones((b, C, C), dtype=torch.bool, device=device)
    for i, rec in enumerate(records):
        mention = np.asarray(rec["mention_positions"], dtype=np.int64).reshape(-1)
        src = np.asarray(rec["source_nll_positions"], dtype=np.int64).reshape(-1)
        if mention.min() < 1 or src.min() < 1 or max(mention.max(), src.max()) >= C:
            raise ValueError("ABORT value mask: mentions and source reads must be in chunk 0")
        rows = torch.as_tensor(src, device=device)[:, None]
        cols = torch.as_tensor(mention, device=device)[None, :]
        km[i, rows, cols] = False
    return km


def _read_batch(model, rows: np.ndarray, records: list[dict], cfg: TrainCfg, *, masked: bool = False) -> dict:
    arr = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(cfg.device)
    ids, tgt = arr[:, :-1], arr[:, 1:]
    valid = tgt != model.cfg.bos_id
    masks = {0: _mask_for_source(records, rows.shape[1] - 1, model.cfg.chunk, arr.device)} if masked else None
    # Control sentences do not write: later reads see only binding/query/host writes.
    writes = torch.from_numpy(write_mask(records, ids.shape[1])).to(arr.device)
    with torch.no_grad():
        res = stream_eval(model, ids, tgt, valid,
                          InnerCfg(rule="ce", lr=cfg.inner_lr, clip=cfg.inner_clip,
                                   seed=cfg.data_seed), barriers=cfg.barriers,
                          inner_mask=writes, fused_attn=False, suffix_key_masks=masks,
                          return_ce_residual=not masked, return_greedy=True)
    return res


def _gather(x: torch.Tensor, positions: list, device) -> np.ndarray:
    p = torch.as_tensor(np.asarray(positions, dtype=np.int64) - 1, device=device)
    b, s, l = p.shape
    return torch.gather(x, 1, p.reshape(b, s * l)).reshape(b, s, l).detach().cpu().numpy()


def _item(item: dict, root: dict, excluded: np.ndarray, injector, started: float) -> dict:
    out = Path(item.get("out", Path(root["out_dir"]) / item["arm"] / f"s{item.get('init_seed', 0)}"))
    out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(root, item, out)
    path = Path(item["ckpt"])
    if not path.exists():
        raise FileNotFoundError(path)
    model, _ = _load(path, cfg)
    hosts, _ = open_split(root["data_dir"], "calib")
    n = min(int(root.get("n_docs", 256)), len(hosts))
    if n < 1:
        raise RuntimeError("no CAL hosts")
    rows, eval_meta = injector.eval_documents(np.asarray(hosts[:n, : cfg.seq_len + 1]),
                                              row_ids=np.arange(n, dtype=np.int64), p=float(root.get("p", .5)))
    records = eval_meta["rows"]
    bs = max(1, int(root.get("eval_batch", 2)))
    source, residual, masked_source, recall, query_nll = [], [], [], [], []
    no_barrier_nll, no_binding_nll, entity_nll = [], [], []
    no_barrier, no_binding, entity = [], [], []
    for b0 in range(0, n, bs):
        sl = slice(b0, min(n, b0 + bs))
        recs = records[sl]
        normal = _read_batch(model, rows[sl], recs, cfg, masked=False)
        masked = _read_batch(model, rows[sl], recs, cfg, masked=True)
        src = _gather(normal["nll"], [r["source_value_positions"] for r in recs], normal["nll"].device)
        res = _gather(normal["ce_residual"], [r["source_value_positions"] for r in recs], normal["nll"].device)
        msrc = _gather(masked["nll"], [r["source_value_positions"] for r in recs], masked["nll"].device)
        q = _gather(normal["nll"], [r["query_value_positions"] for r in recs], normal["nll"].device)
        q_nb = _gather(normal["nll"], [r["no_barrier_value_positions"] for r in recs], normal["nll"].device)
        q_nu = _gather(normal["nll"], [r["no_binding_value_positions"] for r in recs], normal["nll"].device)
        q_en = _gather(normal["nll"], [r["query_entity_value_positions"] for r in recs], normal["nll"].device)
        # All-query exact matches are represented by a per-slot all-token bool.
        g = _gather(normal["greedy"].to(torch.float32), [r["query_value_positions"] for r in recs], normal["nll"].device) > .5
        nb = _gather(normal["greedy"].to(torch.float32), [r["no_barrier_value_positions"] for r in recs], normal["nll"].device) > .5
        nu = _gather(normal["greedy"].to(torch.float32), [r["no_binding_value_positions"] for r in recs], normal["nll"].device) > .5
        en = _gather(normal["greedy"].to(torch.float32), [r["query_entity_value_positions"] for r in recs], normal["nll"].device) > .5
        source.append(src.mean(2)); residual.append(res.mean(2)); masked_source.append(msrc.mean(2)); query_nll.append(q.mean(2));
        no_barrier_nll.append(q_nb.mean(2)); no_binding_nll.append(q_nu.mean(2)); entity_nll.append(q_en.mean(2))
        recall.append(g.all(2)); no_barrier.append(nb.all(2)); no_binding.append(nu.all(2)); entity.append(en.all(2))
    arrays = {
        "source_nll": np.concatenate(source), "source_residual": np.concatenate(residual),
        "source_nll_masked": np.concatenate(masked_source), "query_nll": np.concatenate(query_nll),
        "no_barrier_nll": np.concatenate(no_barrier_nll), "no_binding_nll": np.concatenate(no_binding_nll),
        "query_entity_nll": np.concatenate(entity_nll),
        "query_exact": np.concatenate(recall), "no_barrier_exact": np.concatenate(no_barrier),
        "no_binding_exact": np.concatenate(no_binding), "query_entity_exact": np.concatenate(entity),
        "available": np.asarray([r["available"] for r in records], dtype=bool),
        "row_ids": np.arange(n, dtype=np.int64),
        "binding_paraphrase": np.asarray([r["paraphrase"]["binding"] for r in records], dtype=np.int64),
        "mention_paraphrase": np.asarray([r["paraphrase"]["mention"] for r in records], dtype=np.int64),
        "query_paraphrase": np.asarray([r["paraphrase"]["query"] for r in records], dtype=np.int64),
        "attempt": np.asarray([r["attempt"] for r in records], dtype=np.int64),
        "arm": np.asarray(cfg.arm), "seed": np.asarray(cfg.init_seed),
        "preset": np.asarray(cfg.preset),
        "checkpoint_sha256": np.asarray(__import__("hashlib").sha256(path.read_bytes()).hexdigest()),
    }
    np.savez_compressed(out / "v1_item.npz", **arrays)
    rec = {"cell": item.get("cell", f"{cfg.preset}/{cfg.arm}/s{cfg.init_seed}"),
           "arm": cfg.arm, "seed": cfg.init_seed, "n_docs": n,
           "wall_s": time.perf_counter() - started, "path": str(out / "v1_item.npz")}
    print("V1_READ " + json.dumps(rec, sort_keys=True), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def run(cfg: dict) -> dict:
    """Run parity then the registered six-checkpoint V1 read."""
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME", "TTM-sv1-01")
    _set_process_name(name)
    torch.set_num_threads(4)
    if torch.device(cfg.get("device", "cuda")).type != "cuda":
        raise ValueError("soq_v1 is registered for CUDA; use v1_analysis for CPU smoke")
    items = list(cfg.get("items", []))
    if not items:
        raise ValueError("soq_v1 requires p2.items")
    estimate = float(cfg.get("estimate_wall_s", 0.0))
    print("V1_ESTIMATE " + json.dumps({"items": len(items), "n_docs": cfg.get("n_docs", 256),
          "wall_s": estimate, "max_hours": cfg.get("max_hours", 1.5)}, sort_keys=True), flush=True)
    if estimate > 6 * 3600:
        print("V1_ABORT " + json.dumps({"reason": "estimated wall exceeds 6h", "wall_s": estimate}), flush=True)
        raise RuntimeError("ABORT V1 estimate exceeds 6h")
    data_dir = Path(cfg["data_dir"])
    excluded = np.load(data_dir / "excluded_vocab.npy", allow_pickle=False)
    # Pools are shared by all CAL documents and all six checkpoints.
    injector = make_injector(split="cal", f=1.0, p=float(cfg.get("p", .5)),
                             seed=int(cfg.get("echo_seed", 0)), T=int(cfg.get("T", 8192)),
                             S=int(cfg.get("S", 2048)), excluded_vocab=excluded)
    manifest = injector.manifest()
    print("V1_CONFIG " + json.dumps({"process_name": name, "manifest_sha256": manifest["manifest_sha256"],
          "tables_sha256": manifest["tables_sha256"], "generator_version": manifest["version"],
          "n_docs": cfg.get("n_docs", 256), "p": cfg.get("p", .5),
          "device": cfg.get("device", "cuda")}, sort_keys=True), flush=True)
    pin = str(cfg.get("echo_manifest_sha256", ""))
    if pin and pin != manifest["manifest_sha256"]:
        rec = {"reason": "echo_manifest", "registered": pin, "actual": manifest["manifest_sha256"]}
        print("V1_ABORT " + json.dumps(rec, sort_keys=True), flush=True)
        raise RuntimeError("ABORT echo injector manifest differs from the registered echo_manifest_sha256")
    first = items[0]
    first_cfg = _cfg(cfg, first, Path(cfg["out_dir"]) / "parity")
    parity = _parity(first, cfg, first_cfg, Path(first["ckpt"]), excluded)
    results = []
    started = time.perf_counter()
    root = Path(cfg["out_dir"]); root.mkdir(parents=True, exist_ok=True)
    (root / "echo_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for item in items:
        results.append(_item(item, cfg, excluded, injector, started))
    summary = {"parity": parity, "items": results, "wall_s": time.perf_counter() - started,
               "manifest_sha256": manifest["manifest_sha256"], "tables_sha256": manifest["tables_sha256"],
               "n_docs": int(cfg.get("n_docs", 256))}
    (root / "v1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("V1_DONE " + json.dumps({"items": len(results), "wall_s": summary["wall_s"]}, sort_keys=True), flush=True)
    return summary
