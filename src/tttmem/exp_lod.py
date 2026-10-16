"""Sealed streaming-load protocol. All model and inner-rule arithmetic is imported."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from . import exp_ttt as t

MAIN = (16, 32, 64, 128)
ARMS = dict(lin="e2e1emb_lin", so="e2e2emb", aice="e2e1emb_aice",
            fo="e2e1emb", bridge="e2e1emb_lin", linkv="e2e1emb_linkv")
ROOT = Path(__file__).resolve().parents[2]
_EMIT_LOCK = threading.RLock()
PROCESS_START_SECONDS = 10.  # Conservative per-cell spawn/CUDA setup projection.
ENDPOINT_NAMES = ("A", "A_freeze", "A_snap", "A_last", "A_coupled", "cross_lin", "cross_ce", "W0")


@dataclasses.dataclass(frozen=True)
class LoadVocab:
    pool: int = 256
    n_values: int = 16
    n_fill: int = 16
    write: int = 0
    query: int = 1

    @property
    def V(self):
        return 2 + self.pool + self.n_values + self.n_fill

    @property
    def value0(self):
        return 2 + self.pool

    @property
    def fill0(self):
        return self.value0 + self.n_values


class LoadMeta(TypedDict):
    write_pos: np.ndarray  # B,N
    write_key: np.ndarray
    write_val: np.ndarray
    query_pos: np.ndarray  # B,P
    query_tok: np.ndarray
    query_write_idx: np.ndarray
    query_write_pos: np.ndarray
    probe_write_chunk: np.ndarray
    value_count: np.ndarray
    value_histogram: np.ndarray  # B,16
    snapshot_chunk: int
    chunk: int
    gap: int
    layout: str
    value0: int


WRITE_FIELDS = ("write_pos", "write_key", "write_val")
QUERY_FIELDS = ("query_pos", "query_tok", "query_write_idx", "query_write_pos",
                "probe_write_chunk", "value_count")
SEQ_FIELDS = ("value_histogram",)
ARRAY_FIELDS = WRITE_FIELDS + QUERY_FIELDS + SEQ_FIELDS
SCALAR_FIELDS = ("snapshot_chunk", "chunk", "gap", "layout", "value0")


def slice_meta_load(meta, lo, hi):
    return {k: v[lo:hi] if k in ARRAY_FIELDS else v for k, v in meta.items()}


def concat_meta_load(metas):
    if not metas:
        raise ValueError("empty metadata")
    for m in metas:
        if any(m[k] != metas[0][k] for k in SCALAR_FIELDS):
            raise ValueError("incompatible metadata")
    return {k: np.concatenate([m[k] for m in metas], axis=0) if k in ARRAY_FIELDS
            else v for k, v in metas[0].items()}


def load_length(n_pairs, chunk=64, gap=268, n_probe=16, layout="main", control_load=128,
                shift_chunks=0):
    slots = control_load if layout in ("L16", "coupled") else n_pairs
    return chunk * math.ceil((4 * slots + shift_chunks * chunk + gap + 7 * n_probe - 3) / chunk)


def echo_load_batch(rng, n_seq, T=None, *, n_pairs, n_probe=16, chunk=64,
                    events_per_chunk=16, layout="main", vocab=None, gap=268,
                    control_load=128, shift_chunks=0, given=None):
    """Vectorized W v k v writes; Q k v probes, each preceded by 1–4 fillers.

    `given=(tokens,meta)` rebuilds the exact same history for H1i. RNG for
    removal fillers is separate and never changes surviving writes or probes.
    """
    vocab = vocab or LoadVocab()
    if layout not in ("main", "last", "L16", "coupled"):
        raise ValueError("unknown layout")
    if events_per_chunk != chunk // 4 or chunk % 4:
        raise ValueError("each write chunk must contain chunk//4 events")
    if not 1 <= n_probe <= n_pairs <= vocab.pool or n_pairs % events_per_chunk:
        raise ValueError("invalid load or probe count")
    if layout == "coupled":
        if given is None:
            raise ValueError("coupled requires a given full history")
        raw, old = given
        keep = old["write_pos"][0] // chunk == old["snapshot_chunk"] - 1
        if keep.sum() != events_per_chunk or np.any(old["query_write_idx"] >= len(keep)):
            raise ValueError("invalid coupled source")
        # This diagnostic is defined only for last-chunk probes.
        if not np.all(keep[old["query_write_idx"]]):
            raise ValueError("coupled history requires last-chunk probes")
        tokens = raw.copy()
        drop = old["write_pos"][:, ~keep]
        rows = np.arange(len(raw))[:, None, None]
        event_positions = drop[..., None] + np.arange(-2, 2)
        tokens[rows, event_positions] = rng.integers(vocab.fill0, vocab.V, event_positions.shape)
        meta = dict(old)
        for k in WRITE_FIELDS:
            meta[k] = old[k][:, keep].copy()
        meta["query_write_idx"] = old["query_write_idx"] - np.flatnonzero(keep)[0]
        meta["layout"] = "coupled"
        hist = (meta["write_val"][..., None] == np.arange(vocab.value0, vocab.fill0)).sum(1)
        meta["value_histogram"] = hist
        meta["value_count"] = np.take_along_axis(hist, meta["query_tok"] - vocab.value0, 1)
        return tokens, meta
    slots = control_load if layout == "L16" else n_pairs
    if layout == "L16" and n_pairs != events_per_chunk:
        raise ValueError("L16 must have exactly one active write chunk")
    needed = load_length(n_pairs, chunk, gap, n_probe, layout, control_load, shift_chunks)
    T = needed if T is None else T
    if T < needed or T % chunk:
        raise ValueError("sequence length violates load contract")
    tokens = rng.integers(vocab.fill0, vocab.V, (n_seq, T + 1))
    keys = rng.permuted(np.broadcast_to(np.arange(2, 2 + vocab.pool), (n_seq, vocab.pool)), axis=1)[:, :n_pairs]
    values = rng.integers(vocab.value0, vocab.fill0, (n_seq, n_pairs))
    start = shift_chunks * chunk + (4 * (slots - n_pairs) if layout == "L16" else 0)
    wp = np.broadcast_to(start + 4 * np.arange(n_pairs) + 2, (n_seq, n_pairs)).copy()
    rows = np.arange(n_seq)[:, None]
    tokens[rows, wp - 2] = vocab.write
    tokens[rows, wp - 1] = values
    tokens[rows, wp] = keys
    tokens[rows, wp + 1] = values
    candidates = np.arange(n_pairs - events_per_chunk, n_pairs) if layout == "last" else np.arange(n_pairs)
    if len(candidates) < n_probe:
        raise ValueError("last chunk has too few probes")
    qidx = rng.permuted(np.broadcast_to(candidates, (n_seq, len(candidates))), axis=1)[:, :n_probe]
    spaces = rng.integers(1, 5, (n_seq, n_probe))
    qstart = shift_chunks * chunk + 4 * slots + gap + np.cumsum(spaces + 3, 1) - 3
    qp = qstart + 1
    qt = np.take_along_axis(values, qidx, 1)
    tokens[rows, qp - 1] = vocab.query
    tokens[rows, qp] = np.take_along_axis(keys, qidx, 1)
    tokens[rows, qp + 1] = qt
    hist = (values[..., None] == np.arange(vocab.value0, vocab.fill0)).sum(1)
    qwp = np.take_along_axis(wp, qidx, 1)
    meta = LoadMeta(write_pos=wp, write_key=keys, write_val=values, query_pos=qp,
                    query_tok=qt, query_write_idx=qidx, query_write_pos=qwp,
                    probe_write_chunk=qwp // chunk,
                    value_count=np.take_along_axis(hist, qt - vocab.value0, 1),
                    value_histogram=hist, snapshot_chunk=int(wp.max() // chunk + 1),
                    chunk=chunk, gap=gap, layout=layout, value0=vocab.value0)
    return tokens, meta


def layout_audit(tokens, meta, task):
    wp, qp, qi = (meta[k] for k in ("write_pos", "query_pos", "query_write_idx"))
    rows = np.arange(len(tokens))[:, None]
    unique = np.all(np.diff(np.sort(qi, 1), axis=1) > 0)
    checks = dict(length=tokens.shape[1] == task.T + 1,
                  distinct=bool(unique), subset=bool(((qi >= 0) & (qi < wp.shape[1])).all()),
                  mapping=np.array_equal(np.take_along_axis(wp, qi, 1), meta["query_write_pos"]),
                  keys=np.array_equal(tokens[rows, qp], np.take_along_axis(meta["write_key"], qi, 1)),
                  targets=np.array_equal(tokens[rows, qp + 1], meta["query_tok"]),
                  writes=np.array_equal(tokens[rows, wp], meta["write_key"])
                         and np.array_equal(tokens[rows, wp - 2], np.zeros_like(wp))
                         and np.array_equal(tokens[rows, wp - 1], meta["write_val"])
                         and np.array_equal(tokens[rows, wp + 1], meta["write_val"]),
                  gap=bool((qp.min(1) - (wp.max(1) + 1) > task.receptive_field).all()),
                  snapshot=meta["snapshot_chunk"] == int(wp.max() // task.chunk + 1),
                  histogram=bool((meta["value_histogram"].sum(1) == wp.shape[1]).all()),
                  counts=np.array_equal(meta["value_count"], np.take_along_axis(
                      meta["value_histogram"], meta["query_tok"] - meta["value0"], 1)))
    _, counts = np.unique(wp[0] // task.chunk, return_counts=True)
    checks["density"] = bool((counts == task.events_per_chunk).all())
    return {k: bool(v) for k, v in checks.items()}


def binding_free_baseline(meta):
    hist = meta["value_histogram"]
    ties = hist == hist.max(1, keepdims=True)
    prob = np.take_along_axis(ties, meta["query_tok"] - meta["value0"], 1) / ties.sum(1, keepdims=True)
    return float(prob.mean())


def summarize_load(logits, meta):
    pos = torch.as_tensor(meta["query_pos"], device=logits.device)
    target = torch.as_tensor(meta["query_tok"], device=logits.device)
    z = logits.gather(1, pos[..., None].expand(-1, -1, logits.shape[-1]))
    hit = z.argmax(-1) == target
    ce = t.token_ce(z, target)
    chunks = meta["probe_write_chunk"]
    acc = hit.detach().cpu().numpy()
    return dict(acc=float(hit.float().mean()), ce=float(ce.mean()), n_probes=int(hit.numel()),
                B_nb=binding_free_baseline(meta), chance=1 / meta["value_histogram"].shape[1],
                by_write_chunk={str(c): float(acc[chunks == c].mean()) for c in np.unique(chunks)})


@dataclasses.dataclass
class LoadTask(t.Task):
    pool: int = 256
    n_values: int = 16
    n_fill: int = 16
    n_probe: int = 16
    events_per_chunk: int = 16
    layout: str = "main"
    gap: int = 268
    control_load: int = 128

    def raw(self, rng, n_seq):
        return echo_load_batch(rng, n_seq, self.T, n_pairs=self.n_keys, n_probe=self.n_probe,
            chunk=self.chunk, events_per_chunk=self.events_per_chunk, layout=self.layout,
            vocab=LoadVocab(self.pool, self.n_values, self.n_fill), gap=self.gap,
            control_load=self.control_load)


def make_task(cfg, cell):
    layout = cell.get("layout", "main")
    n = cell["load"]
    chunk, gap, probes = cfg.get("chunk", 64), cfg.get("gap", 268), cfg.get("n_probe", 16)
    vocab = LoadVocab(cfg.get("pool", 256))
    control = cfg.get("control_load", 128)
    return LoadTask(name="echo_load", V=vocab.V,
        T=load_length(n, chunk, gap, probes, layout, control), window=cfg.get("window", 64),
        chunk=chunk, steps=cfg.get("steps", 1500), n_eval=cfg.get("n_eval", 512), n_val=0,
        mode="echo_load", n_keys=n, n_layers=cfg.get("n_layers", 4), pool=vocab.pool,
        n_probe=probes, events_per_chunk=cfg.get("events_per_chunk", chunk // 4),
        layout=layout, gap=gap, control_load=control)


def batches(task, raw, meta, dev, batch):
    for lo in range(0, len(raw), batch):
        yield task.tensors(raw[lo:lo + batch], slice_meta_load(meta, lo, lo + batch), dev)


@torch.no_grad()
def evaluate_load(model, task, st, arm, raw, meta, *, mode="stream", observer=None):
    outputs = []
    for idx, y, m in batches(task, raw, meta, next(model.parameters()).device, st.eval_batch):
        if mode == "w0":
            z = model.read(model.fast_block(model.backbone(idx), idx, model.slow()))
        elif mode == "stream" and observer is None:
            z = t.eval_batch_logits(model, task, arm, idx, y, 10., st.g_clip, task.chunk,
                                    t.GradStats(), meta=m)
        else:
            sel = t.lin_selector(arm, m["write_pos"], len(idx), task.T, idx.device)
            freeze = torch.as_tensor(m["query_pos"].min(1) // task.chunk, device=idx.device)
            z = t.eval_sequential(model, idx, y, model.etas(10.), st.g_clip, task.chunk,
                t.GradStats(), sel, rule=t.arm_inner_loss(arm),
                freeze_from=freeze if mode == "freeze" else None,
                snapshot_chunk=m["snapshot_chunk"] if mode == "snap" else None,
                observer=observer)
        pos = torch.as_tensor(m["query_pos"], device=z.device)
        outputs.append(z.gather(1, pos[..., None].expand(-1, -1, z.shape[-1])).cpu())
    probe_z = torch.cat(outputs)
    compact = dict(meta, query_pos=np.broadcast_to(np.arange(task.n_probe), (len(raw), task.n_probe)).copy())
    return summarize_load(probe_z, compact), probe_z.argmax(-1).numpy()


@torch.no_grad()
def write_log_load(model, task, st, arm, raw, meta, *, geometry=False):
    """Observe the inherited streaming recurrence; no independent fast update."""
    collected, features, residual_batches = [], [], []
    dev = next(model.parameters()).device
    for idx, y, m in batches(task, raw, meta, dev, st.eval_batch):
        h = model.backbone(idx).detach()
        x = model.fast_input(h, idx).detach()
        wp = torch.as_tensor(m["write_pos"], device=dev)
        qp = torch.as_tensor(m["query_pos"], device=dev)
        rows = torch.arange(len(idx), device=dev)[:, None]
        w0_logits = model.read(model.fast_block(h, idx, model.slow()))
        resid = 1 - (-t.token_ce(w0_logits, y)).exp()[rows, wp]
        residual_batches.append(resid.cpu().numpy())
        local = []
        fw = {key: torch.zeros(len(idx), pos.shape[1], st.fast_mult * st.d_model,
                              device=dev) for key, pos in (("write_stream", wp), ("query_stream", qp))}

        def observe(c, logits, reference, W, raw_grads, grads, norm, clipped):
            sl = slice(c * task.chunk, (c + 1) * task.chunk)
            wc = wp // task.chunk == c
            pc = qp // task.chunk == c
            types = ("write" if wc.any() else "probe" if pc.any() else "filler")
            delta = torch.stack([(g.detach() * e.detach()).flatten(1).square().sum(1)
                                  for g, e in zip(grads, model.etas(10.))]).sum(0).sqrt()
            row = dict(chunk=c, type=types, clip_frac=float(clipped.float().mean()),
                       delta_norm=float(delta.mean()), inner_norm=float(norm.mean()))
            if wc.any():
                r = resid[wc]
                row.update(r=float(r.median()), r_q10=float(torch.quantile(r, .1)),
                           r_q90=float(torch.quantile(r, .9)))
                with torch.enable_grad():
                    z = model.read(model.fast_block(h[:, sl], idx[:, sl], W))
                    selector = t.lin_selector(arm, m["write_pos"], len(idx), task.T, dev)
                    inner = t.inner_token_loss(t.arm_inner_loss(arm), z, y[:, sl],
                        sel=None if selector is None else selector[:, sl],
                        reference=None if reference is None else reference[:, sl])
                    mask = torch.zeros_like(y[:, sl], dtype=torch.bool)
                    rr = rows.expand_as(wp)
                    mask[rr[wc], (wp - c * task.chunk)[wc]] = True
                    kv = torch.autograd.grad((inner * mask).mean(1).sum(), W)
                    drive = torch.stack([(g.detach() * e.detach()).flatten(1).square().sum(1)
                              for g, e in zip(kv, model.etas(10.))]).sum(0).sqrt()
                row["kv_drive"] = float(drive.mean())
            local.append(row)
            if geometry:
                hidden = F.gelu(torch.bmm(x[:, sl], W[0].transpose(1, 2)) + W[1][:, None]).detach()
                for key, pos in (("write_stream", wp), ("query_stream", qp)):
                    mask = pos // task.chunk == c
                    gathered = hidden[rows, (pos - c * task.chunk).clamp(0, task.chunk - 1)]
                    fw[key].add_(gathered * mask[..., None])

        sel = t.lin_selector(arm, m["write_pos"], len(idx), task.T, dev)
        t.eval_sequential(model, idx, y, model.etas(10.), st.g_clip, task.chunk,
                          t.GradStats(), sel, h=h, rule=t.arm_inner_loss(arm), observer=observe)
        collected.append(local)
        if geometry:
            w1, b1 = model.slow()[:2]
            features.append(dict(x_write=x[rows, wp].cpu(), x_query=x[rows, qp].cpu(),
                hidden_w0_write=F.gelu(F.linear(x[rows, wp], w1, b1)).cpu(),
                hidden_w0_query=F.gelu(F.linear(x[rows, qp], w1, b1)).cpu(),
                hidden_stream_write=fw["write_stream"].cpu(), hidden_stream_query=fw["query_stream"].cpu()))
    sizes = np.array([min(st.eval_batch, len(raw) - lo) for lo in range(0, len(raw), st.eval_batch)])
    profile = []
    for c in range(task.T // task.chunk):
        profile.append({key: value if isinstance(value, str) or key == "chunk" else
                        float(np.average([rows[c][key] for rows in collected], weights=sizes))
                        for key, value in collected[0][c].items()})
    residuals = [r for r in profile if r["type"] == "write"]
    all_residuals = np.concatenate(residual_batches)
    for row in residuals:
        values = all_residuals[meta["write_pos"] // task.chunk == row["chunk"]]
        row.update(r=float(np.median(values)), r_q10=float(np.quantile(values, .1)),
                   r_q90=float(np.quantile(values, .9)))
    out = dict(profile=profile, r=float(np.median(all_residuals)),
               r_q10=float(np.quantile(all_residuals, .1)), r_q90=float(np.quantile(all_residuals, .9)),
               phase={kind: {k: float(np.mean([r[k] for r in profile if r["type"] == kind]))
                     for k in ("clip_frac", "delta_norm")} for kind in ("write", "filler", "probe")
                     if any(r["type"] == kind for r in profile)})
    if geometry:
        out["geometry"] = feature_geometry({k: torch.cat([f[k] for f in features]) for k in features[0]}, meta)
    return out


def feature_geometry(features, meta):
    out = {}
    targets = torch.as_tensor(meta["write_val"] - meta["value0"])
    truth = torch.as_tensor(meta["query_tok"] - meta["value0"])
    qi = torch.as_tensor(meta["query_write_idx"])
    labels = F.one_hot(targets, meta["value_histogram"].shape[1]).float()
    for name in ("x", "hidden_w0", "hidden_stream"):
        w, q = features[name + "_write"], features[name + "_query"]
        wn, qn = F.normalize(w, dim=-1), F.normalize(q, dim=-1)
        matched = wn.gather(1, qi[..., None].expand(-1, -1, w.shape[-1]))
        centered = w - w.mean(1, keepdim=True)
        sv = torch.linalg.svdvals(centered).square()
        pair_cos = ((wn.sum(1).square().sum(1) - w.shape[1]) / max(1, w.shape[1] * (w.shape[1] - 1)))
        row = dict(shared_mean_cos=float(pair_cos.mean()),
            write_probe_cos=float((matched * qn).sum(-1).mean()),
            participation_ratio=float((sv.sum(1).square() / sv.square().sum(1).clamp_min(1e-24)).mean()),
            write_norm=float(w.norm(dim=-1).mean()), query_norm=float(q.norm(dim=-1).mean()))
        cross, gram = qn @ wn.transpose(1, 2), wn @ wn.transpose(1, 2)
        row["hebbian"] = float(((cross @ labels).argmax(-1) == truth).float().mean())
        eye = torch.eye(w.shape[1], dtype=torch.float64)[None]
        for lam in (.01, .3):
            # gram + lam*I is SPD. Cholesky avoids the batched MKL LU solve, which can fail for
            # 256x256 batches with >1 CPU thread on some torch builds (seen with 2.14.1).
            factor = torch.linalg.cholesky(gram.double() + lam * eye)
            prediction = cross.double() @ torch.cholesky_solve(labels.double(), factor)
            row[f"ridge_{lam}"] = float((prediction.argmax(-1) == truth).float().mean())
        out[name] = row
    return out


def sha(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def emit(kind, value):
    line = "LOD_" + kind + " " + json.dumps(clean(value), sort_keys=True, separators=(",", ":")) + "\n"
    with _EMIT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def init_worker(emit_lock):
    global _EMIT_LOCK
    _EMIT_LOCK = emit_lock


def finite_update(model, opt, loss, grad_clip):
    """All gradients, including eta logits, are checked before an update."""
    opt.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss)):
        return False, None, False
    loss.backward()
    params = list(model.parameters())
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
        opt.zero_grad(set_to_none=True)
        return False, None, False
    norm = torch.nn.utils.clip_grad_norm_(t.clipped_params(model), grad_clip)
    if not bool(torch.isfinite(norm)):
        opt.zero_grad(set_to_none=True)
        return False, None, False
    opt.step()
    bad_weights = not all(bool(torch.isfinite(p).all()) for p in params)
    return True, float(norm), bad_weights


def certified_divergence(completed, skipped, bad_weights=False):
    return bool(bad_weights or (completed >= 20 and skipped / completed > .05))


def composite_endpoints(observed, diverged):
    return {k: 1 / 16 if diverged else v for k, v in observed.items()}


def cell_id(cell):
    return f"{cell['group']}_{cell['arm']}_N{cell['load']}_s{cell['seed']}"


TIERS = dict(T1=(12, 8, 8), T2=(12, 6, 8), T3=(12, 0, 8),
             T4=(8, 8, 8), T5=(8, 6, 8), cheap=(6, 0, 0))


def manifest_cells(tier, addon=False):
    if tier == "X":
        if addon:
            raise ValueError("LOD-X has no add-on")
        return [dict(group="core", arm=arm, load=n, seed=seed, layout="main")
                for seed in range(216, 228) for n in (16, 32, 64) for arm in ("lin", "so")]
    core, aice, control = TIERS[tier]
    loads = (16, 64, 128) if tier == "cheap" else MAIN
    cells = []

    def add(group, arms, ns, count, layout="main"):
        for seed in range(216, 216 + count):
            for n in ns:
                for arm in arms:
                    cells.append(dict(group=group, arm=arm, load=n, seed=seed, layout=layout))

    add("core", ("lin", "so"), loads, core)
    if tier == "cheap":
        add("anchor", ("fo",), (16,), 4)
        return cells
    add("control", ("lin", "so"), (16,), control, "L16")
    add("anchor", ("fo",), (16, 128), 4)
    add("bridge", ("bridge",), (16, 128), 4)
    add("aice", ("aice",), MAIN, aice)
    if addon:
        if tier != "T1":
            raise ValueError("add-on is sealed only for T1")
        add("stretch", ("lin", "so"), (256,), 4)
        add("linkv", ("linkv",), (128,), 4)
    return cells


def evaluable_hypotheses(tier):
    if tier == "X":
        return ["X1", "X2", "X3", "X4", "R"]
    if tier == "cheap":
        return ["H1a(16,64,128)", "H2(three loads)", "H4(16)"]
    return ["H1a", "H1b", "H1c", "H1d", "H1i", "H2", "H4", "Bridge", "H5"] + (
        ["H3a", "H3b"] if TIERS[tier][1] else [])


def seed_guard(cfg, cells):
    import tomllib
    registry = tomllib.loads((ROOT / "configs/toy/lod/seed-registry.toml").read_text())
    allowed = registry["training"] if cfg["mode"] == "production" else registry["smoke"]
    for cell in cells:
        if cell["seed"] not in allowed:
            raise ValueError("seed rejected at phase boundary")
    if set(s + registry["frozen_code_offset"] for s in registry["training"]) & set(
        registry["training"] + registry["reserved_other_route"] + [registry["eval_seed"]]):
        raise ValueError("frozen-code seed collision")


def source_hashes():
    paths = ["src/tttmem/exp_lod.py", "src/tttmem/lod_analysis.py", "src/tttmem/exp_ttt.py",
             "src/tttmem/run.py", "scripts/analyze_lod.py", "scripts/plot_lod.py", "scripts/lod_figstyle.py",
             "configs/toy/lod/seed-registry.toml",
             "configs/toy/lod/capacity-reference.txt", "configs/toy/lod.toml", "configs/toy/smoke/lod-gpu.toml",
             "configs/toy/smoke/lod.toml", "tests/test_lod.py"]
    paths += ["src/tttmem/lodx_analysis.py", "scripts/analyze_lodx.py", "configs/toy/lod-x.toml", "configs/toy/smoke/lod-x-gpu.toml",
              "configs/toy/smoke/lod-x.toml",
              "tests/test_lodx.py"]
    return {p: file_sha(ROOT / p) for p in paths}


def seal_guard(cfg=None):
    # Opt-in: compares source hashes with a registered seal only when require_seal = true.
    if not (cfg or {}).get("require_seal", False):
        return None
    registered = json.loads((ROOT / cfg.get("seal_file", "lod_seal.json")).read_text())
    if registered["files"] != source_hashes():
        raise ValueError("seal mismatch: code/config/registration changed")
    return registered


def smoke_jobs(cfg, pod=False):
    if cfg.get("protocol") == "lodx":
        loads = (64, 32, 16) if pod else cfg["loads"]
        return [dict(group="smoke", arm=arm, load=n, seed=0, layout="main")
                for n in loads for arm in ("so", "lin")]
    lo, hi = (16, 128) if pod else tuple(cfg["loads"])
    if pod:
        return [dict(group="smoke", arm=a, load=n, seed=0, layout=layout) for a, n, layout in
                (("so", 256, "main"), ("lin", 256, "main"), ("so", 128, "main"),
                 ("aice", 128, "main"), ("so", 16, "L16"), ("lin", 16, "main"))]
    return [dict(group="smoke", arm=a, load=n, seed=0, layout=layout) for a, n, layout in
            (("lin", lo, "main"), ("so", lo, "main"), ("lin", hi, "main"),
             ("so", hi, "main"), ("aice", hi, "main"), ("fo", lo, "main"),
             ("bridge", hi, "main"), ("lin", cfg["events_per_chunk"], "L16"),
             ("so", cfg["events_per_chunk"], "L16"), ("linkv", hi, "main"))]


def queue_can_submit(cell, active, workers, long_workers, resource_limits=None):
    """Shared dispatch/projection rule; never bypass the head of the queue."""
    long = cell["load"] == 256
    if len(active) >= max(workers, long_workers):
        return False
    if sum((c["load"] == 256) == long for c in active) >= (long_workers if long else workers):
        return False
    if resource_limits:
        def peak(c):
            return resource_limits["long_peak_gb" if c["load"] == 256 else "main_peak_gb"]
        if sum(peak(c) for c in active) + peak(cell) > resource_limits["memory_budget_gb"] + 1e-9:
            return False
    return True


def ordered_queue_hours(cells, duration_seconds, workers, long_workers, resource_limits=None):
    """List scheduling, including the final wave and mixed-length VRAM occupancy."""
    active, clock = [], 0.
    for cell in cells:
        while not queue_can_submit(cell, [c for _, c in active], workers, long_workers, resource_limits):
            if not active:
                raise ValueError("cell does not fit the admitted worker/memory caps")
            clock = min(end for end, _ in active)
            active = [(end, c) for end, c in active if end > clock]
        active.append((clock + duration_seconds(cell), cell))
    return max((end for end, _ in active), default=clock) / 3600


def timing_admission(receipt, cfg):
    """Pure deterministic function of measured smoke rates/peaks and sealed cost fields.

    Required smoke lacks lin128/32/64. Linear T interpolation uses measured
    lin16 and SO128 corrected by the measured SO256/lin256 ratio; this timing
    extrapolation is outcome-free and explicitly recorded in the receipt.
    """
    # LOD receipts use the general rule below; a LOD-X receipt with exactly the
    # six X cells uses the stricter X receipt contract.
    x_keys = {(a, n, "main") for a in ("lin", "so") for n in (16, 32, 64)}
    receipt_keys = {(r.get("arm"), r.get("load"), r.get("layout")) for r in receipt.get("cells", [])}
    if cfg.get("protocol") == "lodx" and receipt_keys == x_keys and len(receipt.get("cells", [])) == 6:
        return timing_admission_x(receipt, cfg)
    rows = receipt["cells"]
    by = {(r["arm"], r["load"], r["layout"]): r for r in rows}
    required = [("so", 256, "main"), ("lin", 256, "main"), ("so", 128, "main"),
                ("aice", 128, "main"), ("so", 16, "L16"), ("lin", 16, "main")]
    if any(key not in by for key in required):
        raise ValueError("incomplete smoke rate receipt")
    if any(not math.isfinite(by[k]["ms_step"]) or by[k]["ms_step"] <= 0 for k in required):
        raise ValueError("invalid smoke rates")
    measured_workers = receipt["workers"]
    main_peak = max(by[k]["peak_gb"] for k in required if k[1] != 256)
    long_peak = max(by[k]["peak_gb"] for k in required if k[1] == 256)
    memory_budget = .8 * receipt["total_gb"]
    memory_workers = max(0, math.floor(memory_budget / max(main_peak, 1e-9)))
    workers = min(5, measured_workers, memory_workers)
    long_workers = min(4, measured_workers, max(0, math.floor(memory_budget / max(long_peak, 1e-9))))
    if workers < 1:
        return dict(tier=None, reason="main smoke memory does not fit", workers=0, long_workers=long_workers)
    ratio_so = by[("so", 256, "main")]["ms_step"] / by[("lin", 256, "main")]["ms_step"]
    lin16 = by[("lin", 16, "main")]["ms_step"]
    lin128 = by[("so", 128, "main")]["ms_step"] / ratio_so
    ratio_aice = by[("aice", 128, "main")]["ms_step"] / lin128

    def rate(cell):
        arm, n = cell["arm"], cell["load"]
        if cell["layout"] == "L16":
            ms = by[("so", 16, "L16")]["ms_step"] / (ratio_so if arm == "lin" else 1)
        elif n == 256:
            ms = by[(arm, 256, "main")]["ms_step"]
        else:
            length = load_length(n)
            base = lin16 + (lin128 - lin16) * (length - 448) / (896 - 448)
            ms = base * (ratio_so if arm == "so" else ratio_aice if arm == "aice" else 1)
        # Observed ms is per worker under five-way sharing. Dividing work by
        # the admitted concurrency below applies the occupancy correction once.
        return ms

    price = cfg["hourly_usd"]
    overhead = cfg["eval_factor"] * cfg["vocab_factor"] * cfg["diagnostic_factor"]
    setup = max(cfg["setup_hours"], receipt.get("setup_elapsed_hours", 0))
    retry_hours = rate(dict(arm="so", load=128, layout="main")) * 1500 * overhead / (workers * 3_600_000)
    retry_hours += PROCESS_START_SECONDS / (workers * 3600)
    allowance = retry_hours + cfg.get("shutdown_minutes", 5.) / 60
    resources = dict(main_peak_gb=main_peak, long_peak_gb=long_peak, memory_budget_gb=memory_budget)

    def project(cells):
        h = ordered_queue_hours(cells, lambda c: rate(c) * 1500 * overhead / 1000 + PROCESS_START_SECONDS,
                                workers, long_workers, resources)
        return (setup + h + allowance) * price

    costs = {tier: project(manifest_cells(tier)) for tier in ("T1", "T2", "T3", "T4", "T5")}
    admitted = next((tier for tier, cost in costs.items() if cost <= cfg["budget_usd"]), None)
    addon_cost = project(manifest_cells("T1", True)) if long_workers else None
    return dict(tier=admitted, addon=admitted == "T1" and addon_cost is not None and addon_cost <= cfg["budget_usd"],
        workers=workers, long_workers=long_workers, T_cap=1408, costs_usd=costs,
        main_peak_gb=main_peak, long_peak_gb=long_peak, memory_budget_gb=memory_budget,
        reduced_occupancy_projection=workers < measured_workers,
        addon_cost_usd=addon_cost, setup_hours=setup, allowance_hours=allowance,
        retry_hours=retry_hours, process_start_seconds=PROCESS_START_SECONDS,
        queue_projection="ordered list scheduling with tail, separate caps and mixed VRAM occupancy",
        timing_method="linear T interpolation; SO256/lin256 ratio; AICE128/lin128 ratio",
        smoke_sha256=sha(receipt), evaluable=evaluable_hypotheses(admitted) if admitted else [])


def full_eval_steps(cfg):
    steps, total = cfg["steps"], cfg.get("lr_total", cfg["steps"])
    assert 0 < steps <= total, "steps must be positive and <= lr_total"
    points = cfg.get("full_eval_steps", [steps])
    if points != sorted(set(points)) or not points or points[-1] != steps or points[0] < 1:
        raise ValueError("full_eval_steps must be ordered unique endpoints ending at steps")
    return points


def timing_admission_x(receipt, cfg):
    """Direct measured lin/SO rates at all three loads; X or stop, no fallback.

    eval_factor's excess prices one full evaluation as a fraction of 1500
    training steps. Two evaluations are explicitly priced at 3000 steps.
    Quick logs/telemetry keep the inherited diagnostic and vocabulary proxies.
    """
    points = full_eval_steps(cfg)
    if cfg.get("tier_order", ["X"]) != ["X"]:
        raise ValueError("LOD-X admits only tier X")
    by = {(r["arm"], r["load"], r["layout"]): r for r in receipt["cells"]}
    required = {(a, n, "main") for a in ("lin", "so") for n in (16, 32, 64)}
    if set(by) != required or len(receipt["cells"]) != 6:
        raise ValueError("incomplete or unexpected LOD-X smoke rate receipt")
    if any(r.get("seed") != 0 or not math.isfinite(r["ms_step"]) or r["ms_step"] <= 0
           or not math.isfinite(r["peak_gb"]) or r["peak_gb"] <= 0 for r in by.values()):
        raise ValueError("invalid LOD-X smoke rates/peaks/seeds")
    peak = max(r["peak_gb"] for r in by.values())
    memory_budget = .8 * receipt["total_gb"]
    workers = min(5, cfg["workers"], receipt["workers"], math.floor(memory_budget / peak))
    if workers < 1:
        return dict(tier=None, addon=False, workers=0, long_workers=0,
                    reason="main smoke memory does not fit", costs_usd={"X": None})
    eval_equivalent_steps = 1500 * (cfg.get("eval_factor", 1.08) - 1) * len(points)
    overhead = cfg.get("vocab_factor", 1.05) * cfg.get("diagnostic_factor", 1.03)
    def duration(cell):
        ms = by[(cell["arm"], cell["load"], cell["layout"])]["ms_step"]
        return ms * (cfg["steps"] + eval_equivalent_steps) * overhead / 1000 + PROCESS_START_SECONDS
    cells = manifest_cells("X")
    resources = dict(main_peak_gb=peak, long_peak_gb=peak, memory_budget_gb=memory_budget)
    training_hours = ordered_queue_hours(cells, duration, workers, 0, resources)
    # One worst-case 3000-step retry, allocated to one worker share, as in LOD.
    retry_hours = max(map(duration, cells)) / (workers * 3600)
    allowance = retry_hours + cfg.get("shutdown_minutes", 5.) / 60
    setup = max(cfg.get("setup_hours", 0.), receipt.get("setup_elapsed_hours", 0))
    hours = setup + training_hours + allowance
    cost = hours * cfg.get("hourly_usd", .50)
    admitted = cost <= cfg.get("budget_usd", 3.)
    return dict(tier="X" if admitted else None, addon=False, workers=workers, long_workers=0,
        T_cap=load_length(64), costs_usd={"X": cost}, session_hours=hours,
        production_hours=training_hours, full_eval_steps=points,
        full_eval_equivalent_steps=eval_equivalent_steps, **resources,
        reduced_occupancy_projection=workers < receipt["workers"], addon_cost_usd=None,
        setup_hours=setup, allowance_hours=allowance, retry_hours=retry_hours,
        process_start_seconds=PROCESS_START_SECONDS,
        queue_projection="ordered list scheduling with tail and main VRAM occupancy",
        timing_method="direct six lin/SO load rates; steps plus explicit full-eval allowance",
        smoke_sha256=sha(receipt), evaluable=evaluable_hypotheses("X") if admitted else [])


def final_eval(model, task, st, arm, cfg, cell, output, *, suffix=""):
    raw, meta = task.raw(np.random.default_rng([t.EVAL_SEED, 0]), task.n_eval)
    audits = layout_audit(raw, meta, task)
    endpoints, per_probe = {}, {}
    for name, mode, rule in (("A", "stream", arm), ("A_freeze", "freeze", arm),
          ("A_snap", "snap", arm), ("W0", "w0", arm),
          ("cross_lin", "stream", ARMS["lin"]), ("cross_ce", "stream", ARMS["fo"])):
        summary, predictions = evaluate_load(model, task, st, rule, raw, meta, mode=mode)
        endpoints[name] = summary["acc"]
        if name == "A":
            endpoints["ce"] = summary["ce"]
            per_probe = {k: v for k, v in meta.items() if k in ARRAY_FIELDS}
            per_probe["prediction"] = predictions
    last_task = dataclasses.replace(task, layout="last" if task.layout == "main" else "L16")
    last_raw, last_meta = last_task.raw(np.random.default_rng([t.EVAL_SEED, 3]), task.n_eval)
    endpoints["A_last"] = evaluate_load(model, task, st, arm, last_raw, last_meta)[0]["acc"]
    coupled_eligible = task.n_keys == cfg.get("control_load", 128) or task.layout == "L16"
    if coupled_eligible:
        coupled_raw, coupled_meta = echo_load_batch(np.random.default_rng([t.EVAL_SEED, 4]),
            len(last_raw), task.T, n_pairs=last_task.n_keys, n_probe=task.n_probe,
            chunk=task.chunk, events_per_chunk=task.events_per_chunk, layout="coupled",
            vocab=LoadVocab(task.pool), given=(last_raw, last_meta))
        endpoints["A_coupled"] = evaluate_load(model, task, st, arm, coupled_raw, coupled_meta)[0]["acc"]
        audits.update({"coupled_" + k: v for k, v in layout_audit(coupled_raw, coupled_meta, task).items()})
    n_geom = min(cfg.get("n_geometry", 128), len(raw))
    diagnostic = write_log_load(model, task, st, arm, raw[:n_geom], slice_meta_load(meta, 0, n_geom), geometry=True)
    probes = output / (cell_id(cell) + suffix + "_probes.npz")
    np.savez_compressed(probes, **per_probe)
    return endpoints, dict(B_nb=binding_free_baseline(meta), layout_audit=audits,
                           eval_sha256=sha(raw), diagnostic=diagnostic,
                           probes_sha256=file_sha(probes))


def train_cell(cfg, cell, deadline=math.inf):
    eval_points = full_eval_steps(cfg)
    dev = torch.device(cfg["device"])
    threads = cfg.get("threads_per_worker", 2)
    if dev.type == "cpu" and threads > 2:
        raise ValueError("local CPU thread cap is two")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with _EMIT_LOCK:
        t._mark_process(cfg["process_name"] + "-" + cell_id(cell))
    seed_guard(cfg, [cell])
    if cfg["mode"] == "production":
        seal_guard(cfg)
        if dev.type != "cuda":
            raise ValueError("fresh-seed training requires sealed CUDA production")
    task = make_task(cfg, cell)
    st = t._settings(dict(cfg, eta_lr=.01 if cell["arm"] == "bridge" else 0.))
    arm = ARMS[cell["arm"]]
    model = t.build_model(task, st, dev, cell["seed"], arm)
    opt = t.build_optim(model, st)
    rng = np.random.default_rng([cell["seed"], t.TRAIN_SEED])
    init_hash = sha({k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                     for k, v in model.state_dict().items()})
    frozen_hash = sha(model.frozen_emb.detach().cpu().numpy())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    quick_raw, quick_meta = task.raw(np.random.default_rng([t.EVAL_SEED, 2]), cfg.get("n_quick_eval", 64))
    audits = layout_audit(quick_raw, quick_meta, task)
    if not all(audits.values()):
        result = dict(**cell, cell_id=cell_id(cell), status="LAYOUT_INVALID", layout_audit=audits,
                      config_sha256=sha(cfg), source_hashes=source_hashes(),
                      reason="layout audit failed before training")
        atomic_json(output / (cell_id(cell) + ".json"), result)
        emit("CELL", result)
        return result
    acquired = t50 = None
    intermediate = {}
    skipped, diverged, times, logs, done, first_sha = 0, False, [], [], 0, None
    train_started = time.monotonic()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    bad_weights = False
    for step in range(task.steps):
        if time.time() >= deadline:
            break
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * t.lr_scale(step, st.warmup, st.lr_total or task.steps, st.lr_floor)
        raw, meta = task.raw(rng, st.batch)
        if first_sha is None:
            first_sha = sha(raw)
        idx, y, _ = task.tensors(raw, meta, dev)
        t._sync(dev)
        tic = time.perf_counter()
        sel = t.lin_selector(arm, meta["write_pos"], len(idx), task.T, dev)
        loss = t.arm_loss(model, arm, idx, y, 10., st.g_clip, task.chunk, t.GradStats(), sel=sel, meta=meta)
        updated, norm, bad_weights = finite_update(model, opt, loss, st.grad_clip)
        skipped += not updated
        done = step + 1
        diverged = certified_divergence(done, skipped, bad_weights)
        if diverged:
            # Certification is durable before any further CUDA synchronization,
            # diagnostics or checkpoint I/O can fail. A crash cannot erase it.
            atomic_json(output / (cell_id(cell) + ".json"), dict(**cell,
                cell_id=cell_id(cell), status="DIVERGED", certified_divergence=True,
                completed_steps=done, skipped_steps=skipped, config_sha256=sha(cfg),
                source_hashes=source_hashes(), layout_audit=audits, observed={},
                composite={k: 1/16 for k in ENDPOINT_NAMES}, acquired=acquired,
                retained=False, t50=t50, t50_censored=t50 is None, logs=logs,
                first_batch_sha256=first_sha, init_sha256=init_hash,
                frozen_code_sha256=frozen_hash, device=str(dev), tf32=False, **intermediate))
        t._sync(dev)
        times.append(1000 * (time.perf_counter() - tic))
        if not bad_weights and (done % cfg.get("log_every", 25) == 0 or done == task.steps or diverged):
            quick = evaluate_load(model, task, st, arm, quick_raw, quick_meta)[0]
            on_grid = done % cfg.get("log_every", 25) == 0
            if on_grid and acquired is None and quick["acc"] >= .90:
                acquired = done
            if on_grid and t50 is None and quick["acc"] >= .50:
                t50 = done
            write = write_log_load(model, task, st, arm, quick_raw, quick_meta)
            write["measurement"] = "own-rule streaming eval on the fixed quick set"
            row = dict(cell_id=cell_id(cell), step=done, quick=quick, write=write,
                loss=float(loss.detach()), outer_grad_norm=norm, outer_clipped=norm is not None and norm > st.grad_clip,
                lr=opt.param_groups[0]["lr"], skipped=skipped, certified_divergence=diverged)
            if done == 750 or done == task.steps:
                full_raw, full_meta = task.raw(np.random.default_rng([t.EVAL_SEED, 0]), task.n_eval)
                row["full_A"] = evaluate_load(model, task, st, arm, full_raw, full_meta)[0]["acc"]
            logs.append(clean(row))
            emit("LOG", row)
        if done in eval_points[:-1] and not diverged:
            obs, detail = final_eval(model, task, st, arm, cfg, cell, output, suffix=f"_{done}")
            comp = composite_endpoints({k: obs.get(k) for k in ENDPOINT_NAMES}, False)
            intermediate.update({f"observed_{done}": obs, f"composite_{done}": comp,
                f"acquired_{done}": acquired,
                f"retained_{done}": acquired is not None and comp["A"] >= .9,
                f"eval_step_{done}": done, f"details_{done}": detail})
            emit("EVAL", dict(cell_id=cell_id(cell), step=done, composite=comp,
                              acquired=acquired, retained=intermediate[f"retained_{done}"]))
        if diverged:
            break
    status = "DIVERGED" if diverged else "COMPLETE" if done == task.steps else "TRUNCATED"
    observed, details = {}, dict(layout_audit=audits)
    if status != "TRUNCATED" and not bad_weights:
        try:
            observed, details = final_eval(model, task, st, arm, cfg, cell, output)
        except Exception as exc:
            if not diverged:
                raise
            details["final_eval_error"] = type(exc).__name__ + ": " + str(exc)
            details["cuda_error"] = is_cuda_error(cfg, exc)
            # Incomplete diagnostics cannot erase a certified rule failure.
            (output / (cell_id(cell) + "_probes.npz")).unlink(missing_ok=True)
    composite = composite_endpoints({k: observed.get(k) for k in ENDPOINT_NAMES}, diverged)
    if diverged:
        for endpoint in eval_points[:-1]:
            if f"composite_{endpoint}" not in intermediate:
                intermediate.update({f"composite_{endpoint}": {k: 1/16 for k in ENDPOINT_NAMES},
                    f"observed_{endpoint}": {}, f"acquired_{endpoint}": acquired,
                    f"retained_{endpoint}": False, f"eval_step_{endpoint}": None,
                    f"details_{endpoint}": {"certified_divergence": True}})
    result = dict(**cell, cell_id=cell_id(cell), status=status, completed_steps=done, skipped_steps=skipped,
        config_sha256=sha(cfg), source_hashes=source_hashes(),
        certified_divergence=diverged, observed=observed, composite=composite,
        acquired=acquired, retained=acquired is not None and composite.get("A") is not None and composite["A"] >= .9,
        t50=t50, t50_censored=t50 is None, logs=logs, first_batch_sha256=first_sha,
        init_sha256=init_hash, frozen_code_sha256=frozen_hash,
        ms_step=float(np.median(times)) if times else None, wall_s=time.monotonic() - train_started,
        peak_gb=torch.cuda.max_memory_reserved(dev) / 1e9 if dev.type == "cuda" else 0.,
        eta=[float(e.detach()) for e in model.etas(10.)], device=str(dev), tf32=False,
        **details, **intermediate)
    if diverged:
        atomic_json(output / (cell_id(cell) + ".json"), result)
    if status != "TRUNCATED":
        from .exp_chr import _to_cpu
        state = dict(model=_to_cpu(model.state_dict()), optimizer=_to_cpu(opt.state_dict()),
            data_rng=copy.deepcopy(rng.bit_generator.state), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
            completed_steps=done, config_sha256=sha(cfg), source_hashes=source_hashes(),
            first_batch_sha256=first_sha, cell=cell, skipped_steps=skipped)
        target = output / (cell_id(cell) + "_state.pt")
        torch.save(state, target.with_suffix(".tmp"))
        target.with_suffix(".tmp").replace(target)
        result["checkpoint"] = target.name
    atomic_json(output / (cell_id(cell) + ".json"), result)
    emit("CELL", {k: v for k, v in result.items() if k != "logs"})
    return result


def attempt_path(cfg, cell):
    return Path(cfg["output_dir"]) / (cell_id(cell) + "_attempts.json")


def read_attempt(cfg, cell):
    path = attempt_path(cfg, cell)
    return json.loads(path.read_text()) if path.exists() else dict(attempts=0, errors=[])


def missing_cell(cfg, cell, reason):
    result = dict(**cell, cell_id=cell_id(cell), status="MISSING", reason=reason,
                  config_sha256=sha(cfg), source_hashes=source_hashes(),
                  errors=read_attempt(cfg, cell)["errors"])
    atomic_json(Path(cfg["output_dir"]) / (cell_id(cell) + ".json"), result)
    return result


def existing_cell(cfg, cell):
    result_path = Path(cfg["output_dir"]) / (cell_id(cell) + ".json")
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get("config_sha256") != sha(cfg) or old.get("source_hashes") != source_hashes():
            raise ValueError("stale cell receipt; use a fresh diagnostic output directory or the exact sealed source")
        if old["status"] in ("COMPLETE", "DIVERGED", "MISSING", "LAYOUT_INVALID"):
            return old
        if old["status"] == "TRUNCATED":
            return missing_cell(cfg, cell, "budget stop; truncated cell is never retried")
    return None


def prepare_attempt(cfg, cell, deadline):
    old = existing_cell(cfg, cell)
    if old is not None:
        return old
    previous = read_attempt(cfg, cell)
    if previous["attempts"] >= 2 or time.time() >= deadline:
        return missing_cell(cfg, cell, "infrastructure retry exhausted" if previous["attempts"] >= 2 else "budget stop")
    # Reserve in the supervisor, so death before the worker entry point still
    # consumes an attempt. A restart cannot grant a third attempt.
    previous["attempts"] += 1
    atomic_json(attempt_path(cfg, cell), previous)
    return None


def is_cuda_error(cfg, exc):
    error = type(exc).__name__ + ": " + str(exc)
    return cfg["device"] == "cuda" and (isinstance(exc, torch.OutOfMemoryError) or any(
        token in error.lower() for token in ("cuda", "cudnn", "cublas", "device-side assert", "illegal memory access", "xid")))


def record_failure(cfg, cell, deadline, exc):
    previous = read_attempt(cfg, cell)
    error = type(exc).__name__ + ": " + str(exc)
    previous["errors"].append(error)
    gpu_error = is_cuda_error(cfg, exc)
    previous["last_cuda_error"] = gpu_error
    atomic_json(attempt_path(cfg, cell), previous)
    emit("INFRA", dict(cell_id=cell_id(cell), **previous))
    if previous["attempts"] < 2 and time.time() < deadline:
        return dict(**cell, cell_id=cell_id(cell), status="RETRY", cuda_error=gpu_error, error=error)
    result = missing_cell(cfg, cell, "infrastructure retry exhausted" if previous["attempts"] >= 2 else "budget stop")
    result["cuda_error"] = gpu_error
    return result


def execute_with_retry(cfg, cell, deadline, runner=None, attempt_reserved=False):
    """One attempt only. RETRY is resubmitted by the supervisor to a fresh worker."""
    if not attempt_reserved:
        old = prepare_attempt(cfg, cell, deadline)
        if old is not None:
            return old
    try:
        return (runner or train_cell)(cfg, cell, deadline)
    except Exception as exc:
        old = existing_cell(cfg, cell)
        if old is not None and old["status"] == "DIVERGED":
            old["post_certification_error"] = type(exc).__name__ + ": " + str(exc)
            old["cuda_error"] = is_cuda_error(cfg, exc)
            if not old.get("probes_sha256"):
                (Path(cfg["output_dir"]) / (cell_id(cell) + "_probes.npz")).unlink(missing_ok=True)
            atomic_json(Path(cfg["output_dir"]) / (cell_id(cell) + ".json"), old)
            emit("CELL", {k: v for k, v in old.items() if k != "logs"})
            return old
        return record_failure(cfg, cell, deadline, exc)


def close_pool(pool, abort=False):
    if abort:
        terminate = getattr(pool, "terminate_workers", None)
        if terminate is not None:  # Public API on Python >= 3.14.
            terminate()
            return
        # Python 3.11–3.13 have no public way to kill active pool workers.
        # Copy the process list before shutdown clears the executor's mapping.
        processes = list((getattr(pool, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    pool.shutdown(wait=True, cancel_futures=True)


def dispatch(cfg, cells, workers, deadline, long_workers=None, resource_limits=None):
    """One ordered queue, once-only retries, and a persistent infrastructure stop.

    CUDA cells always use fresh child processes, even with a single worker.
    The CPU diagnostic runs inline (one local torch process at a time); it is
    not a production execution path.
    """
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool
    from collections import deque
    import multiprocessing as mp

    if cfg["device"] == "cpu" and workers != 1:
        raise ValueError("one local torch process at a time")
    long_workers = workers if long_workers is None else long_workers
    if workers < 1 or any(c["load"] == 256 for c in cells) and long_workers < 1:
        raise ValueError("invalid admitted worker caps")
    todo, pending, results = deque(cells), {}, {}
    stop_path = Path(cfg["output_dir"]) / "session_infra.json"
    stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    if stop and (stop.get("config_sha256") != sha(cfg) or stop.get("source_hashes") != source_hashes()):
        raise ValueError("stale infrastructure stop receipt")
    consecutive_gpu_errors = 0
    pool = None
    context = mp.get_context("spawn")
    emit_lock = context.RLock()
    init_worker(emit_lock)

    def new_pool():
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        return cf.ProcessPoolExecutor(max_workers=max(workers, long_workers),
            mp_context=context, max_tasks_per_child=1, initializer=init_worker, initargs=(emit_lock,))

    def accept(cell, result):
        nonlocal stop, consecutive_gpu_errors
        if result.get("cuda_error"):
            consecutive_gpu_errors += 1
        else:
            consecutive_gpu_errors = 0
        if consecutive_gpu_errors >= 3:
            stop = dict(status="INFRA_GPU_ERRORS", reason="three consecutive CUDA-error attempts",
                        cell_id=cell_id(cell), error=result.get("error", result.get("errors")),
                        config_sha256=sha(cfg), source_hashes=source_hashes())
            atomic_json(stop_path, stop)
            emit("STOP", stop)
        if result["status"] == "RETRY":
            todo.appendleft(cell)
        else:
            results[cell_id(cell)] = result

    def rebuild_pool(exc):
        nonlocal pool, emit_lock
        stranded = list(pending.items())
        pending.clear()
        close_pool(pool, abort=True)
        pool = None
        # A killed process may have owned the old stdout lock. Replacement
        # workers and the supervisor must not inherit that poisoned lock.
        emit_lock = context.RLock()
        init_worker(emit_lock)
        emit("POOL_REBUILD", dict(error=str(exc), affected_cells=[cell_id(c) for _, c in stranded]))
        for future, cell in stranded:
            old = existing_cell(cfg, cell)
            if old is None and future is not None and future.done():
                try:
                    old = future.result()
                except BrokenProcessPool:
                    pass
            accept(cell, old if old is not None else record_failure(cfg, cell, deadline, exc))
        if not stop and time.time() < deadline:
            pool = new_pool()

    try:
        if cfg["device"] != "cpu" and not stop:
            pool = new_pool()
        while (todo or pending) and not stop:
            while todo and time.time() < deadline and queue_can_submit(
                    todo[0], list(pending.values()), workers, long_workers, resource_limits):
                cell = todo.popleft()
                old = prepare_attempt(cfg, cell, deadline)
                if old is not None:
                    accept(cell, old)
                elif cfg["device"] == "cpu":
                    accept(cell, execute_with_retry(cfg, cell, deadline, attempt_reserved=True))
                else:
                    try:
                        future = pool.submit(execute_with_retry, cfg, cell, deadline, attempt_reserved=True)
                        pending[future] = cell
                    except BrokenProcessPool as exc:
                        pending[None] = cell
                        rebuild_pool(exc)
                        break
                if stop:
                    break
            if stop or not pending:
                if todo and time.time() < deadline and not stop:
                    if not queue_can_submit(todo[0], [], workers, long_workers, resource_limits):
                        raise ValueError("queued cell exceeds admitted memory caps")
                    continue
                break
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            # Preserve submission order when several futures become ready at once.
            for future in [f for f in pending if f in done]:
                cell = pending[future]
                try:
                    result = future.result()
                except BrokenProcessPool as exc:
                    rebuild_pool(exc)
                    break
                pending.pop(future)
                accept(cell, result)
                if stop:
                    break
    finally:
        if pool is not None:
            close_pool(pool, abort=bool(stop))
        init_worker(threading.RLock())
    # Explicit missing receipts make both the deadline and circuit stop auditable.
    for cell in cells:
        cid = cell_id(cell)
        if cid not in results:
            results[cid] = existing_cell(cfg, cell) or missing_cell(
                cfg, cell, "session stopped: INFRA_GPU_ERRORS" if stop else "budget stop")
    return [results[cell_id(c)] for c in cells]


def pod_smoke(cfg):
    lodx = cfg.get("protocol") == "lodx"
    settings = dict(cfg, mode="pod_smoke", steps=30, log_every=30,
                    process_name="TTM-lodx-81-podsmoke" if lodx else "TTM-lod-73-podsmoke",
                    output_dir=cfg.get("smoke_output_dir", "outputs_lod_podsmoke"))
    settings.pop("full_eval_steps", None)  # The timing smoke has its own 30-step endpoint.
    jobs = smoke_jobs(settings, pod=True)
    seed_guard(settings, jobs)
    # The N256 pair runs in a five-worker occupancy cohort. The sixth job is
    # queued without lowering occupancy during the measured steps.
    results = dispatch(settings, jobs, 5, time.time() + 1800)
    if len(results) != len(jobs) or any(r["status"] != "COMPLETE" for r in results):
        raise RuntimeError("pod smoke incomplete; production remains locked")
    receipt = dict(version=1, workers=5, device="cuda", total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
        recipe_sha256=sha({k: cfg[k] for k in RECIPE_KEYS}), source_hashes=source_hashes(),
        cells=[{k: r[k] for k in ("arm", "load", "layout", "seed", "ms_step", "peak_gb")} for r in results])
    atomic_json(cfg.get("smoke_receipt", "outputs_lod_podsmoke/rates.json"), receipt)
    return receipt


RECIPE_KEYS = ("d_model", "n_layers", "n_heads", "backbone_mult", "fast_mult", "window", "chunk",
               "batch", "pool", "events_per_chunk", "n_probe", "gap", "control_load", "g_clip",
               "grad_clip", "lr", "eta_lr", "warmup", "lr_total", "lr_floor", "tf32")


def run(cfg, seed):
    cfg = copy.deepcopy(cfg)
    full_eval_steps(cfg)
    cfg["device"] = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"]
    if cfg["device"] == "auto":
        cfg["device"] = "cpu"
    t._mark_process(cfg["process_name"])
    torch.set_num_threads(min(2, cfg.get("threads_per_worker", 2)))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    mode = cfg["mode"]
    if mode not in ("diagnostic", "pod_smoke", "production"):
        raise ValueError("unknown protocol phase")
    if mode != "diagnostic":
        if cfg["device"] != "cuda":
            raise ValueError("production and pod smoke require CUDA")
        seal_guard(cfg)
    if mode == "pod_smoke":
        receipt = pod_smoke(cfg)
        admission = timing_admission(receipt, cfg)
        emit("TIMING", admission)
        return dict(mode=mode, admission=admission, receipt=cfg["smoke_receipt"])
    manifest_path = output / "manifest.json"
    if mode == "production" and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_hashes"] != source_hashes() or manifest["config_sha256"] != sha(cfg):
            raise ValueError("resume refused: source/config mismatch")
        admission = manifest["admission"]
        cells = manifest["cells"]
    else:
        if mode == "production":
            receipt_path = Path(cfg["smoke_receipt"])
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else pod_smoke(cfg)
            if receipt["source_hashes"] != source_hashes() or receipt["recipe_sha256"] != sha({k: cfg[k] for k in RECIPE_KEYS}):
                raise ValueError("smoke receipt mismatch")
            if any(r["seed"] != 0 for r in receipt["cells"]):
                raise ValueError("smoke touched a fresh seed")
            admission = timing_admission(receipt, cfg)
            if admission["tier"] is None:
                emit("STOP", admission)
                return dict(mode=mode, status="NO_TIER_FITS", admission=admission)
            cells = manifest_cells(admission["tier"], admission["addon"])
        else:
            cells = smoke_jobs(cfg)
            admission = dict(tier="diagnostic", workers=1, long_workers=1, addon=False,
                             evaluable=[], T_cap=max(make_task(cfg, c).T for c in cells))
        seed_guard(cfg, cells)
        manifest = dict(version=1, mode=mode, config=cfg, config_sha256=sha(cfg),
            source_hashes=source_hashes(),
            plan_sha256=file_sha(ROOT / cfg["plan_file"]) if cfg.get("plan_file") else None,
            admission=admission, cells=cells, created_before_production=True,
            budget_usd=cfg.get("budget_usd", 1), spent_hours=0.)
        if mode == "production":
            # Absolute session deadline survives a killed supervisor; resuming
            # cannot reset the sealed budget.
            manifest["production_started_at"] = time.time()
            remaining = cfg["budget_usd"] / cfg["hourly_usd"] - admission["setup_hours"] - cfg["shutdown_minutes"] / 60
            manifest["hard_stop_at"] = time.time() + max(0, remaining) * 3600
        atomic_json(manifest_path, manifest)
    seed_guard(cfg, cells)
    emit("MANIFEST", manifest)
    started = time.monotonic()
    if mode == "production":
        deadline = manifest["hard_stop_at"]
    else:
        deadline = time.time() + 600
    results = []
    # Submission order protects the mandatory groups without completion barriers.
    try:
        resources = {k: admission[k] for k in ("main_peak_gb", "long_peak_gb", "memory_budget_gb")} if mode == "production" else None
        results = dispatch(cfg, cells, admission["workers"], deadline,
                           long_workers=admission["long_workers"], resource_limits=resources)
        infra_path = output / "session_infra.json"
        if infra_path.exists():
            manifest["infra_stop"] = json.loads(infra_path.read_text())
    finally:
        manifest["spent_hours"] += (time.monotonic() - started) / 3600
        atomic_json(manifest_path, manifest)
    # Receipt files are the resumable, authoritative analyzer inputs.
    if cfg.get("protocol") == "lodx":
        from .lodx_analysis import analyze_directory
    else:
        from .lod_analysis import analyze_directory
    scores = analyze_directory(output, diagnostic=mode == "diagnostic")
    atomic_json(output / "scores.json", scores)
    counts = {status: sum(r["status"] == status for r in results) for status in ("COMPLETE", "DIVERGED", "TRUNCATED", "MISSING")}
    emit("SUMMARY", dict(counts=counts, claims=scores["claims"], output_dir=str(output)))
    return dict(mode=mode, status=manifest.get("infra_stop", {}).get("status", "FINISHED"),
        manifest=str(manifest_path), config_sha256=sha(cfg),
        plan_sha256=manifest["plan_sha256"], admission=admission, counts=counts, claims=scores["claims"])
