"""E1-VE-D: why FO-CE loses the agent's own overwrites (A) without a copy path. Eval only.

E1-VE wave 1 showed that FO-CE with the C copy edge blocked keeps N but
still loses A, reverting to the pre-action value. At those A writes the memory
gives the old value only chance probability (``oldp_A`` ~ 1/16), so a CE write
has no erase term. This module reads the saved wave-1 checkpoints and asks, on
the same 256 clean evaluation episodes:

* **readout**: at the R queries, the answer with the chunk-0 (E) or the chunk-1
  (C) fast-weight update subtracted, and each update's share of the A margin
  ``z[v1] - z[v0]``;
* **injection**: the streaming evaluation re-run with one extra term in the
  chunk-1 inner loss at the A writes only. ``erase_v0`` adds ``beta * log p(v0)``,
  which is the erase a CE write makes when the memory predicts v0 with
  probability ``beta``. ``erase_rand`` erases a random third value instead (a
  control of the same size). ``add_v1`` adds ``beta`` more CE on v1, a larger add
  without an erase;
* **key match**: under ``W0``, the dot product of the fast block's hidden
  activation at the R query with that at the location's E write and C write. A
  ``W2`` write at position p is read at query q in proportion to ``<a_q, a_p>``.

Every stream is ``exp_ttt.eval_sequential``, unchanged; the readout uses its
``observer`` hook and the injection its ``inner_transform`` hook. The base stream
must reproduce the wave-1 final accuracies (gate G0).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as tt
from . import exp_ve as ve

CONDITIONS = ("base", "erase_v0", "erase_rand", "add_v1")
RAND_SEED = 4_242
G0_TOL = 0.02        # base accuracy must match the wave-1 final within this, per type


@dataclass
class DiagSettings:
    process_name: str = "TTM-ved-01"
    device: str = "cpu"
    threads: int = 8
    ckpt_dir: str = ""
    reference_csv: str = ""
    beta: float = 0.5
    n_eval: int = 256
    eval_batch: int = 32
    # wave-1 model and layout settings ([ve] keys); empty = the wave-1 defaults
    ve: dict = field(default_factory=dict)
    cells: list[dict] = field(default_factory=list)


def settings(cfg: dict) -> DiagSettings:
    known = {f.name for f in dataclasses.fields(DiagSettings)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown [ve_diag] keys: {unknown}")
    return DiagSettings(**cfg)


def load_model(vst: ve.VeSettings, lay: ve.Layout, cell: dict, ckpt_dir: str,
               dev: torch.device) -> tt.TTTModel:
    """Wave-1 final weights of ``cell``, built exactly as the run built them."""
    path = Path(ckpt_dir) / f"ve__{ve.cell_id(cell).replace('/', '__')}.pt"
    model = ve.build(vst, lay, int(cell["seed"]), dev)
    state = torch.load(path, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def c_chunk(lay: ve.Layout) -> int:
    """The chunk holding the C events (chunk 1 in the wave-1 layout)."""
    return lay.c_start // lay.chunk


def _a_writes(lay: ve.Layout, meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """C-chunk mask of the A write positions, (B, chunk) bool, and their in-chunk indices."""
    B = meta["c_write"].shape[0]
    local = meta["c_write"] - c_chunk(lay) * lay.chunk
    mask = torch.zeros(B, lay.chunk, dtype=torch.bool, device=local.device)
    mask.scatter_(1, local, meta["c_type"] == ve.A)
    return mask, local


def injection_targets(cond: str, lay: ve.Layout, meta: dict, y: torch.Tensor,
                      gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """C-chunk A-write mask and the token whose log-probability the condition adds there."""
    if cond not in CONDITIONS[1:]:
        raise ValueError(f"unknown injection {cond}")
    mask, local = _a_writes(lay, meta)
    cw = meta["c_write"]
    v1 = y.gather(1, cw) - lay.val0                         # the observed value at the write
    v0 = meta["c_old"].clamp_min(lay.val0) - lay.val0       # N rows are masked out below
    if cond == "erase_v0":
        val = v0
    elif cond == "erase_rand":
        # a third value: neither the old nor the new one (A has v0 != v1)
        safe_v0 = torch.where(meta["c_type"] == ve.A, v0, (v1 + 1) % lay.n_val)
        val = ve._other2(v1.cpu(), safe_v0.cpu(), gen, lay.n_val).to(v1.device)
    else:
        val = v1
    target = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
    target.scatter_(1, local, lay.val0 + val)
    return mask, target


def injection(cond: str, lay: ve.Layout, meta: dict, y: torch.Tensor, beta: float,
              gen: torch.Generator):
    """``inner_transform`` adding the condition's term at the C-chunk A writes (None for base)."""
    if cond == "base":
        return None
    mask, target = injection_targets(cond, lay, meta, y, gen)
    sign = -1.0 if cond == "add_v1" else 1.0
    weight = mask.to(torch.float32) * (sign * beta)
    cc = c_chunk(lay)

    def transform(c: int, logits: torch.Tensor, targets: torch.Tensor,
                  inner: torch.Tensor) -> torch.Tensor:
        if c != cc:
            return inner
        logp = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
        return inner + weight * logp

    return transform


def _queries(logits_at_q: torch.Tensor, meta: dict) -> dict[str, torch.Tensor]:
    """Per-type hit and stale counts at the R queries from (B, K, V) logits."""
    pred = logits_at_q.argmax(-1)
    hit = (pred == meta["query_tok"]).float()
    stale = (pred == meta["query_old"]).float()
    qtype = F.one_hot(meta["query_type"], 5).float()
    out = {}
    for name, val in (("hit", hit), ("stale", stale)):
        per = torch.einsum("bk,bkt->t", val, qtype)
        for i, k in enumerate(ve.TYPES):
            out[f"{name}_{k}"] = float(per[i])
    count = qtype.sum((0, 1))
    for i, k in enumerate(ve.TYPES):
        out[f"n_{k}"] = float(count[i])
    return out


def _margin_A(logits_at_q: torch.Tensor, meta: dict) -> torch.Tensor:
    """``z[v1] - z[v0]`` at every A query, flattened."""
    is_a = meta["query_type"] == ve.A
    z1 = logits_at_q.gather(-1, meta["query_tok"][..., None]).squeeze(-1)
    z0 = logits_at_q.gather(-1, meta["query_old"].clamp_min(0)[..., None]).squeeze(-1)
    return (z1 - z0)[is_a]


def _match_index(loc_q: torch.Tensor, loc_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each query location, the index of the write with the same location token."""
    eq = loc_q[:, :, None] == loc_w[:, None, :]
    return eq.float().argmax(-1), eq.any(-1)


def stream(model: tt.TTTModel, lay: ve.Layout, vst: ve.VeSettings, cell: dict, tok: torch.Tensor,
           meta: dict, cond: str, beta: float, gen: torch.Generator) -> dict:
    """One evaluation batch under ``cond``; the base condition adds the readout and key match."""
    idx, y = tok[:, :-1], tok[:, 1:]
    rule = ve.RULES[cell["rule"]]
    sel = ve.event_selector(lay, meta, len(idx)) if cell["rule"] == "linev" else None
    with torch.no_grad():
        h = model.backbone(idx, edge_mask=ve.edge_mask_for(cell, lay, idx.device))
    etas = [e.detach() for e in model.etas(float(cell["eta"]))]
    n_chunk = lay.T // lay.chunk
    first_q = int(meta["query_pos"].min()) // lay.chunk
    captured: dict[tuple[str, int], list[torch.Tensor]] = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        if c >= first_q:
            captured[("W", c)] = [w.detach() for w in W]
        if c in (0, c_chunk(lay)):
            captured[("dW", c)] = [(-eta * g).detach() for eta, g in zip(etas, grads)]

    logits = tt.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, tt.GradStats(), sel,
                                h=h, rule=rule,
                                inner_transform=injection(cond, lay, meta, y, beta, gen),
                                observer=observer if cond == "base" else None)
    zq = ve._gather_last(logits, meta["query_pos"])
    out: dict[str, Any] = dict(q=_queries(zq, meta), margin=_margin_A(zq, meta).tolist())
    if cond != "base":
        return out

    with torch.no_grad():
        x = model.fast_input(h, idx)
        lo = first_q * lay.chunk
        variants = {"full": None, "minus_E": 0, "minus_C": c_chunk(lay)}
        rebuilt = {}
        for name, k in variants.items():
            parts = []
            for c in range(first_q, n_chunk):
                sl = slice(c * lay.chunk, (c + 1) * lay.chunk)
                W = captured[("W", c)]
                if k is not None:
                    W = [w - d for w, d in zip(W, captured[("dW", k)])]
                parts.append(model.read(tt.fast_block(h[:, sl], model.head.ln, W, x[:, sl])))
            rebuilt[name] = ve._gather_last(torch.cat(parts, 1), meta["query_pos"] - lo)
        out["rebuild_maxdiff"] = float((rebuilt["full"] - zq).abs().max())
        for name in ("minus_E", "minus_C"):
            out[f"q_{name}"] = _queries(rebuilt[name], meta)
            out[f"margin_{name}"] = _margin_A(rebuilt[name], meta).tolist()

        # key match under W0: <a_R, a_E>, <a_R, a_C> for the query's own location
        act = F.gelu(x @ model.head.W1.T + model.head.b1)
        loc_q = idx.gather(1, meta["query_pos"])
        a_q = ve._gather_last(act, meta["query_pos"])
        km = {}
        for name, pos in (("E", meta["e_write"]), ("C", meta["c_write"])):
            j, found = _match_index(loc_q, idx.gather(1, pos))
            a_w = ve._gather_last(act, pos.gather(1, j))
            dot = (a_q * a_w).sum(-1)
            cos = F.cosine_similarity(a_q, a_w, dim=-1)
            km[name] = (dot, cos, found)
        for t_i, t in enumerate(ve.TYPES):
            is_t = meta["query_type"] == t_i
            for name, (dot, cos, found) in km.items():
                m = is_t & found
                out[f"dot_{name}_{t}"] = dot[m].tolist()
                out[f"cos_{name}_{t}"] = cos[m].tolist()
    return out


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def cell_summary(parts: dict[str, list[dict]]) -> dict:
    """Pool the batches of each condition into rates and medians."""
    out: dict[str, Any] = {}

    def rates(qs: list[dict], prefix: str) -> None:
        tot = {k: sum(q[k] for q in qs) for k in qs[0]}
        for t in ve.TYPES:
            n = tot[f"n_{t}"] or 1.0
            out[f"{prefix}acc_{t}"] = tot[f"hit_{t}"] / n
            if t in ("A", "X"):
                out[f"{prefix}stale_{t}"] = tot[f"stale_{t}"] / n

    for cond, batches in parts.items():
        rates([b["q"] for b in batches], f"{cond}.")
        out[f"{cond}.margin_A"] = _median(sum((b["margin"] for b in batches), []))
    base = parts["base"]
    for name in ("minus_E", "minus_C"):
        rates([b[f"q_{name}"] for b in base], f"{name}.")
        full = sum((b["margin"] for b in base), [])
        abl = sum((b[f"margin_{name}"] for b in base), [])
        out[f"contrib_{name[-1]}_A"] = _median([f - a for f, a in zip(full, abl)])
    out["rebuild_maxdiff"] = max(b["rebuild_maxdiff"] for b in base)
    for t in ve.TYPES:
        for name in ("E", "C"):
            dots = sum((b[f"dot_{name}_{t}"] for b in base), [])
            coss = sum((b[f"cos_{name}_{t}"] for b in base), [])
            out[f"dot_{name}_{t}"] = _median(dots)
            out[f"cos_{name}_{t}"] = _median(coss)
        de = sum((b[f"dot_E_{t}"] for b in base), [])
        dc = sum((b[f"dot_C_{t}"] for b in base), [])
        if de and dc and len(de) == len(dc):
            out[f"dot_ratio_CE_{t}"] = _median([c / e for c, e in zip(dc, de)])
    return out


def run_cell(cell: dict, st: DiagSettings, vst: ve.VeSettings, lay: ve.Layout,
             dev: torch.device) -> dict:
    t0 = time.time()
    model = load_model(vst, lay, cell, st.ckpt_dir, dev)
    data = ve.fixed_set(lay, st.n_eval, ve.EVAL_SEED, dev, st.eval_batch)
    parts: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        gen = torch.Generator().manual_seed(RAND_SEED + int(cell["seed"]))
        parts[cond] = [stream(model, lay, vst, cell, tok, meta, cond, st.beta, gen)
                       for tok, meta in data]
    rec = dict(cell_id=ve.cell_id(cell), **cell, **cell_summary(parts),
               wall_s=round(time.time() - t0, 1))
    return rec


# --------------------------------------------------------------------------- scoring


def count_verdict(n: int, k: int) -> str:
    """Wave-1 count rule: S if n >= ceil(5k/6), F if n <= floor(k/2), else I; NE if k < 4."""
    if k < 4:
        return "NE"
    if n >= math.ceil(5 * k / 6):
        return "S"
    if n <= k // 2:
        return "F"
    return "I"


def load_reference(path: str) -> dict[str, dict]:
    if not path:
        return {}
    with open(path) as fh:
        return {r["cell_id"]: r for r in csv.DictReader(fh)}


def gate_g0(rec: dict, ref: dict[str, dict]) -> tuple[bool | None, float | None]:
    """Base accuracies reproduce the wave-1 finals within ``G0_TOL`` for every type."""
    r = ref.get(rec["cell_id"])
    if r is None:
        return None, None
    diff = max(abs(rec[f"base.acc_{t}"] - float(r[f"acc_{t}"])) for t in ve.TYPES)
    return diff <= G0_TOL, diff


def score(recs: list[dict]) -> dict:
    """Registered D1-D3 for the lesion arm, per eta; cells need G0 and V3 (U, S >= .90)."""
    out = {}
    for eta in sorted({float(r["eta"]) for r in recs}):
        les = [r for r in recs if r["rule"] == "ce" and r.get("edge") and float(r["eta"]) == eta
               and r.get("g0") is not False
               and r["base.acc_U"] >= .90 and r["base.acc_S"] >= .90]
        k = len(les)
        d1 = sum(r["minus_E.acc_A"] >= .90 for r in les)
        d2 = sum(r["erase_v0.acc_A"] >= .90 and all(r[f"erase_v0.acc_{t}"] >= .90
                                                     for t in ("U", "S", "X", "N")) for r in les)
        ctrl = sum(r["erase_rand.acc_A"] >= .90 for r in les)
        d3 = sum((r["contrib_C_A"] or 0) > 0 and (r["contrib_E_A"] or 0) < -(r["contrib_C_A"] or 0)
                 for r in les)
        tag = f"eta{eta:g}"
        out[f"D1_{tag}"] = dict(verdict=count_verdict(d1, k), n=d1, k=k)
        out[f"D2_{tag}"] = dict(verdict=count_verdict(d2, k), n=d2, k=k)
        # control: S when the random-value erase does NOT rescue
        out[f"D2c_{tag}"] = dict(verdict=count_verdict(k - ctrl, k), rescued=ctrl, k=k)
        out[f"D3_{tag}"] = dict(verdict=count_verdict(d3, k), n=d3, k=k)
    return out


def table(recs: list[dict]) -> str:
    cols = ["base.acc_A", "minus_E.acc_A", "minus_C.acc_A", "erase_v0.acc_A", "erase_rand.acc_A",
            "add_v1.acc_A", "base.acc_N", "erase_v0.acc_X", "contrib_E_A", "contrib_C_A",
            "dot_ratio_CE_A", "dot_ratio_CE_X"]
    lines = ["cell".ljust(28) + " g0  " + " ".join(c.replace(".acc", "").ljust(10)[:10] for c in cols)]
    for r in sorted(recs, key=lambda r: r["cell_id"]):
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(("   -" if v is None else f"{v:7.3f}").ljust(10))
        g0 = {True: "ok ", False: "BAD", None: " ? "}[r.get("g0")]
        lines.append(r["cell_id"].ljust(28) + f" {g0} " + " ".join(vals))
    return "\n".join(lines)


def run(cfg: dict, seed: int) -> dict:
    st = settings(cfg)
    tt._mark_process(st.process_name)
    torch.set_num_threads(max(1, int(st.threads)))
    dev = tt._device(st.device)
    tt.configure_tf32(False, st.process_name)
    vst = ve.VeSettings(**{**st.ve, "n_eval": st.n_eval, "eval_batch": st.eval_batch,
                           "device": st.device})
    lay = vst.layout()
    for cell in st.cells:
        ve.check_cell(cell)
    if not st.cells:
        raise ValueError("no cells")
    ref = load_reference(st.reference_csv)
    print("VE_DIAG_CONFIG " + json.dumps(dict(dataclasses.asdict(st), cells=len(st.cells),
                                             device_resolved=str(dev))), flush=True)
    recs = []
    for i, cell in enumerate(st.cells):
        rec = run_cell(cell, st, vst, lay, dev)
        rec["g0"], rec["g0_maxdiff"] = gate_g0(rec, ref)
        recs.append(rec)
        print("CELL_DIAG " + json.dumps(rec), flush=True)
        print(f"  [{i + 1}/{len(st.cells)}] {rec['cell_id']} g0={rec['g0']} "
              f"A base/-E/erase/rand/add = {rec['base.acc_A']:.3f}/{rec['minus_E.acc_A']:.3f}/"
              f"{rec['erase_v0.acc_A']:.3f}/{rec['erase_rand.acc_A']:.3f}/{rec['add_v1.acc_A']:.3f} "
              f"{rec['wall_s']}s", flush=True)
    print(table(recs), flush=True)
    bad = [r["cell_id"] for r in recs if r["g0"] is False]
    print(f"G0 reproduction: {len(recs) - len(bad)}/{len(recs)} within {G0_TOL}; "
          f"failing: {bad or 'none'}", flush=True)
    res = score(recs)
    for key, val in res.items():
        print(f"{key}: {json.dumps(val)}", flush=True)
    print("VE_DIAG_SUMMARY " + json.dumps(dict(cells=len(recs), g0_fail=bad, scores=res)),
          flush=True)
    return dict(cells=len(recs), g0_fail=len(bad), scores=res)
