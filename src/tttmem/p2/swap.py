"""P2-2: where FO-CE loses the use of written memory — writer/reader swap.

Eval-only on the P2-1 checkpoints of SO (``e2e2``, tag ``s``) and FO-CE (``e2e1-ce``, tag ``f``)
at one training step.

Writer ``y`` runs S (chunks 0..3 in the barrier layout, no writes in P) and returns the fast
weights entering P, ``W4_y[k]``, for the suppression ``k``: ``none``, ``F`` (focal sources) or
``NF`` (count-matched non-focal sources). Reader ``r`` (its slow weights) reads P alone (chunks
4..7, one attention segment, no writes) starting from ``W0_x + (W4_y[k] - W0_y)`` (an additive
delta transplant). Attention resets at chunk 4, so in frozen-P evaluation P sees S only through W4,
and the cell ``r = x = y`` reproduces the saved ``frozenP`` / ``fP_focal`` / ``fP_nonfocal`` probe
NLL (``y = 0`` reproduces ``noadapt``).

Cells ``r{R}x{X}y{Y}_{k}``: R, X, Y in {s, f}; Y = 0 means no written delta. Per-layer cells
``r{R}x{R}y{R}+{j}_{k}`` take fast layer j's delta from the other writer. Crossed-start cells
``r{R}x{R}y{Y}c_{k}`` (Y != R): writer Y's weights write S starting from the reader's W0.

Per-document quantities (L = probe-mean NLL in P), all oriented as an FO-CE deficit:
- M = L_F - L_NF (focal memory), G = L_y0 - L_none (gain from everything written in S),
  L' = -L_none (so that its total is D = L_ff - L_ss).
- Primary 2-player split (reader = R and X together, writer = Y) of total = Q_ss - Q_ff:
  write = ((Q_ss - Q_sf) + (Q_fs - Q_ff)) / 2, read = ((Q_ss - Q_fs) + (Q_sf - Q_ff)) / 2,
  write + read = total exactly; interaction = (Q_ss - Q_sf) - (Q_fs - Q_ff).
- Owen values with the unions {R, X} and {Y}: delta = write, slow + w0 = read (exactly).
- Crossed-start sensitivity: write_c with Q_sf, Q_fs replaced by the crossed-start cells.
- Per-layer transplant sensitivity: ((Q_ss - Q_ss<-f_j) + (Q_ff<-s_j - Q_ff)) / 2 (not additive).
Intervals: URL-domain cluster bootstrap over documents (10,000 resamples, seed 0); ``lo`` / ``hi``
are one-sided 95 % bounds; ``dominance95`` is the two-sided 95 % interval of write - read.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .meta import InnerCfg, stream_eval
from .probe import BARRIERS, PROBE_START, STRATA, boot_mean, bootstrap_weights, boundary_valid, \
    find_probes, one_sided

TAGS = ("s", "f")
ARMS = {"s": "e2e2", "f": "e2e1-ce"}
OTHER = {"s": "f", "f": "s"}
SUPP = ("none", "F", "NF")
REF_MODES = ("frozenP", "fP_focal", "fP_nonfocal", "noadapt")
PRIMARY_REQUIRED = REF_MODES
TRAJECTORY_REQUIRED = ("frozenP", "noadapt")
GEO_KEYS = ("dw_norm_s", "dw_norm_f", "dw_cos", "phi_norm_s", "phi_norm_f", "phi_cos")
KINDS = ("M", "G", "L")


def diag_cell(a: str, mode: str) -> str:
    return {"frozenP": f"r{a}x{a}y{a}_none", "fP_focal": f"r{a}x{a}y{a}_F",
            "fP_nonfocal": f"r{a}x{a}y{a}_NF", "noadapt": f"r{a}x{a}y0_none"}[mode]


@dataclass
class SwapCfg:
    so_dir: str = "outputs/p21/e2e2/attempt1"
    fo_dir: str = "outputs/p21/e2e1-ce/attempt1"
    data_dir: str = "data/p2data"
    out_dir: str = "outputs/p22/swap"
    preset: str = "125m"
    dtype: str = "bf16"
    device: str = "cuda"
    seq_len: int = 8192
    primary_step: int = 400
    primary_docs: int = 1024          # = the endpoint N frozen by P2-1 sizing (decisions/sizing.json)
    trajectory_steps: list = field(default_factory=lambda: [30, 60, 100, 200])
    trajectory_docs: int = 256        # = P2-1 trajectory documents (the first 256)
    ref_docs: int = 256               # same-GPU reference and identical-arm control (first docs)
    crossed: bool = True              # crossed-start cells at the primary step
    batch: int = 4                    # = P2-1 eval_batch
    fused_attn: bool = True
    inner_lr: float = 1.0
    inner_clip: float = 1.0
    n_boot: int = 10000
    seed: int = 0
    alpha: float = 0.05
    tol: dict = field(default_factory=dict)  # overrides of ``Tol`` fields (none registered)

    @classmethod
    def from_dict(cls, d: dict) -> "SwapCfg":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"mode", "process_name"}
        if unknown:
            raise ValueError(f"unknown swap keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})


# ------------------------------------------------------------------------------ cells
class Cell(NamedTuple):
    name: str
    R: str
    X: str
    Y: str
    k: str
    layer: int | None = None
    crossed: bool = False


def cells(n_fast: int, crossed: bool = False) -> list[Cell]:
    """Every reader pass (``layer``: the swapped fast layer; ``crossed``: crossed-start write)."""
    out = []
    for r in TAGS:
        for x in TAGS:
            out.append(Cell(f"r{r}x{x}y0_none", r, x, "0", "none"))
            for y in TAGS:
                out += [Cell(f"r{r}x{x}y{y}_{k}", r, x, y, k) for k in SUPP]
        for j in range(n_fast):
            out += [Cell(f"r{r}x{r}y{r}+{j}_{k}", r, r, r, k, j) for k in SUPP]
        if crossed:
            o = OTHER[r]
            out += [Cell(f"r{r}x{r}y{o}c_{k}", r, r, o, k, None, True) for k in SUPP]
    return out


def fast_for(cell: Cell, W0b: dict, W4: dict, W4c: dict | None = None):
    """Initial fast weights of one reader pass (None = the reader's own W_0)."""
    if cell.crossed:
        return W4c[(cell.Y, cell.R)][cell.k]
    if cell.Y == "0":
        return None if cell.X == cell.R else [w.clone() for w in W0b[cell.X]]
    y, k = cell.Y, cell.k
    if cell.layer is None:
        if cell.X == y:
            return W4[y][k]
        return [a + (b - c) for a, b, c in zip(W0b[cell.X], W4[y][k], W0b[y])]
    o = OTHER[y]
    out = list(W4[y][k])
    for i in range(3 * cell.layer, 3 * cell.layer + 3):
        out[i] = W0b[y][i] + (W4[o][k][i] - W0b[o][i])
    return out


# ---------------------------------------------------------------------- GPU passes
@torch.no_grad()
def write_states(model, ids, tgt, bmask, focal, nonfocal, inner: InnerCfg, *,
                 fused_attn: bool = True, fast_init=None) -> dict:
    """Fast weights entering P after the S writes (P frozen), per suppression ``k``.
    ``fast_init`` starts the writes from other initial fast weights (crossed start)."""
    C = model.cfg.chunk
    T5 = (PROBE_START + 1) * C  # chunk PROBE_START is the last chunk, so chunk 3 still writes
    ids5, tgt5, m5 = ids[:, :T5], tgt[:, :T5], bmask[:, :T5]
    s_only = torch.zeros_like(m5)
    s_only[:, :PROBE_START * C] = True
    sup = {"none": s_only, "F": s_only & ~focal[:, :T5], "NF": s_only & ~nonfocal[:, :T5]}
    cache, out = {}, {}
    for k in SUPP:
        r = stream_eval(model, ids5, tgt5, m5, inner, barriers=list(BARRIERS), inner_mask=sup[k],
                        fused_attn=fused_attn, return_fast=True, prefix_cache=cache,
                        nll_positions=torch.zeros_like(m5), fast_init=fast_init)
        out[k] = r["fast"]
    return out


@torch.no_grad()
def read_P(model, ids, tgt, bmask, pmask, inner: InnerCfg, fast, cache: dict, *,
           fused_attn: bool = True) -> torch.Tensor:
    """Per-token NLL over P ([B, T - PROBE_START * C]) read from ``fast``, no writes."""
    P = slice(PROBE_START * model.cfg.chunk, None)
    r = stream_eval(model, ids[:, P], tgt[:, P], bmask[:, P], inner, adapt=False,
                    fused_attn=fused_attn, prefix_cache=cache, nll_positions=pmask[:, P],
                    fast_init=fast)
    return r["nll"]


def _layer_flat(ts: list[torch.Tensor], j: int) -> torch.Tensor:
    return torch.cat([t.flatten(1) for t in ts[3 * j: 3 * j + 3]], 1).double()


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30)


def swap_eval(models: dict, docs: np.ndarray, probes: list, *, crossed: bool = False,
              batch: int = 4, device="cuda", fused_attn: bool = True, inner_lr: float = 1.0,
              inner_clip: float = 1.0, log=None) -> dict:
    """Per-cell per-probe NLL (``tok``), per-document probe means (``doc``) and per-document,
    per-fast-layer delta geometry (``geo``) for both arms at one step."""
    cfg = models["s"].cfg
    C, bos, S = cfg.chunk, cfg.bos_id, cfg.suffix_len
    inner = InnerCfg(rule="ce", lr=inner_lr, clip=inner_clip)
    N, T1 = docs.shape
    T = T1 - 1
    off = PROBE_START * C
    bvalid = torch.from_numpy(boundary_valid(T, C)).to(device)
    cl = cells(S, crossed)
    tok = {c.name: [None] * N for c in cl}
    doc = {c.name: np.full(N, np.nan) for c in cl}
    geo = {key: np.full((N, S), np.nan) for key in GEO_KEYS}
    W0 = {a: [p.detach() for p in models[a].prime_params()] for a in TAGS}
    cuda = torch.device(device).type == "cuda"
    t0 = time.time()
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        arr = torch.from_numpy(docs[sl].astype(np.int64)).to(device, non_blocking=True)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        bmask = (tgt != bos) & bvalid
        B = ids.shape[0]
        pmask = torch.zeros(B, T, dtype=torch.bool, device=device)
        focal = torch.zeros_like(pmask)
        nonfocal = torch.zeros_like(pmask)
        for i, p in enumerate(probes[sl]):
            pmask[i, torch.from_numpy(p.pos).to(device)] = True
            focal[i] = torch.from_numpy(p.focal).to(device)
            nonfocal[i] = torch.from_numpy(p.nonfocal).to(device)
        kw = dict(fused_attn=fused_attn)
        W0b = {a: [w.unsqueeze(0).expand(B, *w.shape) for w in W0[a]] for a in TAGS}
        W4 = {a: write_states(models[a], ids, tgt, bmask, focal, nonfocal, inner, **kw)
              for a in TAGS}
        W4c = {(OTHER[r], r): write_states(models[OTHER[r]], ids, tgt, bmask, focal, nonfocal,
                                           inner, fast_init=W0b[r], **kw)
               for r in TAGS} if crossed else None
        caches = {a: {} for a in TAGS}
        for cell in cl:
            nll = read_P(models[cell.R], ids, tgt, bmask, pmask, inner,
                         fast_for(cell, W0b, W4, W4c), caches[cell.R], **kw)
            nll = nll.cpu().numpy()
            for i, p in enumerate(probes[sl]):
                v = nll[i, p.pos - off]
                tok[cell.name][b0 + i] = v.astype(np.float32)
                doc[cell.name][b0 + i] = float(v.mean())
        for j in range(S):
            d = {a: _layer_flat([w - w0 for w, w0 in zip(W4[a]["none"], W0b[a])], j) for a in TAGS}
            phi = {a: _layer_flat([n - f for n, f in zip(W4[a]["NF"], W4[a]["F"])], j) for a in TAGS}
            for name, v in (("dw_norm_s", d["s"].norm(dim=1)), ("dw_norm_f", d["f"].norm(dim=1)),
                            ("dw_cos", _cos(d["s"], d["f"])),
                            ("phi_norm_s", phi["s"].norm(dim=1)),
                            ("phi_norm_f", phi["f"].norm(dim=1)),
                            ("phi_cos", _cos(phi["s"], phi["f"]))):
                geo[name][sl, j] = v.cpu().numpy()
        del W4, W4c, W0b, caches
        if log is not None and ((b0 // batch) % 16 == 0 or sl.stop == N):
            rec = {"docs": sl.stop, "of": N, "elapsed_s": round(time.time() - t0, 1)}
            if cuda:
                rec.update(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            log("P22_PROGRESS", rec)
    return {"tok": tok, "doc": doc, "geo": geo, "cells": [c.name for c in cl]}


# ---------------------------------------------------------------------- analysis (NumPy)
def quantity(doc: dict, kind: str, R: str, X: str, Y: str, layer: int | None = None,
             crossed: bool = False) -> np.ndarray:
    """Per-document M, G or L' (FO-deficit orientation) of one cell group."""
    if crossed and Y not in (R, "0"):
        base = f"r{R}x{R}y{Y}c"
    else:
        base = f"r{R}x{X}y{Y}" + ("" if layer is None else f"+{layer}")
    if kind == "M":
        return doc[base + "_F"] - doc[base + "_NF"]
    if kind == "G":
        return doc[f"r{R}x{X}y0_none"] - doc[base + "_none"]
    if kind == "L":
        return -doc[base + "_none"]
    raise ValueError(kind)


def owen(Q) -> dict:
    """Owen values of Q(s, s, s) - Q(f, f, f) with the unions {R, X} (reader) and {Y} (writer).
    ``delta`` equals the 2-player write; ``slow + w0`` equals the 2-player read."""
    def v(S):
        return Q(*("s" if p in S else "f" for p in ("R", "X", "Y")))
    phi = {}
    for i, j in (("R", "X"), ("X", "R")):
        phi[i] = sum(v(H | S | {i}) - v(H | S)
                     for H in (frozenset(), frozenset("Y")) for S in (frozenset(), frozenset(j))) / 4
    phi["Y"] = ((v({"Y"}) - v(set())) + (v({"R", "X", "Y"}) - v({"R", "X"}))) / 2
    return {"slow": phi["R"], "w0": phi["X"], "delta": phi["Y"]}


def components(doc: dict, kind: str, n_fast: int, crossed: bool = False) -> dict:
    """Per-document vectors of every registered and descriptive component of ``kind``."""
    Q = lambda r, y: quantity(doc, kind, r, r, y)  # noqa: E731
    ss, sf, fs, ff = Q("s", "s"), Q("s", "f"), Q("f", "s"), Q("f", "f")
    out = {"total": ss - ff,
           "write": ((ss - sf) + (fs - ff)) / 2,
           "read": ((ss - fs) + (sf - ff)) / 2,
           "interaction": (ss - sf) - (fs - ff),
           "write_at_reader_s": ss - sf, "write_at_reader_f": fs - ff,
           "read_with_writer_s": ss - fs, "read_with_writer_f": sf - ff}
    out["write_minus_read"] = out["write"] - out["read"]
    for p, v in owen(lambda R, X, Y: quantity(doc, kind, R, X, Y)).items():
        out[f"owen_{p}"] = v
    for j in range(n_fast):
        out[f"layer{j}_transplant"] = ((ss - quantity(doc, kind, "s", "s", "s", j))
                                       + (quantity(doc, kind, "f", "f", "f", j) - ff)) / 2
    if crossed:
        csf = quantity(doc, kind, "s", "s", "f", crossed=True)
        cfs = quantity(doc, kind, "f", "f", "s", crossed=True)
        out["write_c"] = ((ss - csf) + (cfs - ff)) / 2
        out["read_c"] = out["total"] - out["write_c"]
        out["write_c_minus_read_c"] = out["write_c"] - out["read_c"]
        out["write_c_at_reader_s"], out["write_c_at_reader_f"] = ss - csf, cfs - ff
    return out


def classify(total: dict, write: dict, read: dict, dominance: dict) -> str:
    """Registered rule: one-sided 95 % bounds for the deficit and the
    components, the two-sided 95 % interval for write - read."""
    if not total["lo"] > 0:
        return "no-deficit"
    if write["lo"] > 0 and dominance["lo"] > 0:
        return "write-dominant"
    if read["lo"] > 0 and dominance["hi"] < 0:
        return "read-dominant"
    if write["lo"] > 0 and read["lo"] > 0:
        return "both"
    return "unresolved"


def _two_sided(draws: np.ndarray, alpha: float) -> dict:
    q = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2]) if np.isfinite(draws).any() else [np.nan] * 2
    return {"lo": float(q[0]), "hi": float(q[1])}


def _share(num: np.ndarray, den: np.ndarray, est_num: float, total: dict, alpha: float) -> dict:
    """Write share; undefined unless the deficit's one-sided lower bound is positive."""
    if not total["lo"] > 0:
        return {"est": None, "lo": None, "hi": None, "note": "total interval includes 0"}
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    q = np.nanquantile(r, [alpha, 1 - alpha])
    return {"est": est_num / total["est"], "lo": float(q[0]), "hi": float(q[1])}


def decompose(doc: dict, clusters, n_fast: int, *, n_boot=10000, seed=0, alpha=0.05,
              crossed: bool = False, kinds=KINDS) -> dict:
    w = bootstrap_weights(np.asarray(clusters), n_boot, seed)
    res = {}
    for kind in kinds:
        comp = components(doc, kind, n_fast, crossed)
        st, draws = {}, {}
        for name, v in comp.items():
            point, dr = boot_mean(v, w)
            st[name] = one_sided(point, dr, alpha)
            draws[name] = dr
        st["dominance95"] = _two_sided(draws["write_minus_read"], alpha)
        st["write_share"] = _share(draws["write"], draws["total"], st["write"]["est"],
                                   st["total"], alpha)
        st["class"] = classify(st["total"], st["write"], st["read"], st["dominance95"])
        ws, wf = st["write_at_reader_s"], st["write_at_reader_f"]
        I = st["interaction"]
        st["flags"] = {"total_pos": st["total"]["lo"] > 0, "write_pos": st["write"]["lo"] > 0,
                       "read_pos": st["read"]["lo"] > 0,
                       "sign_reversal": bool((ws["lo"] > 0 and wf["hi"] < 0)
                                             or (ws["hi"] < 0 and wf["lo"] > 0)),
                       "non_additive": bool(abs(I["est"]) > abs(st["total"]["est"])
                                            and (I["lo"] > 0 or I["hi"] < 0))}
        if crossed:
            st["dominance95_c"] = _two_sided(draws["write_c_minus_read_c"], alpha)
            st["class_c"] = classify(st["total"], st["write_c"], st["read_c"], st["dominance95_c"])
            cs, cf = st["write_c_at_reader_s"], st["write_c_at_reader_f"]
            st["flags_c"] = {"write_pos": st["write_c"]["lo"] > 0, "read_pos": st["read_c"]["lo"] > 0,
                             "sign_reversal": bool((cs["lo"] > 0 and cf["hi"] < 0)
                                                   or (cs["hi"] < 0 and cf["lo"] > 0))}
            st["crossed_agrees"] = st["class_c"] == st["class"]
        res[kind] = st
    res["n_docs"] = len(clusters)
    res["n_clusters"] = int(len(np.unique(clusters)))
    return res


def stratum_docs(tok: dict, probes: list, stratum: int):
    """Per-document cell means over the probes of one stratum, for documents that have one."""
    idx = np.array([i for i, p in enumerate(probes) if (p.stratum == stratum).any()], dtype=int)
    doc = {c: np.array([float(tok[c][i][probes[i].stratum == stratum].mean()) for i in idx])
           for c in tok}
    return doc, idx


# --------------------------------------------------------------------- validity checks
@dataclass(frozen=True)
class Tol:
    """Validity tolerances. Tails are calibrated on two GPUs evaluating identical step-0
    weights: per-document |diff| max 0.027, per-probe q99
    0.063 and max 0.19 (writing modes; ``noadapt`` was bit-identical)."""
    floor: float = 2e-4         # per arm x mode: |mean| <= min(cap, max(floor, k_se * SE))
    k_se: float = 4.0
    cap: float = 1e-3
    mean_abs: float = 6e-3      # per arm x mode: mean |diff|
    doc_max: float = 0.06       # per arm x mode: max per-document |diff|
    probe_q99: float = 0.15     # per arm x mode: per-probe |diff| quantile 0.99
    probe_max: float = 0.6      # per arm x mode: max per-probe |diff|
    total_budget: float = 1e-3  # between-arm totals: |mean| <= min(total_cap, max(budget, k_se * SE))
    total_cap: float = 2e-3


def _close(d: np.ndarray, tol: Tol, *, total: bool = False, probe: np.ndarray | None = None) -> dict:
    """One validity comparison. Per-mode comparisons also bound the document and probe tails, so
    a few gross errors cannot pass by inflating the SE."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0 or not np.isfinite(d).all():
        return {"n": int(d.size), "ok": False, "note": "empty, non-finite or not 1-D"}
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    if total:
        tm = min(tol.total_cap, max(tol.total_budget, tol.k_se * se))
    else:
        tm = min(tol.cap, max(tol.floor, tol.k_se * se))
    out = {"n": int(len(d)), "mean": float(d.mean()), "se": se, "tol_mean": tm,
           "mean_abs": float(np.abs(d).mean()), "max_abs": float(np.abs(d).max())}
    ok = abs(out["mean"]) <= tm
    if not total:
        ok = ok and out["mean_abs"] < tol.mean_abs and out["max_abs"] <= tol.doc_max
    if probe is not None:
        p = np.abs(np.asarray(probe, dtype=np.float64))
        if len(p) == 0 or not np.isfinite(p).all():
            ok, out["note"] = False, "empty or non-finite probe differences"
        else:
            q = np.quantile(p, [0.5, 0.9, 0.99, 1.0])
            out["probe_abs_q50_q90_q99_max"] = [float(x) for x in q]
            ok = ok and q[2] <= tol.probe_q99 and q[3] <= tol.probe_max
    out["ok"] = bool(ok)
    return out


def _derived(get, a: str, kind: str) -> np.ndarray:
    if kind == "M":
        return get(a, "fP_focal") - get(a, "fP_nonfocal")
    if kind == "G":
        return get(a, "noadapt") - get(a, "frozenP")
    return -get(a, "frozenP")


DERIVED_MODES = {"M": ("fP_focal", "fP_nonfocal"), "G": ("noadapt", "frozenP"), "L": ("frozenP",)}


def _cell_mode(c: Cell) -> str:
    """Reference mode a cell must equal when both tags are the same weights."""
    return "noadapt" if c.Y == "0" else {"none": "frozenP", "F": "fP_focal", "NF": "fP_nonfocal"}[c.k]


def _probe_diff(tok: dict | None, ref_tok: dict | None, cell: str, n: int):
    if tok is None or ref_tok is None:
        return None
    t = np.concatenate(tok[cell][:n])
    if len(t) != len(ref_tok):
        return np.array([np.inf])
    return t - ref_tok


def reference_check(doc: dict, ref: dict, required, tol: Tol, *, tok: dict | None = None,
                    ref_tok: dict | None = None) -> dict:
    """Diagonal (reader = writer) cells against reference per-document means ``ref[a][mode]``
    (saved P2-1 evaluation, or the original ``eval_docs`` path on this GPU) and, when given,
    per-probe NLL ``ref_tok[a][mode]``. Every required mode must be present; between-arm totals
    of M, G and L' are checked against the bias budget."""
    out, missing = {}, [f"{a}_{m}" for a in TAGS for m in required if m not in ref[a]]
    for a in TAGS:
        for mode in REF_MODES:
            if mode not in ref[a]:
                continue
            n = len(ref[a][mode])
            cell = diag_cell(a, mode)
            rt = None if ref_tok is None else ref_tok.get(a, {}).get(mode)
            out[f"{a}_{mode}"] = _close(doc[cell][:n] - ref[a][mode], tol,
                                        probe=_probe_diff(tok, rt, cell, n) if rt is not None else None)
    for kind, modes in DERIVED_MODES.items():
        if all(m in ref[a] for a in TAGS for m in modes):
            n = min(len(ref[a][m]) for a in TAGS for m in modes)
            sw = lambda a, m: doc[diag_cell(a, m)][:n]  # noqa: E731
            rf = lambda a, m: ref[a][m][:n]  # noqa: E731
            d = (_derived(sw, "s", kind) - _derived(sw, "f", kind)) \
                - (_derived(rf, "s", kind) - _derived(rf, "f", kind))
            out[f"total_{kind}"] = _close(d, tol, total=True)
    out["missing"] = missing
    out["ok"] = not missing and all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def identical_arm_check(doc: dict, tok: dict, ref: dict, ref_tok: dict, n_fast: int, tol: Tol, *,
                        crossed: bool) -> dict:
    """Identical-arm control: with the same weights in both roles every cell (cross, W0-swapped,
    per-layer, crossed-start) must equal the original-path reference of its suppression mode,
    per document and per probe. Catches routing errors that a symmetric total would hide."""
    out = {}
    for c in cells(n_fast, crossed):
        mode = _cell_mode(c)
        n = len(ref[mode])
        out[c.name] = _close(doc[c.name][:n] - ref[mode], tol,
                             probe=_probe_diff(tok, ref_tok[mode], c.name, n))
    comp = components({k: v for k, v in doc.items()}, "M", n_fast)
    out["components_M"] = {c: float(np.mean(comp[c])) for c in ("total", "write", "read")}
    out["ok"] = all(v["ok"] for k, v in out.items() if k != "components_M")
    return out


def geometry_summary(geo: dict) -> dict:
    return {k: [float(np.nanmean(v[:, j])) for j in range(v.shape[1])] for k, v in geo.items()}


# --------------------------------------------------------------------------- driver
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _param_distance(a, b) -> dict:
    """Relative L2 distance of parameter groups between the two arms."""
    groups = {"prime": [], "suffix_slow": [], "prefix": [], "embed": [], "other": []}
    prime = {id(p) for p in a.prime_params()}
    suffix = {id(p) for p in a.slow_suffix_params()}
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        g = "prime" if id(pa) in prime else "embed" if "wte" in name else \
            "suffix_slow" if id(pa) in suffix else "prefix" if name.startswith("layers.") else "other"
        groups[g].append(((pa.detach().double() - pb.detach().double()).pow(2).sum().item(),
                          pa.detach().double().pow(2).sum().item()))
    return {g: float(np.sqrt(sum(d for d, _ in v) / max(sum(n for _, n in v), 1e-300)))
            for g, v in groups.items() if v}


def _finite(o):
    """JSON-safe copy: non-finite floats become None, NumPy scalars become Python ones."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _load_saved(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _check_offsets(off: np.ndarray, n_docs: int, n_items: int, name: str) -> None:
    off = np.asarray(off)
    if off.ndim != 1 or len(off) != n_docs + 1 or off[0] != 0 or off[-1] != n_items \
            or (np.diff(off) < 0).any():
        raise ValueError(f"malformed {name}")


def check_meta(meta: dict) -> None:
    n = len(meta["eval_rows"])
    for key in ("eval_rows", "eval_sha1", "eval_domain", "probe_pos", "focal_pos"):
        if meta[key].ndim != 1:
            raise ValueError(f"eval_meta {key} is not 1-D")
    if len(meta["eval_sha1"]) != n or len(meta["eval_domain"]) != n:
        raise ValueError("eval_meta identities are not aligned with eval_rows")
    _check_offsets(meta["probe_offset"], n, len(meta["probe_pos"]), "probe_offset")
    _check_offsets(meta["focal_offset"], n, len(meta["focal_pos"]), "focal_offset")


def preflight(cfg: SwapCfg, dirs: dict, meta: dict, root: Path, steps: list) -> dict:
    """Before any GPU work: inputs exist, checkpoints carry the expected arm and step, saved
    references are complete and aligned with ``eval_meta`` (rows, sha1, domain, lengths, probe
    counts), and no output of this run exists yet. Returns the saved per-step references."""
    check_meta(meta)
    saved = {}
    off = meta["probe_offset"]
    for step in steps:
        for ext in ("json", "npz"):
            if (root / f"swap_step{step:04d}.{ext}").exists():
                raise FileExistsError(f"output already exists: swap_step{step:04d}.{ext}")
        n = cfg.primary_docs if step == cfg.primary_step else cfg.trajectory_docs
        required = PRIMARY_REQUIRED if step == cfg.primary_step else TRAJECTORY_REQUIRED
        saved[step] = {}
        for a in TAGS:
            ck = dirs[a] / f"model_step{step:04d}.pt"
            if not ck.exists():
                raise FileNotFoundError(f"missing checkpoint {a} step {step}")
            payload = torch.load(ck, map_location="cpu", mmap=True, weights_only=True)
            if not isinstance(payload, dict) or payload.get("step") != step \
                    or payload.get("arm") != ARMS[a]:
                raise ValueError(f"checkpoint {ck} is not {ARMS[a]} step {step}")
            del payload
            z = _load_saved(dirs[a] / f"eval_step{step:04d}.npz")
            m = len(z["eval_sha1"])
            if m < n or any(not np.array_equal(z[k], meta[k][:m])
                            for k in ("eval_rows", "eval_sha1", "eval_domain")):
                raise ValueError(f"saved evaluation {a} step {step} is short or misaligned")
            ref, ref_tok = {}, {}
            for mode in REF_MODES:
                if f"doc_{mode}" not in z:
                    continue
                dm, tm = z[f"doc_{mode}"], z.get(f"tok_{mode}", np.zeros((0, 0)))
                if dm.ndim != 1 or tm.ndim != 1 or len(dm) != m or len(tm) != off[m]:
                    raise ValueError(f"saved {a} step {step} {mode}: inconsistent arrays")
                ref[mode] = dm[:n].astype(np.float64)
                ref_tok[mode] = tm[:off[n]].astype(np.float64)
            lack = [mode for mode in required if mode not in ref]
            if lack:
                raise ValueError(f"saved {a} step {step} lacks required modes {lack}")
            saved[step][a] = (ref, ref_tok)
    return saved


def _peak(cuda: bool) -> dict:
    if not cuda:
        return {}
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def run(cfg: SwapCfg) -> dict:
    from .data import open_split
    from .evaluate import eval_docs
    from .model import PRESETS, build_model, load_state

    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    def log(prefix, rec):
        line = prefix + " " + json.dumps(_finite(rec), sort_keys=True, allow_nan=False)
        with (root / "events.jsonl").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    tol = Tol(**cfg.tol)
    log("P22_CONFIG", {**asdict(cfg), "tolerances": asdict(tol)})
    dirs = {"s": Path(cfg.so_dir), "f": Path(cfg.fo_dir)}
    metas = {}
    for a in TAGS:
        with np.load(dirs[a] / "eval_meta.npz", allow_pickle=False) as z:
            metas[a] = {k: z[k] for k in z.files}
    for key in ("eval_rows", "eval_sha1", "eval_domain", "probe_offset", "probe_pos",
                "focal_offset", "focal_pos"):
        if not np.array_equal(metas["s"][key], metas["f"][key]):
            raise ValueError(f"arms disagree on evaluation metadata: {key}")
    meta = metas["s"]
    steps = [cfg.primary_step] + [s for s in cfg.trajectory_steps if s != cfg.primary_step]
    if cfg.ref_docs > min(cfg.primary_docs, cfg.trajectory_docs):
        raise ValueError("ref_docs must not exceed the documents of any step")
    saved = preflight(cfg, dirs, meta, root, steps)
    n_max = max(cfg.primary_docs, cfg.trajectory_docs)
    rows = meta["eval_rows"][:n_max]
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[cfg.dtype]
    excluded = np.load(Path(cfg.data_dir) / "excluded_vocab.npy", allow_pickle=False)
    arr, manifest = open_split(cfg.data_dir, "eval")
    for i, row in enumerate(rows):  # the current data rows are the documents P2-1 evaluated
        if manifest[row]["sha1"] != meta["eval_sha1"][i] or manifest[row]["domain"] != meta["eval_domain"][i]:
            raise ValueError(f"current eval manifest differs from eval_meta at document {i}")
    docs = np.asarray(arr[rows, :cfg.seq_len + 1])
    C, bos = PRESETS[cfg.preset].chunk, PRESETS[cfg.preset].bos_id
    probes = []
    po, fo = meta["probe_offset"], meta["focal_offset"]
    for i, (row, d) in enumerate(zip(rows, docs)):
        p = find_probes(d[:-1], d[1:], C, valid=d[1:] != bos, excluded=excluded, seed=int(row))
        if not (np.array_equal(p.pos, meta["probe_pos"][po[i]:po[i + 1]])
                and np.array_equal(np.flatnonzero(p.focal), meta["focal_pos"][fo[i]:fo[i + 1]])
                and p.nf_ok):
            raise ValueError(f"recomputed probes differ from the saved annotation (doc {i})")
        probes.append(p)
    log("P22_DATA", {"docs": len(rows), "probes": int(sum(len(p.pos) for p in probes)),
                     "clusters_primary": int(len(np.unique(meta["eval_domain"][:cfg.primary_docs])))})
    ekw = dict(batch=cfg.batch, device=cfg.device, fused_attn=cfg.fused_attn,
               inner_lr=cfg.inner_lr, inner_clip=cfg.inner_clip)
    cuda = torch.device(cfg.device).type == "cuda"

    results, invalid, status = {}, [], "valid"
    for step in steps:
        primary = step == cfg.primary_step
        crossed = cfg.crossed and primary
        n = cfg.primary_docs if primary else cfg.trajectory_docs
        t0 = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        def phase(name, t):
            log("P22_PHASE", {"step": step, "phase": name, "s": round(time.time() - t, 1), **_peak(cuda)})
            return time.time()

        models, shas = {}, {}
        for a in TAGS:
            path = dirs[a] / f"model_step{step:04d}.pt"
            shas[a] = _sha256(path)
            m = build_model(cfg.preset, compute_dtype=dt, device="cpu")
            if dt == torch.float64:
                m = m.double()
            load_state(m, str(path))
            models[a] = m.to(cfg.device).eval()
        log("P22_CKPT", {"step": step, "sha256": shas, "load_s": round(time.time() - t0, 1)})
        dist = _param_distance(models["s"], models["f"])
        t = time.time()
        r = swap_eval(models, docs[:n], probes[:n], crossed=crossed, log=log, **ekw)
        t = phase("swap", t)
        n_fast = r["geo"]["dw_cos"].shape[1]
        # V3: the original evaluation path on this GPU, first ref_docs documents
        m_ref = cfg.ref_docs
        same = {}
        for a in TAGS:
            e = eval_docs(models[a], docs[:m_ref], probes[:m_ref], "ce", modes=list(REF_MODES),
                          **ekw)
            same[a] = ({mode: np.asarray(e["doc"][mode], dtype=np.float64) for mode in REF_MODES},
                       {mode: np.concatenate(e["tok"][mode]).astype(np.float64) for mode in REF_MODES})
        t = phase("same_gpu_reference", t)
        # V4: identical-arm control (SO weights in both roles), every cell vs the SO reference
        ctrl = None
        if primary:
            rc = swap_eval({"s": models["s"], "f": models["s"]}, docs[:m_ref], probes[:m_ref],
                           crossed=crossed, **ekw)
            ctrl = identical_arm_check(rc["doc"], rc["tok"], same["s"][0], same["s"][1], n_fast,
                                       tol, crossed=crossed)
            t = phase("identical_arm_control", t)
        del models
        if cuda:
            torch.cuda.empty_cache()
        required = PRIMARY_REQUIRED if primary else TRAJECTORY_REQUIRED
        v2 = reference_check(r["doc"], {a: saved[step][a][0] for a in TAGS}, required, tol,
                             tok=r["tok"], ref_tok={a: saved[step][a][1] for a in TAGS})
        v3 = reference_check(r["doc"], {a: same[a][0] for a in TAGS}, REF_MODES, tol,
                             tok=r["tok"], ref_tok={a: same[a][1] for a in TAGS})
        valid = v2["ok"] and v3["ok"] and (ctrl is None or ctrl["ok"])
        validity = {"saved_reference": v2, "same_gpu_reference": v3, "identical_arm_control": ctrl,
                    "ok": valid}
        fails = [f"{name}.{k}" for name, v in (("saved", v2), ("same_gpu", v3), ("identical", ctrl))
                 if v is not None for k, x in v.items() if isinstance(x, dict) and "ok" in x
                 and not x["ok"]]
        log("P22_VALID", {"step": step, "ok": valid, "saved_reference": v2["ok"],
                          "same_gpu_reference": v3["ok"],
                          "identical_arm_control": None if ctrl is None else ctrl["ok"],
                          "missing": v2["missing"] + v3["missing"], "failed": fails[:20]})
        clusters = meta["eval_domain"][:n]
        probe_off = np.cumsum([0] + [len(p.pos) for p in probes[:n]])
        res = {"step": step, "n_docs": n, "status": "valid" if valid else "invalid",
               "checkpoint_sha256": shas, "validity": validity, "param_rel_dist": dist,
               "cell_means": {c: float(np.mean(r["doc"][c])) for c in r["cells"]},
               "geometry_mean_per_layer": geometry_summary(r["geo"]), "config": asdict(cfg),
               "tolerances": asdict(tol)}
        if valid:
            dkw = dict(n_boot=cfg.n_boot, seed=cfg.seed, alpha=cfg.alpha)
            dec = decompose(r["doc"], clusters, n_fast, crossed=crossed, **dkw)
            res["decomposition"] = dec
            if primary:
                m = cfg.trajectory_docs
                res["decomposition_first_trajectory_docs"] = decompose(
                    {k: v[:m] for k, v in r["doc"].items()}, clusters[:m], n_fast, **dkw)
                res["strata"] = {}
                for name in ("E2c", "E2b"):
                    sd, idx = stratum_docs(r["tok"], probes[:n], STRATA.index(name))
                    if len(idx) >= 20:
                        res["strata"][name] = decompose(sd, clusters[idx], n_fast, kinds=("M", "L"),
                                                        crossed=crossed, **dkw)
            for kind in KINDS:
                st = dec[kind]
                rec = {"step": step, "n_docs": n, "kind": kind,
                       "role": "confirmatory" if primary and kind == "M" else "descriptive",
                       "class": st["class"], "flags": st["flags"],
                       **{c: st[c] for c in ("total", "write", "read", "dominance95", "interaction",
                                             "write_at_reader_s", "write_at_reader_f",
                                             "read_with_writer_s", "read_with_writer_f",
                                             "owen_slow", "owen_w0", "write_share")}}
                if "class_c" in st:
                    rec.update({c: st[c] for c in ("class_c", "flags_c", "crossed_agrees", "write_c",
                                                   "read_c", "dominance95_c")})
                log("P22_DECOMP", rec)
        log("P22_GEOM", {"step": step, "mean_per_layer": res["geometry_mean_per_layer"],
                         "param_rel_dist": dist})
        res["eval_s"] = round(time.time() - t0, 1)
        with (root / f"swap_step{step:04d}.npz").open("xb") as f:
            np.savez_compressed(f, probe_offset=probe_off, eval_rows=rows[:n], eval_domain=clusters,
                                probe_stratum=np.concatenate([p.stratum for p in probes[:n]]),
                                **{f"doc_{c}": r["doc"][c] for c in r["cells"]},
                                **{f"tok_{c}": np.concatenate(r["tok"][c]) for c in r["cells"]},
                                **{f"geo_{k}": v for k, v in r["geo"].items()})
        with (root / f"swap_step{step:04d}.json").open("x") as f:
            json.dump(_finite(res), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        results[step] = res
        if not valid:  # registered: nothing is interpreted for this step
            invalid.append(step)
            log("P22_ABORT" if primary else "P22_STEP_INVALID",
                {"step": step, "reason": "validity_check_failed"})
            if primary:  # the confirmatory step failed: stop and investigate
                status = "invalid"
                break
    summary = {"status": status, "primary_step": cfg.primary_step, "steps_done": sorted(results),
               "steps_invalid": invalid}
    p = results[cfg.primary_step].get("decomposition")
    if status == "valid" and p is not None:
        M = p["M"]
        summary.update(class_M=M["class"], flags_M=M["flags"],
                       **{f"M_{c}": [M[c]["est"], M[c]["lo"], M[c]["hi"]]
                          for c in ("total", "write", "read")},
                       M_dominance95=[M["dominance95"]["lo"], M["dominance95"]["hi"]],
                       class_G_descriptive=p["G"]["class"], class_L_descriptive=p["L"]["class"])
        if "class_c" in M:
            summary.update(class_M_crossed=M["class_c"], flags_M_crossed=M["flags_c"],
                           attribution="robust to W0 compatibility" if M["crossed_agrees"]
                           else "W0-compatibility-dependent")
    log("P22_RESULT", summary)
    return summary


def run_config(d: dict) -> dict:
    return run(SwapCfg.from_dict(d))
