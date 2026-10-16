"""E1-VE: world-model echo. Does first-order write starvation transfer to agent-caused changes?

A sequence is one episode in a world of ``n_loc`` locations, tokenised as 4-token
events ``(marker, argument, location, observed value)``. Five location types:

* **E** (explore, chunk 0): ``O z L v0`` for U, S, A and X locations; ``z`` is a
  filler, so the observed value is new information.
* **C** (change, chunk 1, after the memory has stored E):
  ``S``: ``O z L v0``, a re-observation without change (most things persist);
  ``X``: ``O z L v1``, the location changed by itself: a surprising change;
  ``A``: ``P v1 L v1``, the agent PUTs ``v1`` at an explored location. The new
  value is copyable from the action one token back, the echo condition;
  ``N``: ``P v1 L v1`` at a location E never showed: a predictable first binding.
* **workbench** (chunk 2): ``P v B v`` on ``n_scratch_loc`` workbench tokens that are
  never queried. They make copying frequent, as in echo, so the copy path is
  learned early in every arm; without them FO arms had not learned it by step
  1500 in the design pilot.
* **gap**: fillers until no query can attend to E, C or the workbench start.
* **R** (revisit): ``Q L s`` for every location, ``s`` its current state.
  ``U`` locations were explored once and never seen again.

S events make "unchanged" the right default at a re-observation, so a surprising
change carries an erase signal under CE. Without them every C observation is a
change, the slow path learns to bet against the old value there, and FO-CE then
failed A and X alike (the design pilot).

The types separate two properties of an inner rule. Whole-model CE starves when a
cheaper path predicts the target (A, N). A prediction-independent linear loss
cannot erase an old association (A, X). AICE (CE of the memory-attributable logits
``z - z(W0)``) is error-correcting on the memory's own prediction, which the copy
path does not enter. Under FO-CE the registered failure is *stale reversion*:
A-queries answered with the pre-action value ``v0``.

``slip`` is the action-noise dose: with that probability a PUT's argument is
replaced by a different value while the observation and state stay ``v1``. This
is the echo task's copy noise, transferred.

``edge = true`` is the lesion control (echo arm a5, R4-4a): the attention edge
from every C location token to its argument (``t -> t-1``) is blocked in every
backbone layer, in training and evaluation. The events keep their format, but A
and N are no longer predictable at the write, so A and N against their lesioned
selves isolate predictability from the PUT format and from the C chunk. The
workbench is not masked, so the copy path is still learned.

The model, the chunked inner loop and the streaming evaluation are
``exp_ttt``'s (``TTTModel``, ``sequential_inner``, ``eval_sequential``, frozen
token code as in ``e2e1emb``), unchanged. This module adds the task, the cell
loop and the metrics. The hot loop is device-resident: batches are sampled on
the device, metrics are accumulated there, and fused AdamW skips non-finite
steps through ``found_inf`` without a host round trip. The host syncs once per
quick evaluation.
"""
from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as tt

OBS, PUT, QRY = 0, 1, 2
TYPES = ("U", "S", "A", "X", "N")
U, S, A, X, N = range(5)
C_TYPES = ("S", "A", "X", "N")
RULES = {"ce": "ce", "linev": "linkv", "aice": "aice"}
ORDERS = ("fo", "so")
TRAIN_SEED = 7_331
SLIP_SEED = 9_173
QUICK_SEED = 20_261_008
EVAL_SEED = 20_261_009
EMB_SEED_OFFSET = 7919  # exp_ttt.build_model's frozen-code seed offset


@dataclass(frozen=True)
class Layout:
    """Token ids and fixed phase positions shared by every sequence."""

    n_loc: int = 20
    n_val: int = 16
    n_fill: int = 16
    per_type: int = 4
    n_scratch_loc: int = 4
    n_scratch: int = 16
    q_space_max: int = 3
    T: int = 512
    chunk: int = 64
    window: int = 64
    n_layers: int = 4

    @property
    def loc0(self) -> int:
        return 3

    @property
    def sloc0(self) -> int:
        return 3 + self.n_loc

    @property
    def val0(self) -> int:
        return self.sloc0 + self.n_scratch_loc

    @property
    def fill0(self) -> int:
        return self.val0 + self.n_val

    @property
    def V(self) -> int:
        return self.fill0 + self.n_fill

    @property
    def n_e(self) -> int:
        """Explored locations: U, S, A, X."""
        return 4 * self.per_type

    @property
    def n_c(self) -> int:
        """Change-phase events: S, A, X, N."""
        return 4 * self.per_type

    @property
    def c_start(self) -> int:
        """C starts on the first chunk boundary after E."""
        return math.ceil(4 * self.n_e / self.chunk) * self.chunk

    @property
    def c_end(self) -> int:
        return self.c_start + 4 * self.n_c

    @property
    def s_start(self) -> int:
        """The workbench starts on the first chunk boundary after C."""
        return math.ceil(self.c_end / self.chunk) * self.chunk

    @property
    def receptive(self) -> int:
        """Tokens back the stacked sliding window can see, inclusive."""
        return self.n_layers * (self.window - 1) + 1

    @property
    def r_start(self) -> int:
        """First query: its location token sees nothing at or before ``c_end - 1``."""
        return max(self.c_end + self.receptive - 1, self.s_start + 4 * self.n_scratch)

    def check(self) -> None:
        if self.n_loc != 5 * self.per_type:
            raise ValueError("n_loc must be 5 * per_type (U, S, A, X, N)")
        if self.n_val < 3:
            raise ValueError("slip sampling needs at least three values")
        if 4 * self.n_e > self.chunk or 4 * self.n_c > self.chunk:
            raise ValueError("E and C must each fit in one chunk")
        if self.n_scratch and self.n_scratch_loc < 1:
            raise ValueError("workbench events need workbench locations")
        if self.T % self.chunk:
            raise ValueError("T must be a multiple of chunk")
        last = self.r_start + (self.n_loc - 1) * (3 + self.q_space_max) + 2
        if last > self.T:
            raise ValueError(f"T={self.T} too short: the last query value lands at {last}")
        if self.r_start + 1 - (self.receptive - 1) <= self.c_end - 1:
            raise ValueError("the first query can attend to C")


def slot_types(lay: Layout, device: torch.device) -> torch.Tensor:
    """Type of each permutation slot: U, S, A, X, N blocks of ``per_type``."""
    return torch.arange(5, device=device).repeat_interleave(lay.per_type)


def _other(base: torch.Tensor, gen: torch.Generator, n: int) -> torch.Tensor:
    """A value uniform over ``range(n)`` minus ``base``."""
    r = torch.randint(0, n - 1, base.shape, generator=gen, device=base.device)
    return r + (r >= base).long()


def _other2(a: torch.Tensor, b: torch.Tensor, gen: torch.Generator, n: int) -> torch.Tensor:
    """A value uniform over ``range(n)`` minus ``{a, b}``; only valid where ``a != b``."""
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    r = torch.randint(0, n - 2, a.shape, generator=gen, device=a.device)
    r = r + (r >= lo).long()
    return r + (r >= hi).long()


def ve_batch(lay: Layout, B: int, gen: torch.Generator, slip: float = 0.0,
             slip_gen: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """``B`` episodes of length ``T + 1`` on ``gen``'s device, plus metadata.

    The clean stream consumes ``gen`` identically for every ``slip``; slips draw
    from ``slip_gen`` only, so cells that differ in ``slip`` see the same episodes
    up to the replaced PUT arguments.
    """
    dev = gen.device
    K, M, P = lay.n_loc, lay.n_val, lay.per_type
    kind = slot_types(lay, dev)                                   # (K,)
    rnd = lambda *s: torch.rand(*s, generator=gen, device=dev)
    rint = lambda lo, hi, s: torch.randint(lo, hi, s, generator=gen, device=dev)
    perm = rnd(B, K).argsort(1)                                   # slot -> location
    v0 = rint(0, M, (B, K))
    v_shift = (v0 + rint(1, M, (B, K))) % M
    v_fresh = rint(0, M, (B, K))
    v1 = torch.where((kind == A) | (kind == X), v_shift, torch.where(kind == N, v_fresh, v0))
    e_slot = rnd(B, lay.n_e).argsort(1)                           # U, S, A, X slots
    c_slot = rnd(B, lay.n_c).argsort(1) + P                       # S, A, X, N slots
    q_slot = rnd(B, K).argsort(1)
    spacing = rint(1, lay.q_space_max + 1, (B, K - 1))
    tok = rint(0, lay.n_fill, (B, lay.T + 1)) + lay.fill0
    z_e = rint(0, lay.n_fill, (B, lay.n_e)) + lay.fill0
    z_c = rint(0, lay.n_fill, (B, lay.n_c)) + lay.fill0
    s_loc = rint(0, max(1, lay.n_scratch_loc), (B, lay.n_scratch)) + lay.sloc0
    s_val = rint(0, M, (B, lay.n_scratch))

    def put(pos: torch.Tensor, cols: list[torch.Tensor]) -> None:
        for j, values in enumerate(cols):
            tok.scatter_(1, pos + j, values)

    e_pos = 4 * torch.arange(lay.n_e, device=dev).expand(B, -1)
    put(e_pos, [torch.full_like(e_pos, OBS), z_e, lay.loc0 + perm.gather(1, e_slot),
                lay.val0 + v0.gather(1, e_slot)])

    c_kind = kind[c_slot]
    c_v0, c_v1 = v0.gather(1, c_slot), v1.gather(1, c_slot)
    is_put = (c_kind == A) | (c_kind == N)
    c_arg, s_arg = c_v1, s_val
    c_slipped = torch.zeros_like(is_put)
    s_slipped = torch.zeros(B, lay.n_scratch, dtype=torch.bool, device=dev)
    if slip > 0:
        if slip_gen is None:
            raise ValueError("slip needs its own generator")
        srand = lambda s: torch.rand(s, generator=slip_gen, device=dev) < slip
        c_slipped = srand(c_v1.shape) & is_put
        wrong = torch.where(c_kind == A, _other2(c_v1, c_v0, slip_gen, M),
                            _other(c_v1, slip_gen, M))
        c_arg = torch.where(c_slipped, wrong, c_v1)
        s_slipped = srand(s_val.shape)
        s_arg = torch.where(s_slipped, _other(s_val, slip_gen, M), s_val)
    c_pos = lay.c_start + 4 * torch.arange(lay.n_c, device=dev).expand(B, -1)
    put(c_pos, [torch.where(is_put, PUT, OBS), torch.where(is_put, lay.val0 + c_arg, z_c),
                lay.loc0 + perm.gather(1, c_slot), lay.val0 + c_v1])
    s_pos = lay.s_start + 4 * torch.arange(lay.n_scratch, device=dev).expand(B, -1)
    if lay.n_scratch:
        put(s_pos, [torch.full_like(s_pos, PUT), lay.val0 + s_arg, s_loc, lay.val0 + s_val])

    steps = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), 3 + spacing], 1)
    q_pos = lay.r_start + steps.cumsum(1)
    q_kind = kind[q_slot]
    state = v1.gather(1, q_slot)                                  # v1 == v0 for U and S
    put(q_pos, [torch.full_like(q_pos, QRY), lay.loc0 + perm.gather(1, q_slot), lay.val0 + state])
    changed = (q_kind == A) | (q_kind == X)
    q_old = torch.where(changed, lay.val0 + v0.gather(1, q_slot), torch.full_like(q_slot, -1))
    c_old = torch.where(c_kind == N, torch.full_like(c_v0, -1), lay.val0 + c_v0)
    meta = dict(query_pos=q_pos + 1, query_tok=lay.val0 + state, query_type=q_kind,
                query_old=q_old, e_write=e_pos + 2, c_write=c_pos + 2, c_type=c_kind,
                c_old=c_old, c_slipped=c_slipped, s_write=s_pos + 2, s_slipped=s_slipped)
    return tok, meta


_EDGE_MASKS: dict[tuple, torch.Tensor] = {}


def c_edge_mask(lay: Layout, dev: torch.device) -> torch.Tensor:
    """(1, 1, T, T) mask blocking ``c_write -> c_write - 1`` (location -> argument).

    The C write positions are fixed by the layout (``ve_batch``'s ``c_pos + 2``),
    so one static mask serves every batch; it is built once per device.
    """
    key = (lay, str(dev))
    if key not in _EDGE_MASKS:
        pos = lay.c_start + 4 * torch.arange(lay.n_c) + 2
        _EDGE_MASKS[key] = tt.copy_edge_mask(pos[None], lay.T, torch.device(dev))
    return _EDGE_MASKS[key]


def edge_mask_for(cell: dict, lay: Layout, dev: torch.device) -> torch.Tensor | None:
    return c_edge_mask(lay, dev) if cell.get("edge", False) else None


def event_selector(lay: Layout, meta: dict, B: int) -> torch.Tensor:
    """``linev`` positions: the location token of every E and C event (not the workbench)."""
    sel = torch.zeros(B, lay.T, dtype=torch.bool, device=meta["e_write"].device)
    sel.scatter_(1, meta["e_write"], True)
    sel.scatter_(1, meta["c_write"], True)
    return sel


def layout_sanity(lay: Layout, B: int = 64) -> dict:
    """Decode a CPU batch token by token and check every structural claim."""
    gen = torch.Generator().manual_seed(1)
    tok, m = ve_batch(lay, B, gen, slip=0.5, slip_gen=torch.Generator().manual_seed(2))
    t = tok.numpy()
    ok = dict(types=True, targets=True, distinct=True, slip_excludes=True, isolation=True,
              chunks=True, selector=True, workbench=True)
    for b in range(B):
        types = m["query_type"][b].numpy()
        ok["types"] &= all((types == k).sum() == lay.per_type for k in range(5))
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for p in range(0, lay.c_end, 4):
            if t[b, p] in (OBS, PUT) and lay.loc0 <= t[b, p + 2] < lay.sloc0:
                loc, val = int(t[b, p + 2]), int(t[b, p + 3])
                first.setdefault(loc, val)
                last[loc] = val
                if t[b, p] == PUT and int(t[b, p + 1]) != val:
                    ok["slip_excludes"] &= int(t[b, p + 1]) not in (val, first[loc])
        for p in range(lay.s_start, lay.s_start + 4 * lay.n_scratch, 4):
            ok["workbench"] &= (t[b, p] == PUT and lay.sloc0 <= t[b, p + 2] < lay.val0
                                and lay.val0 <= t[b, p + 3] < lay.fill0)
        for j in range(lay.n_loc):
            qp = int(m["query_pos"][b, j])
            loc, target = int(t[b, qp]), int(t[b, qp + 1])
            ok["targets"] &= t[b, qp - 1] == QRY and last.get(loc) == target
            ok["targets"] &= target == int(m["query_tok"][b, j])
            kind = int(types[j])
            if kind in (A, X):
                ok["distinct"] &= first[loc] != target and first[loc] == int(m["query_old"][b, j])
            if kind in (U, S):
                ok["distinct"] &= first[loc] == target
            ok["isolation"] &= qp - (lay.receptive - 1) > lay.c_end - 1
        ok["chunks"] &= int(m["e_write"][b].max()) < lay.c_start
        ok["chunks"] &= int(m["c_write"][b].min()) >= lay.c_start
        ok["chunks"] &= int(m["c_write"][b].max()) < lay.c_start + lay.chunk
    ok["selector"] = bool((event_selector(lay, m, B).sum(1) == lay.n_e + lay.n_c).all())
    ok = {k: bool(v) for k, v in ok.items()}
    return dict(passed=all(ok.values()), **ok)


# --------------------------------------------------------------------------- settings


@dataclass
class VeSettings:
    process_name: str = "TTM-ve-01"
    device: str = "auto"
    workers: int = 1
    max_minutes: float = 120.0
    tf32: bool = False
    fused_adam: bool = True
    compile_backbone: bool = False
    # torch's MPS cap, relative to Metal's recommended working set (local runs only)
    mps_memory_fraction: float | None = None
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    backbone_mult: int = 4
    fast_mult: int = 4
    window: int = 64
    chunk: int = 64
    T: int = 512
    n_loc: int = 20
    n_val: int = 16
    n_fill: int = 16
    per_type: int = 4
    n_scratch_loc: int = 4
    n_scratch: int = 16
    q_space_max: int = 3
    steps: int = 2000
    batch: int = 32
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    warmup: int = 200
    lr_total: int = 3000
    lr_floor: float = 0.1
    grad_clip: float = 1.0
    g_clip: float = 10.0
    eta_lr: float = 0.0
    quick_every: int = 25
    n_quick: int = 64
    n_eval: int = 256
    eval_batch: int = 32
    save_final: bool = True
    cells: list[dict] = field(default_factory=list)

    def layout(self) -> Layout:
        lay = Layout(self.n_loc, self.n_val, self.n_fill, self.per_type, self.n_scratch_loc,
                     self.n_scratch, self.q_space_max, self.T, self.chunk, self.window,
                     self.n_layers)
        lay.check()
        return lay


def settings(cfg: dict) -> VeSettings:
    known = {f.name for f in dataclasses.fields(VeSettings)}
    unknown = sorted(set(cfg) - known - {"_deadline"})
    if unknown:
        raise ValueError(f"unknown [ve] keys: {unknown}")
    return VeSettings(**{k: v for k, v in cfg.items() if k in known})


def cell_id(cell: dict) -> str:
    edge = "-edge" if cell.get("edge", False) else ""
    return (f"{cell['order']}-{cell['rule']}{edge}/eta{float(cell['eta']):g}"
            f"/slip{float(cell.get('slip', 0.0)):g}/s{int(cell['seed'])}")


def check_cell(cell: dict) -> None:
    if cell.get("order") not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}: {cell}")
    if cell.get("rule") not in RULES:
        raise ValueError(f"rule must be one of {tuple(RULES)}: {cell}")
    if float(cell.get("eta", 0)) <= 0:
        raise ValueError(f"eta must be positive: {cell}")
    if not 0.0 <= float(cell.get("slip", 0.0)) < 1.0:
        raise ValueError(f"slip must be in [0, 1): {cell}")
    if not isinstance(cell.get("edge", False), bool):
        raise ValueError(f"edge must be a bool: {cell}")
    extra = set(cell) - {"order", "rule", "eta", "slip", "seed", "edge"}
    if extra:
        raise ValueError(f"unknown cell keys {sorted(extra)}")


# --------------------------------------------------------------------------- model


def build(st: VeSettings, lay: Layout, seed: int, dev: torch.device) -> tt.TTTModel:
    """``exp_ttt.build_model`` for an ``e2e1emb`` arm: same init stream, frozen token code.

    The initialisation depends on ``seed`` only, so every rule and order of a seed
    starts from the same weights.
    """
    torch.manual_seed(seed)
    model = tt.TTTModel(lay.V, st.d_model, st.n_layers, st.n_heads, st.backbone_mult,
                        st.fast_mult, st.window)
    gen = torch.Generator(device="cpu").manual_seed(seed + EMB_SEED_OFFSET)
    emb = torch.randn(lay.V, st.d_model, generator=gen)
    model.frozen_emb = emb / emb.square().mean(-1, keepdim=True).sqrt()
    if st.eta_lr == 0:
        model.rho.requires_grad_(False)
    return model.to(dev)


def build_optim(model: tt.TTTModel, st: VeSettings, dev: torch.device) -> torch.optim.AdamW:
    """``exp_ttt.build_optim``'s groups; fused on accelerators, where ``found_inf`` works."""
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and n != "rho"]
    plain = [p for n, p in model.named_parameters() if p.ndim < 2 and n != "rho"]
    groups = [{"params": decay, "weight_decay": st.weight_decay, "base_lr": st.lr},
              {"params": plain, "weight_decay": 0.0, "base_lr": st.lr}]
    if model.rho.requires_grad:
        groups.append({"params": [model.rho], "weight_decay": 0.0, "base_lr": st.eta_lr})
    fused = bool(st.fused_adam) and dev.type in ("cuda", "mps")
    return torch.optim.AdamW(groups, lr=st.lr, betas=(st.beta1, st.beta2), fused=fused)


class DeviceStats:
    """``GradStats`` drop-in: per-chunk sums of inner-gradient norms and clip hits on device."""

    def __init__(self, n_chunk: int, dev: torch.device) -> None:
        self.norm = torch.zeros(n_chunk, device=dev)
        self.clip = torch.zeros(n_chunk, device=dev)
        self.count = torch.zeros(n_chunk, device=dev)

    def add(self, norm: torch.Tensor, clipped: torch.Tensor, index: int = -1) -> None:
        self.norm[index] += norm.detach().sum()
        self.clip[index] += clipped.detach().sum()
        self.count[index] += norm.numel()

    def summary(self) -> dict:
        n = self.count.clamp_min(1)
        return dict(norm_mean_by_chunk=(self.norm / n).tolist(),
                    clipped_by_chunk=(self.clip / n).tolist())


def train_loss(model: tt.TTTModel, lay: Layout, st: VeSettings, cell: dict, tok: torch.Tensor,
               meta: dict, stats: DeviceStats) -> torch.Tensor:
    """Outer loss of one batch: exp_ttt's sequential inner loop with the cell's rule."""
    idx, y = tok[:, :-1], tok[:, 1:]
    sel = event_selector(lay, meta, len(idx)) if cell["rule"] == "linev" else None
    h = model.backbone(idx, edge_mask=edge_mask_for(cell, lay, idx.device))
    return tt.sequential_inner(model, h, y, model.per_sequence(len(idx)),
                               model.etas(float(cell["eta"])), st.g_clip,
                               cell["order"] == "so", stats, lay.chunk, idx, sel=sel,
                               rule=RULES[cell["rule"]])


# --------------------------------------------------------------------------- evaluation


def _names() -> list[str]:
    names = []
    for k in TYPES:
        names += [f"hit_{k}", f"ce_{k}", f"stale_{k}", f"n_{k}"]
    for k in C_TYPES:
        names += [f"resid_{k}", f"resid0_{k}", f"oldp_{k}", f"oldp0_{k}", f"act_{k}", f"nc_{k}"]
    return names + ["act_E", "n_E", "resid0_wb", "n_wb"]


METRIC_NAMES = _names()


def _gather_last(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    return x.gather(1, pos[..., None].expand(-1, -1, x.shape[-1]))


def batch_metrics(model: tt.TTTModel, lay: Layout, st: VeSettings, cell: dict,
                  tok: torch.Tensor, meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-side sums (``METRIC_NAMES`` order) for one batch, and per-sequence accuracy.

    Queries: hit, CE and ``stale`` (prediction == the pre-change value) per type.
    C writes: ``resid`` = 1 - p(observed value) under the streaming weights (``W_1``)
    and ``resid0`` under ``W0``; ``oldp`` / ``oldp0`` = p(pre-change value), the
    size of the erase term in a CE write. ``act``: norm of the fast block's hidden
    activation ``gelu(W1 x + b1)`` under ``W0`` at the write, the factor that scales
    every ``W2`` write there (``act_E`` at the E writes for reference). ``resid0_wb``:
    the copy residual at the workbench, i.e. whether the copy path has been learned.
    """
    idx, y = tok[:, :-1], tok[:, 1:]
    sel = event_selector(lay, meta, len(idx)) if cell["rule"] == "linev" else None
    with torch.no_grad():
        h = model.backbone(idx, edge_mask=edge_mask_for(cell, lay, idx.device))
    etas = [e.detach() for e in model.etas(float(cell["eta"]))]
    logits = tt.eval_sequential(model, idx, y, etas, st.g_clip, lay.chunk, tt.GradStats(), sel,
                                h=h, rule=RULES[cell["rule"]])
    with torch.no_grad():
        sums: dict[str, torch.Tensor] = {}
        qz = _gather_last(logits, meta["query_pos"])
        pred = qz.argmax(-1)
        ce = F.cross_entropy(qz.transpose(1, 2), meta["query_tok"], reduction="none")
        hit = (pred == meta["query_tok"]).float()
        stale = (pred == meta["query_old"]).float()
        qtype = F.one_hot(meta["query_type"], 5).float()                       # (B,K,5)
        for name, val in (("hit", hit), ("ce", ce), ("stale", stale)):
            per = torch.einsum("bk,bkt->t", val, qtype)
            for i, k in enumerate(TYPES):
                sums[f"{name}_{k}"] = per[i]
        count = qtype.sum((0, 1))
        for i, k in enumerate(TYPES):
            sums[f"n_{k}"] = count[i]

        def probs_at(pos: torch.Tensor, weights) -> torch.Tensor:
            hz = _gather_last(h, pos)
            return model.read(model.fast_block(hz, idx.gather(1, pos), weights)).softmax(-1)

        def act_at(pos: torch.Tensor) -> torch.Tensor:
            x = model.fast_input(_gather_last(h, pos), idx.gather(1, pos))
            return F.gelu(x @ model.head.W1.T + model.head.b1).norm(dim=-1)

        cw = meta["c_write"]
        target = y.gather(1, cw)
        old = meta["c_old"].clamp_min(0)
        p_stream = _gather_last(logits, cw).softmax(-1)
        p_w0 = probs_at(cw, model.slow())
        ctype = F.one_hot(meta["c_type"] - 1, 4).float()                        # S, A, X, N
        has_old = (meta["c_old"] >= 0).float()
        rows = dict(resid=1 - p_stream.gather(-1, target[..., None]).squeeze(-1),
                    resid0=1 - p_w0.gather(-1, target[..., None]).squeeze(-1),
                    oldp=p_stream.gather(-1, old[..., None]).squeeze(-1) * has_old,
                    oldp0=p_w0.gather(-1, old[..., None]).squeeze(-1) * has_old,
                    act=act_at(cw))
        for name, val in rows.items():
            per = torch.einsum("bc,bct->t", val, ctype)
            for i, k in enumerate(C_TYPES):
                sums[f"{name}_{k}"] = per[i]
        ccount = ctype.sum((0, 1))
        for i, k in enumerate(C_TYPES):
            sums[f"nc_{k}"] = ccount[i]
        sums["act_E"] = act_at(meta["e_write"]).sum()
        sums["n_E"] = torch.tensor(float(meta["e_write"].numel()), device=idx.device)
        if lay.n_scratch:
            sw = meta["s_write"]
            pw = probs_at(sw, model.slow())
            sums["resid0_wb"] = (1 - pw.gather(-1, y.gather(1, sw)[..., None])).sum()
            sums["n_wb"] = torch.tensor(float(sw.numel()), device=sw.device)
        else:
            sums["resid0_wb"] = sums["n_wb"] = torch.zeros((), device=idx.device)
        row = torch.stack([sums[n] for n in METRIC_NAMES])
        per_seq = torch.einsum("bk,bkt->bt", hit, qtype) / qtype.sum(1)
    return row, per_seq


def finish(row: np.ndarray) -> dict:
    """Turn a summed metric row into per-type rates."""
    s = dict(zip(METRIC_NAMES, row.tolist()))
    out: dict[str, Any] = {}
    for k in TYPES:
        n = s[f"n_{k}"] or 1.0
        out[f"acc_{k}"] = s[f"hit_{k}"] / n
        out[f"ce_{k}"] = s[f"ce_{k}"] / n
        if k in ("A", "X"):
            err = s[f"n_{k}"] - s[f"hit_{k}"]
            out[f"stale_{k}"] = s[f"stale_{k}"] / n
            out[f"stale_err_{k}"] = s[f"stale_{k}"] / err if err > 0 else None
    for k in C_TYPES:
        n = s[f"nc_{k}"] or 1.0
        out[f"resid_{k}"] = s[f"resid_{k}"] / n
        out[f"resid0_{k}"] = s[f"resid0_{k}"] / n
        out[f"act_{k}"] = s[f"act_{k}"] / n
        if k != "N":
            out[f"oldp_{k}"] = s[f"oldp_{k}"] / n
            out[f"oldp0_{k}"] = s[f"oldp0_{k}"] / n
    out["act_E"] = s["act_E"] / s["n_E"] if s["n_E"] else None
    out["resid0_wb"] = s["resid0_wb"] / s["n_wb"] if s["n_wb"] else None
    return out


def fixed_set(lay: Layout, n: int, seed: int, dev: torch.device, batch: int) -> list:
    """Clean evaluation batches drawn on the CPU, so every device scores the same episodes."""
    tok, meta = ve_batch(lay, n, torch.Generator().manual_seed(seed))
    return [(tok[lo:lo + batch].to(dev), {k: v[lo:lo + batch].to(dev) for k, v in meta.items()})
            for lo in range(0, n, batch)]


def evaluate(model, lay, st, cell, data) -> tuple[torch.Tensor, torch.Tensor]:
    was = model.training
    model.eval()
    rows, seqs = zip(*(batch_metrics(model, lay, st, cell, tok, meta) for tok, meta in data))
    model.train(was)
    return torch.stack(rows).sum(0), torch.cat(seqs)


# --------------------------------------------------------------------------- one cell

QUICK_COLS = ["step", *(f"acc_{k}" for k in TYPES), "stale_A", "stale_X", "oldp_X", "oldp_A",
              "resid0_A", "resid0_N", "resid0_wb", "act_A", "act_X", "act_N", "act_E",
              "train_loss"]


class StepTimer:
    """Device-event timing of training blocks, read once at the end."""

    def __init__(self, dev: torch.device) -> None:
        self.dev, self.pairs = dev, []

    def mark(self):
        if self.dev.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
        elif self.dev.type == "mps":
            e = torch.mps.event.Event(enable_timing=True)
        else:
            return time.perf_counter()
        e.record()
        return e

    def block(self, start, end, n: int) -> None:
        self.pairs.append((start, end, n))

    def ms_per_step(self, skip: int = 1) -> float | None:
        tt._sync(self.dev)
        rates = []
        for start, end, n in self.pairs[skip:]:
            ms = (end - start) * 1e3 if isinstance(start, float) else start.elapsed_time(end)
            rates.append(ms / n)
        return round(float(np.median(rates)), 3) if rates else None


_DEADLINE = [float("inf")]
GRACE_S = 300.0  # the parent stops waiting this long after the deadline


def _device_used_mb(dev: torch.device) -> float | None:
    """Whole-device memory in use (every process and the CUDA contexts), MiB; CUDA only."""
    if dev.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(dev)
    return (total - free) / 2**20


def train_cell(cell: dict, st: VeSettings, dev: torch.device, tag: str) -> dict:
    """Train one cell and return its JSON record (CELL_RESULT)."""
    check_cell(cell)
    lay = st.layout()
    seed = int(cell["seed"])
    slip = float(cell.get("slip", 0.0))
    t_start = time.time()
    model = build(st, lay, seed, dev)
    if st.compile_backbone:
        for blk in model.blocks:
            blk.compile()
    opt = build_optim(model, st, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    device_skip = dev.type in ("cuda", "mps") and bool(st.fused_adam)
    gen = torch.Generator(device=dev).manual_seed(TRAIN_SEED * 1000 + seed)
    slip_gen = torch.Generator(device=dev).manual_seed(SLIP_SEED * 1000 + seed)
    quick = fixed_set(lay, st.n_quick, QUICK_SEED, dev, st.eval_batch)
    stats = DeviceStats(lay.T // lay.chunk, dev)
    n_steps, total = st.steps, max(st.lr_total, st.steps)
    loss_buf = torch.zeros(n_steps, device=dev)
    norm_buf = torch.zeros(n_steps, device=dev)
    bad_buf = torch.zeros(n_steps, device=dev)
    found_inf = torch.zeros((), device=dev)
    if device_skip:
        opt.found_inf = found_inf
    timer = StepTimer(dev)
    quick_rows: list[list] = []
    every = max(1, st.quick_every)
    diverged, stopped, done = False, None, 0
    dev_used, t_first_quick = 0.0, None
    block_start, block_n = timer.mark(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(n_steps):
        scale = tt.lr_scale(step, st.warmup, total, st.lr_floor)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        tok, meta = ve_batch(lay, st.batch, gen, slip, slip_gen)
        loss = train_loss(model, lay, st, cell, tok, meta, stats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, st.grad_clip)
        bad = ~torch.isfinite(norm) | ~torch.isfinite(loss.detach())
        if device_skip:
            found_inf.copy_(bad.float())
            opt.step()                       # fused AdamW: no-op and step rollback if found_inf
        elif not bool(bad):
            opt.step()
        loss_buf[step] = loss.detach()
        norm_buf[step] = norm.detach()
        bad_buf[step] = bad.float()
        block_n += 1
        done = step + 1
        if done % every == 0 or done == n_steps:
            timer.block(block_start, timer.mark(), block_n)
            row, _ = evaluate(model, lay, st, cell, quick)
            host = torch.cat([row, bad_buf[:done].sum()[None],
                              loss_buf[done - 1:done]]).cpu().numpy()
            rates = finish(host[:len(METRIC_NAMES)])
            n_bad, last_loss = int(host[-2]), float(host[-1])
            dev_used = max(dev_used, _device_used_mb(dev) or 0.0)
            if t_first_quick is None:
                t_first_quick = round(time.time() - t_start, 2)  # build + compile + first block
            quick_rows.append([done] + [round(float(rates[c]), 4) for c in QUICK_COLS[1:-1]]
                              + [round(last_loss, 4)])
            if done % (4 * every) == 0 or done == n_steps:
                print(f"  {tag} step {done}/{n_steps} loss={last_loss:.4f} "
                      + " ".join(f"{k}={rates[f'acc_{k}']:.3f}" for k in TYPES)
                      + f" staleA={rates['stale_A']:.3f} oldpX={rates['oldp_X']:.3f}"
                      f" resid0A={rates['resid0_A']:.3f} resid0wb={rates['resid0_wb']:.3f}"
                      f" bad={n_bad}", flush=True)
            if done >= 20 and n_bad / done > 0.05:
                diverged, stopped = True, "diverged"
                print(f"  WARNING {tag}: diverged after {done} steps", flush=True)
                break
            if time.time() > _DEADLINE[0]:
                stopped = "deadline"
                print(f"  WARNING {tag}: deadline hit at step {done}", flush=True)
                break
            block_start, block_n = timer.mark(), 0
    ev = fixed_set(lay, st.n_eval, EVAL_SEED, dev, st.eval_batch)
    row, per_seq = evaluate(model, lay, st, cell, ev)
    final = finish(row.cpu().numpy())
    norms = norm_buf[:done].cpu().numpy()
    post = norms[st.warmup:] if done > st.warmup else norms
    finite = post[np.isfinite(post)]
    out = dict(cell_id=tag, order=cell["order"], rule=cell["rule"],
               edge=bool(cell.get("edge", False)), seed=seed,
               eta=float(cell["eta"]), slip=slip, steps_done=done, steps=n_steps, stopped=stopped,
               device=str(dev), torch=torch.__version__, diverged=diverged,
               nonfinite_steps=int(bad_buf[:done].sum().item()),
               ms_step=timer.ms_per_step(), wall_s=round(time.time() - t_start, 1),
               t_first_quick_s=t_first_quick, t_start=round(t_start, 2),
               peak_mb=tt._peak_mem(dev),
               peak_reserved_mb=round(torch.cuda.max_memory_reserved(dev) / 2**20, 1)
               if dev.type == "cuda" else None,
               dev_used_max_mb=round(dev_used, 1) if dev.type == "cuda" else None,
               final=final,
               per_seq_acc={k: per_seq[:, i].cpu().tolist() for i, k in enumerate(TYPES)},
               quick_cols=QUICK_COLS, quick=quick_rows, inner_grad=stats.summary(),
               outer_grad=dict(median=float(np.median(finite)) if finite.size else None,
                               max=float(finite.max()) if finite.size else None,
                               clip_fraction=float(np.mean(finite > st.grad_clip))
                               if finite.size else None),
               train_loss_last=float(loss_buf[done - 1].item()),
               config=dict(layout=dataclasses.asdict(lay), tf32=st.tf32, fused=device_skip,
                           compile_backbone=st.compile_backbone))
    if st.save_final:
        folder = Path("checkpoints")
        folder.mkdir(exist_ok=True)
        stem = f"ve__{tag.replace('/', '__')}"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   folder / f"{stem}.pt")
        np.savez_compressed(folder / f"{stem}__trace.npz", loss=loss_buf[:done].cpu().numpy(),
                            outer_norm=norms)
        out["checkpoint"] = str(folder / f"{stem}.pt")
    return out


# --------------------------------------------------------------------------- runners


def _setup_worker(st: VeSettings, name: str, dev: torch.device) -> None:
    tt.configure_tf32(st.tf32, name)
    if dev.type in ("cuda", "mps"):
        torch.set_num_threads(2)
        if dev.type == "mps" and st.mps_memory_fraction is not None:
            torch.mps.set_per_process_memory_fraction(float(st.mps_memory_fraction))
    else:
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // max(1, st.workers))))


def _run_one(cell: dict, st: VeSettings, dev: torch.device) -> dict:
    tag = cell_id(cell)
    if time.time() > _DEADLINE[0]:
        return dict(cell_id=tag, error="not started: deadline")
    try:
        return train_cell(cell, st, dev, tag)
    except Exception as exc:  # a cell error must not stop the wave
        traceback.print_exc()
        return dict(cell_id=tag, error=repr(exc))


def _worker(index: int, cfg: dict, jobs, results, lock) -> None:
    sys.stdout = tt._LockedLines(sys.stdout, lock)
    sys.stderr = tt._LockedLines(sys.stderr, lock)
    st = settings(cfg)
    _DEADLINE[0] = float(cfg.get("_deadline", float("inf")))
    tt._mark_process(f"{st.process_name}-w{index}")
    dev = tt._device(st.device)
    _setup_worker(st, str(index), dev)
    while (cell := jobs.get()) is not None:
        record = _run_one(cell, st, dev)
        print("CELL_RESULT " + json.dumps(record), flush=True)
        results.put((record["cell_id"], record))


def run(cfg: dict, seed: int) -> dict:
    st = settings(cfg)
    tt._mark_process(st.process_name)
    lay = st.layout()
    for cell in st.cells:
        check_cell(cell)
    ids = [cell_id(c) for c in st.cells]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cells")
    sanity = layout_sanity(lay)
    print("VE_SANITY " + json.dumps(sanity), flush=True)
    if not sanity["passed"]:
        raise RuntimeError("E1-VE layout sanity failed")
    print(f"ve | V={lay.V} T={lay.T} c_start={lay.c_start} s_start={lay.s_start} "
          f"r_start={lay.r_start} receptive={lay.receptive} cells={len(ids)} "
          f"workers={st.workers}", flush=True)
    t0 = time.time()
    deadline = t0 + st.max_minutes * 60
    records: dict[str, dict] = {}
    if st.workers <= 1:
        _DEADLINE[0] = deadline
        dev = tt._device(st.device)
        _setup_worker(st, "main", dev)
        for cell in st.cells:
            records[cell_id(cell)] = rec = _run_one(cell, st, dev)
            print("CELL_RESULT " + json.dumps(rec), flush=True)
    else:
        ctx = mp.get_context("spawn")
        jobs, results, lock = ctx.Queue(), ctx.Queue(), ctx.Lock()
        for cell in st.cells:
            jobs.put(cell)
        for _ in range(st.workers):
            jobs.put(None)
        shared = {**cfg, "_deadline": deadline}
        procs = [ctx.Process(target=_worker, args=(i, shared, jobs, results, lock))
                 for i in range(st.workers)]
        for p in procs:
            p.start()
        def drain(wait: float) -> None:
            while True:
                try:
                    tag, record = results.get(timeout=wait)
                except queue.Empty:
                    return
                records[tag] = record

        while len(records) < len(ids):
            if time.time() > deadline + GRACE_S:
                print(f"!! parent deadline: {len(ids) - len(records)} cells without a result",
                      flush=True)
                break
            try:
                tag, record = results.get(timeout=30)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            records[tag] = record
        # one shared, bounded cleanup: keep draining (a child cannot exit while its queue
        # feeder is blocked), then terminate, then kill whatever is left
        stop = time.time() + 60
        while any(p.is_alive() for p in procs) and time.time() < stop:
            drain(0.2)
            for p in procs:
                p.join(timeout=0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        drain(0.5)
        codes = [p.exitcode for p in procs]
        if any(c != 0 for c in codes):
            print(f"!! worker exit codes {codes}", flush=True)
    elapsed = time.time() - t0
    done = [r for r in records.values() if "steps_done" in r]

    def most(key):
        values = [r[key] for r in done if r.get(key) is not None]
        return max(values) if values else None

    first = [r["t_first_quick_s"] for r in done if r.get("t_first_quick_s") is not None]
    steps = sum(r["steps_done"] for r in done)
    # end-to-end rate of the whole pool: this, not a sum of per-cell rates, sizes a wave
    complete = len(done) == len(ids) and all(r.get("stopped") is None for r in done)
    throughput = dict(workers=st.workers, cells=len(ids), finished=len(done), complete=complete,
                      steps_done=steps, elapsed_s=round(elapsed, 1),
                      agg_steps_per_s=round(steps / elapsed, 2),
                      dev_used_max_mb=most("dev_used_max_mb"),
                      peak_reserved_max_mb=most("peak_reserved_mb"),
                      t_first_quick_median_s=float(np.median(first)) if first else None)
    print("RUN_SUMMARY " + json.dumps(throughput), flush=True)
    summary = {}
    for tag in ids:
        rec = records.get(tag, dict(cell_id=tag, error="no result: worker died or parent deadline"))
        if "final" in rec:
            summary[tag] = {k: round(rec["final"][f"acc_{k}"], 4) for k in TYPES} | dict(
                stale_err_A=rec["final"]["stale_err_A"], ms_step=rec["ms_step"],
                wall_s=rec["wall_s"], stopped=rec["stopped"])
        else:
            summary[tag] = dict(error=rec.get("error"))
    return dict(sanity=sanity, cells=summary, throughput=throughput)
