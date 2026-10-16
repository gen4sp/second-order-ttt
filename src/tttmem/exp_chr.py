"""CHR: chronology of first-order write starvation on echo.

Three measurements on the R4-4b/TH-RC toy at a fixed shared eta (``eta_lr = 0``):

1. **Chronology.** Every ``log_every`` steps each cell logs the kv write residual
   ``r = 1 - p_W0(v)`` at the write positions (probe set, chunk 0, slow weights),
   the query accuracy per query chunk (quick set, streaming inner loop) and the
   read geometry at the query positions: backbone, slow and memory components of
   the pre-``ln_f`` vector.
2. **Acquisition-point forks.** In the fork arms (FO-CE ``e2e1emb`` and FO + 3 %
   copy noise ``e2e1emb_cn97``) the first log with quick accuracy >= ``fork_acc``
   clones model, AdamW state and the data stream. Each configured branch continues
   from that state to the last step with another inner rule: ``ce`` (a twin of the
   main run), ``rp`` (the inner CE gradient of every kv token floored at residual
   ``r_ref``), ``nc`` (the same added inner-gradient norm, per sequence, on the
   value->W positions instead), ``rs`` (every kv token's CE gradient capped at
   residual ``r_ref``) or ``ns`` (the same removed norm, per sequence, taken from
   the other chunk tokens instead). The main run itself continues as the CE branch.
3. **Finals intervention.** On every final model the late query chunk's slow (or
   backbone) contribution is scaled by a label-free per-sequence alpha, against an
   equal-norm random direction and a sham on the early query chunk.
4. **Per-event mediation (finals, fixed weights).** On every main final, half of the
   eval set's write events get a corrupted copy source; the chunk-0 kv CE terms are
   reweighted so that corrupted events write with their clean twin's residual
   (CLAMP) or clean events with the corrupted residual (FLOOR).

The inherited training step (``exp_ttt.arm_loss`` + AdamW + clipping) is used
unchanged for the ``ce`` rule; the ``rp``/``nc`` inner loop reproduces
``exp_ttt.sequential_inner``'s CE path and only adds the chunk's extra gradient
(``chr_sanity`` checks both identities at start-up).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_kag as kag
from . import exp_ttt as ttt

RULES = ("ce", "rp", "nc", "rs", "ns")
FORK_ARMS_ALLOWED = ("e2e1emb", "e2e1emb_cn97")
CHR_ARMS = ("e2e1emb", "e2e2emb", "e2e1emb_cn97")
RND_SEED = 31337
MEDIATION_STEP = 3  # corrupt_echo-style stream index of the mediation eval set (0/1/2 are taken)

CHR_DEFAULTS: dict[str, Any] = {
    "log_every": 5,
    "probe_every": 25,
    "cross_every": 25,
    "fork_arms": ["e2e1emb"],
    "fork_acc": 0.90,
    "forks": [],  # a list (every fork arm) or a table arm -> list; same for cross/final rules
    "rp_cap": 1000.0,
    "cross_rules": [],
    "final_rules": [],
    "chunk_early": 5,
    "chunk_late": 6,
    "ckpt_steps": [],
    "ckpt_dense_arms": ["e2e1emb"],  # main runs of these arms save at ckpt_steps
    "ckpt_steps_sparse": None,  # main-run checkpoint steps of the other arms (None: ckpt_steps)
    "branch_ckpt_every": 100,
    "branch_ckpt_tags": [],  # branch rule tags that also save every branch_ckpt_every steps
    "mediation_p": 0.5,  # per-event copy-corruption probability of the mediation set (0: off)
    "mediation_cap": 1000.0,  # FLOOR weight cap
}


@dataclasses.dataclass(frozen=True)
class Rule:
    """An inner rule: ``ce``; ``rp``/``nc`` with residual floor ``r_ref`` and weight cap;
    ``rs``/``ns`` with residual cap ``r_ref``."""

    name: str = "ce"
    r_ref: float = 0.0
    cap: float = 1000.0
    outer: str = "none"

    def __post_init__(self) -> None:
        if self.outer not in ("none", "kv", "rnd"):
            raise ValueError(f"unknown outer mask {self.outer!r}")
        if self.name not in RULES:
            raise ValueError(f"unknown inner rule {self.name!r}")
        if self.name in ("rp", "nc") and not (0.0 <= self.r_ref < 1.0 and self.cap >= 1.0):
            raise ValueError(f"bad rule parameters {self}")
        if self.name in ("rs", "ns") and not 0.0 < self.r_ref <= 1.0:
            raise ValueError(f"bad rule parameters {self}")

    @property
    def tag(self) -> str:
        return "ce" if self.name == "ce" else f"{self.name}{round(100 * self.r_ref):02d}"


CE = Rule()


def parse_rule(spec: dict | str, cap: float) -> Rule:
    """``"ce"`` / ``"rp03"``-style tags, or ``{rule, r_min | r_cap, cap}`` tables."""
    if isinstance(spec, str):
        if spec == "ce":
            return CE
        name, pct = spec[:2], spec[2:]
        return Rule(name, int(pct) / 100.0, cap)
    keys = [k for k in ("r_min", "r_cap") if k in spec]
    if len(keys) > 1:
        raise ValueError(f"give r_min or r_cap, not both: {spec}")
    return Rule(str(spec["rule"]), float(spec[keys[0]]) if keys else 0.0, float(spec.get("cap", cap)),
                outer=str(spec.get("outer", "none")))


def _rules_by_arm(value: Any, cap: float) -> dict[str, list[Rule]]:
    """A list applies to every arm (key ``*``); a table maps arm -> list."""
    if isinstance(value, dict):
        return {str(a): [parse_rule(f, cap) for f in v] for a, v in value.items()}
    return {"*": [parse_rule(f, cap) for f in value]}


# --------------------------------------------------------------------------- positions


W_TOKEN = 0  # exp_ttt.EchoVocab.write


def position_masks(write_pos: Any, y: torch.Tensor, dev: torch.device) -> dict[str, torch.Tensor]:
    """Float (B, T) masks of the kv write positions and the value->W control positions.

    The control is ``write_pos + 1`` only where its target is the W marker: the last write
    event's value predicts the first filler, which is not a value->W transition.
    """
    T = y.shape[1]
    wp = torch.as_tensor(np.asarray(write_pos), device=dev, dtype=torch.long)
    if wp.ndim != 2 or bool(((wp < 0) | (wp + 1 >= T)).any()):
        raise ValueError("write positions must be B x K inside [0, T-1)")
    kv = torch.zeros(wp.shape[0], T, device=dev)
    vw = torch.zeros(wp.shape[0], T, device=dev)
    kv.scatter_(1, wp, 1.0)
    vw.scatter_(1, wp + 1, 1.0)
    vw = vw * (y.to(dev) == W_TOKEN).float()
    if bool((kv * vw).any()):
        raise ValueError("kv and control positions overlap")
    return dict(kv=kv, vw=vw)


def write_chunks(write_pos: Any, chunk: int) -> set[int]:
    """Chunks holding a kv or control position (host-side, no device sync)."""
    wp = np.asarray(write_pos)
    return set((wp // chunk).flatten().tolist()) | set(((wp + 1) // chunk).flatten().tolist())


def _joint_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    """Per-leading-index norm over the four fast tensors, as ``_clip_fast`` computes it."""
    sq = None
    for g, key in zip(grads, ttt.FAST_KEYS):
        tail = ttt.MATRIX_NDIM[key]
        part = (g.reshape(*g.shape[:g.ndim - tail], -1) ** 2).sum(-1)
        sq = part if sq is None else sq + part
    return sq.clamp_min(0).sqrt()


def _bcast(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return v.reshape(-1, *([1] * (g.ndim - 1)))


def rule_weights(ce: torch.Tensor, rule: Rule) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals ``r = 1 - exp(-ce)`` (via expm1) and the kv token weights of ``rule``.

    ``rp``/``nc``: ``clamp(r_ref / r, 1, cap)`` (floor); ``rs``/``ns``: ``min(1, r_ref / r)`` (cap).
    """
    r = -torch.expm1(-ce)
    ratio = rule.r_ref / r.clamp_min(1e-30)
    if rule.name in ("rs", "ns"):
        return r, ratio.clamp(max=1.0)
    return r, ratio.clamp(1.0, rule.cap)


def rule_extra(rule: Rule, ce: torch.Tensor, W: list[torch.Tensor], kv: torch.Tensor,
               vw: torch.Tensor, chunk: int, retain: bool) -> tuple[list[torch.Tensor] | None, dict]:
    """Extra inner gradient of one chunk under ``rule`` (None for ``ce``).

    ``ce`` is the chunk's (B, chunk) token CE with graph to ``W``; ``kv`` / ``vw`` are
    the chunk slices of the position masks. ``rp``/``rs`` add the gradient of
    ``sum_kv (w_i - 1) ce_i / chunk`` with detached ``rule_weights`` ``w_i``,
    ``r_i = 1 - p_i(v)``: each kv token's CE gradient then has at least (``rp``) or at
    most (``rs``) the size it would have at residual ``r_ref``. ``nc`` adds, per
    sequence, the gradient of the value->W tokens' CE scaled to the joint norm rp
    adds; ``ns`` subtracts, per sequence, the other (non-kv) tokens' CE gradient
    scaled to the joint norm rs removes, with the scale capped at 1 (those tokens'
    gradient is at most removed, never reversed; ``cap_bound`` flags the cap).
    ``info["added"]`` is the *requested* joint norm (what rp adds / rs removes);
    ``info["actual"]`` the norm of the returned extra (NC: 0 without a control direction;
    NS: at most the control's own norm when the cap binds).
    """
    if rule.name == "ce":
        return None, {}
    r, w = rule_weights(ce.detach(), rule)
    tok = (w - 1.0) * kv
    extra = torch.autograd.grad(((tok * ce).sum(1) / chunk).sum(), W, retain_graph=True)
    added = _joint_norm(list(extra)).detach()
    sel = kv > 0
    info = dict(added=added, w_kv=w.detach(), r_kv=r, kv=kv)
    if rule.name == "rp":  # floor compliance: the cap binds where r < r_ref / cap
        info["cap_hit"] = ((w >= rule.cap) & sel).float().sum(1) / sel.float().sum(1).clamp_min(1)
    info["r_eff"] = torch.where(sel, (w * r), torch.full_like(r, float("nan")))
    if rule.name in ("rp", "rs"):
        info["actual"], info["match_fail"] = added, torch.zeros_like(added, dtype=torch.bool)
        return list(extra), info
    ctl_mask = vw if rule.name == "nc" else 1.0 - kv
    ctl = torch.autograd.grad(((ctl_mask * ce).sum(1) / chunk).sum(), W, retain_graph=retain)
    norm = _joint_norm(list(ctl)).detach()
    ok = torch.isfinite(norm) & (norm > 1e-12)  # no usable control direction otherwise
    coef = torch.where(ok, added / norm.clamp_min(1e-12), torch.zeros_like(added))
    target = added
    if rule.name == "ns":
        info["cap_bound"] = norm < added  # cannot remove the full norm (incl. no control at all)
        coef = coef.clamp(max=1.0)
        target = torch.minimum(added, norm)
        coef = -coef
    out = [_bcast(coef, c) * c for c in ctl]
    actual = _joint_norm(out).detach()
    info["coef"], info["actual"] = coef, actual
    info["match_fail"] = ((actual - target).abs() > 1e-4 * target + 1e-12) | ~torch.isfinite(actual)
    return out, info


# --------------------------------------------------------------------------- training loss


def chr_sequential_inner(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                         write_pos: Any, etas: list[torch.Tensor], g_clip: float, chunk: int,
                         stats: ttt.GradStats, rule: Rule, diag: dict | None = None) -> torch.Tensor:
    """First-order chunk recurrence of ``exp_ttt.sequential_inner`` plus ``rule``'s extra.

    The CE path (fast block, token CE, chunk mean, retained first-order gradient,
    joint clip, ``W - eta g``, per-chunk CE outer loss) is the inherited one; ``rp``
    and ``nc`` add their extra gradient before the clip in the chunks that hold a
    kv write. ``diag`` (optional) receives the detached added norms and clip flags
    of those chunks.
    """
    T, b = h.shape[1], chunk
    W, total, n_chunk = model.per_sequence(idx.shape[0]), 0.0, T // b
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    active = write_chunks(write_pos, b) if rule.name != "ce" else set()
    masks = position_masks(write_pos, y, h.device) if active else None
    for c in range(n_chunk):
        sl = slice(c * b, (c + 1) * b)
        ce = ttt._ce_tokens(model, ttt.fast_block(h[:, sl], model.head.ln, W,
                                                  None if x is None else x[:, sl]), y[:, sl])
        loss_c = ce.mean(1)
        grads = torch.autograd.grad(loss_c.sum(), W, create_graph=False, retain_graph=True)
        raw = list(grads)
        if c in active:
            extra, info = rule_extra(rule, ce, W, masks["kv"][:, sl], masks["vw"][:, sl], b, True)
            raw = [g + e for g, e in zip(raw, extra)]
        grads, norm, clipped = ttt._clip_fast(raw, g_clip)
        if c in active and diag is not None:
            for key, val in (("added", info["added"]), ("actual", info["actual"]), ("clipped", clipped.detach()),
                             ("match_fail", info["match_fail"]), ("cap_bound", info.get("cap_bound")),
                             ("cap_hit", info.get("cap_hit"))):
                if val is not None:
                    diag.setdefault(key, []).append(val.detach())
        stats.add(norm, clipped, c)
        total = total + loss_c.mean()
        W = [w - eta * g for w, eta, g in zip(W, etas, grads)]
    return total / n_chunk


def chr_arm_loss(model: ttt.TTTModel, arm: str, idx: torch.Tensor, y: torch.Tensor, eta_base: float,
                 g_clip: float, chunk: int, stats: ttt.GradStats, meta: dict, rule: Rule,
                 diag: dict | None = None, mask_rng: np.random.Generator | None = None) -> torch.Tensor:
    """The inherited ``exp_ttt.arm_loss`` for ``ce``; the FO rule loop otherwise."""
    if rule.name == "ce":
        keep = ttt.outer_keep(meta, tuple(y.shape), rule.outer, mask_rng, idx.device)
        return ttt.arm_loss(model, arm, idx, y, eta_base, g_clip, chunk, stats, keep=keep, meta=meta)
    if arm not in FORK_ARMS_ALLOWED:
        raise ValueError(f"rule {rule.tag} is defined for {FORK_ARMS_ALLOWED} only, not {arm}")
    h = model.backbone(idx, edge_mask=ttt.arm_edge_mask(model, arm, idx, meta))
    etas = ttt.regime_etas(model, eta_base)
    return chr_sequential_inner(model, h, y, idx, meta["write_pos"], etas, g_clip, chunk, stats,
                                rule, diag)


# --------------------------------------------------------------------------- streaming eval


@torch.no_grad()
def chr_stream(model: ttt.TTTModel, idx: torch.Tensor, y: torch.Tensor, write_pos: Any,
               eta_base: float, g_clip: float, chunk: int, rule: Rule,
               capture: tuple[int, ...] = (), kv_weights: torch.Tensor | None = None,
               clip_out: list | None = None, norm_out: list | None = None
               ) -> tuple[torch.Tensor, dict[int, list[torch.Tensor]], dict]:
    """``exp_ttt.eval_sequential``'s CE streaming loop with ``rule``'s write extra.

    ``kv_weights`` (B, K), only with ``ce``, weights each write event's kv CE term
    by a given detached factor (adds ``sum_kv (w_i - 1) ce_i / chunk``'s gradient).
    ``clip_out`` / ``norm_out`` (optional) receive the per-sequence clip flags / pre-clip
    joint norms of the write chunks.
    Returns logits ``(B, T, V)``, the fast weights each captured chunk was read with,
    and the backbone/fast-input tensors used (for the read decomposition).
    """
    if kv_weights is not None and rule.name != "ce":
        raise ValueError("kv_weights replace the rule; use them with ce only")
    h = model.backbone(idx).detach()
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    T, b = h.shape[1], chunk
    etas = ttt.regime_etas(model, eta_base)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(idx.shape[0])]
    active = (write_chunks(write_pos, b) if rule.name != "ce" or kv_weights is not None
              or clip_out is not None or norm_out is not None else set())
    masks = position_masks(write_pos, y, h.device) if active else None
    tokw = None
    if kv_weights is not None:
        wp = torch.as_tensor(np.asarray(write_pos), device=h.device, dtype=torch.long)
        tokw = torch.zeros(idx.shape[0], T, device=h.device).scatter_(1, wp, kv_weights.float() - 1.0)
    out, captured = [], {}
    with torch.enable_grad():
        for c in range(T // b):
            sl = slice(c * b, (c + 1) * b)
            if c in capture:
                captured[c] = [w.detach() for w in W]
            logits = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                               None if x is None else x[:, sl]))
            out.append(logits.detach())
            ce = ttt.token_ce(logits, y[:, sl])
            extra_on = c in active and (rule.name != "ce" or tokw is not None)
            grads = torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=extra_on)
            raw = list(grads)
            if extra_on and tokw is not None:
                extra = torch.autograd.grad(((tokw[:, sl] * ce).sum(1) / b).sum(), W)
                raw = [g + e for g, e in zip(raw, extra)]
            elif extra_on:
                extra, _ = rule_extra(rule, ce, W, masks["kv"][:, sl], masks["vw"][:, sl], b, False)
                raw = [g + e for g, e in zip(raw, extra)]
            grads, norm, clipped = ttt._clip_fast(raw, g_clip)
            if clip_out is not None and c in active:
                clip_out.append(clipped.detach().float())
            if norm_out is not None and c in active:
                norm_out.append(norm.detach().float())
            W = [(w - eta * g).detach().requires_grad_(True) for w, eta, g in zip(W, etas, grads)]
    return torch.cat(out, dim=1), captured, dict(h=h, x=x)


def _centred(v: torch.Tensor) -> torch.Tensor:
    return v - v.mean(-1, keepdim=True)


@torch.no_grad()
def read_parts(model: ttt.TTTModel, tensors: dict, captured: dict[int, list[torch.Tensor]],
               query_pos: Any, chunk: int) -> dict[str, torch.Tensor]:
    """Pre-``ln_f`` read vector at every query and its slow / memory split.

    ``v = h + head_W(x)`` with ``W`` the fast weights of the query's own chunk;
    ``v_slow = h + head_W0(x)`` (backbone plus the slow head at the same input);
    ``v_mem = v - v_slow``. The fast input ``x`` is the natural one (fixed code).
    """
    qp = torch.as_tensor(np.asarray(query_pos), device=tensors["h"].device, dtype=torch.long)
    hq = kag._gather(tensors["h"], qp)
    xq = None if tensors["x"] is None else kag._gather(tensors["x"], qp)
    qc = qp // chunk
    if not set(torch.unique(qc).tolist()) <= set(captured):
        raise ValueError(f"query chunks {sorted(set(torch.unique(qc).tolist()))} not all captured")
    v = torch.zeros_like(hq)
    for c, W in captured.items():
        vc = ttt.fast_block(hq, model.head.ln, W, xq)
        v = torch.where((qc == c)[..., None], vc, v)
    v0 = ttt.fast_block(hq, model.head.ln, model.per_sequence(hq.shape[0]), xq)
    return dict(h=hq, v=v, v_slow=v0, head0=v0 - hq, v_mem=v - v0, chunk=qc)


def _sigma(model: ttt.TTTModel, v: torch.Tensor) -> torch.Tensor:
    ln = model.ln_f
    return (v.var(-1, unbiased=False) + ln.eps).sqrt()


def geometry(model: ttt.TTTModel, parts: dict, chunks: tuple[int, ...]) -> dict:
    """Median read-geometry fields per query chunk."""
    out = {}
    ch, cv0 = _centred(parts["h"]), _centred(parts["v_slow"])
    cm, ch0 = _centred(parts["v_mem"]), _centred(parts["head0"])
    cos = F.cosine_similarity(ch, ch0, dim=-1, eps=1e-12)
    fields = dict(bb=ch.norm(dim=-1), slow=cv0.norm(dim=-1), mem=cm.norm(dim=-1),
                  sigma=_sigma(model, parts["v"]), cos_bb_head0=cos)
    for c in chunks:
        sel = parts["chunk"] == c
        if not bool(sel.any()):
            continue
        out[str(c)] = {k: round(float(v[sel].median()), 6) for k, v in fields.items()}
        out[str(c)]["n"] = int(sel.sum())
    return out


def _hits(model: ttt.TTTModel, v: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (model.read(v).argmax(-1) == target).float()


def _per_seq_ratio(num: torch.Tensor, chunk_id: torch.Tensor, early: int, late: int) -> torch.Tensor:
    """Per sequence: median over early-chunk queries / median over late-chunk queries.

    Lower-median convention (torch); a sequence without early or late queries takes the
    median ratio of the others (1 if none); clipped to [0, 1]. Vectorised (one host sync).
    """
    nan = torch.full_like(num, float("nan"))
    e = torch.nanmedian(torch.where(chunk_id == early, num, nan).cpu().double(), dim=1).values
    l = torch.nanmedian(torch.where(chunk_id == late, num, nan).cpu().double(), dim=1).values
    t = (e / l.clamp_min(1e-30)).float()
    t = torch.where(torch.isfinite(e) & torch.isfinite(l), t, torch.full_like(t, float("nan")))
    fill = torch.nanmedian(t) if bool(torch.isfinite(t).any()) else torch.tensor(1.0)
    return torch.where(torch.isfinite(t), t, fill).clamp(0.0, 1.0).to(num.device)


def _random_units(shape: tuple[int, ...], d: int, dev: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(*shape, d, generator=gen).to(dev)
    g = _centred(g)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)


N_RND = 4  # random directions per RND control (accuracies averaged)


@torch.no_grad()
def intervention(model: ttt.TTTModel, parts: dict, target: torch.Tensor, early: int, late: int,
                 seed: int) -> dict:
    """Finals intervention variants at the late (and, as sham, the early) query chunk.

    BB scales the backbone skip ``h`` (the fast input is LN(h)-based, so the code
    stays fixed) by ``alpha_bb``; SLOW scales ``v_slow = h + head_W0(x)`` by
    ``alpha_slow``; HEAD (descriptive) scales the slow head term ``head_W0(x)`` alone.
    Each alpha is per sequence: median centred norm at early-chunk queries / the same
    at late-chunk queries, clipped to [0, 1] (label-free). RND_* subtract a random
    centred unit direction with the norm the matching variant removes (accuracy
    averaged over ``N_RND`` fixed directions). SHAM5_* apply the late chunk's alpha to
    the early-chunk queries instead.
    """
    qc = parts["chunk"]
    late_m, early_m = (qc == late)[..., None], (qc == early)[..., None]
    v = parts["v"]
    terms = dict(BB=parts["h"], SLOW=parts["v_slow"], HEAD=parts["head0"])
    out: dict = {}

    def score(name, vv):
        hit = _hits(model, vv, target)
        row = {str(c): float(hit[qc == c].mean()) for c in (early, late) if bool((qc == c).any())}
        row["all"] = float(hit.mean())
        return row

    out["NAT"] = score("NAT", v)
    units = [_random_units(tuple(v.shape[:-1]), v.shape[-1], v.device, seed + 7919 * j) for j in range(N_RND)]
    for name, term in terms.items():
        alpha = _per_seq_ratio(_centred(term).norm(dim=-1), qc, early, late)[:, None, None]
        rm = (1 - alpha) * term
        n_rm = _centred(rm).norm(dim=-1, keepdim=True)
        out[name] = score(name, torch.where(late_m, v - rm, v))
        rnd = [score("RND", torch.where(late_m, v - n_rm * u, v)) for u in units]
        out[f"RND_{name}"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
        out[f"SHAM5_{name}"] = score(name, torch.where(early_m, v - rm, v))
        out[f"alpha_{name.lower()}_median"] = float(alpha.median())
    return out


# --------------------------------------------------------------------------- logged probes


@torch.no_grad()
def quick_log(model: ttt.TTTModel, data: list, eta_base: float, g_clip: float, chunk: int,
              rule: Rule, chunks: tuple[int, ...], geometry_on: bool = True) -> dict:
    """Quick-set accuracy per query chunk under ``rule`` plus the read geometry."""
    hits, ces, cids, geo_parts = [], [], [], []
    for idx, y, meta in data:
        logits, captured, tens = chr_stream(model, idx, y, meta["write_pos"], eta_base, g_clip,
                                            chunk, rule, capture=chunks if geometry_on else ())
        qp = torch.as_tensor(np.asarray(meta["query_pos"]), device=idx.device)
        tgt = torch.as_tensor(np.asarray(meta["query_tok"]), device=idx.device)
        lq = logits.gather(1, qp[..., None].expand(-1, -1, logits.shape[-1]))
        hits.append((lq.argmax(-1) == tgt).float())
        ces.append(F.cross_entropy(lq.reshape(-1, lq.shape[-1]), tgt.reshape(-1),
                                   reduction="none").view(tgt.shape))
        cids.append(qp // chunk)
        if geometry_on:
            geo_parts.append(read_parts(model, tens, captured, meta["query_pos"], chunk))
    hit, ce, cid = torch.cat(hits), torch.cat(ces), torch.cat(cids)
    out = dict(acc=round(float(hit.mean()), 6), qce=round(float(ce.mean()), 6))
    for c in torch.unique(cid).tolist():
        sel = cid == c
        out[f"acc{c}"] = round(float(hit[sel].mean()), 6)
        out[f"qce{c}"] = round(float(ce[sel].mean()), 6)
    if geometry_on:
        merged = {k: torch.cat([p[k] for p in geo_parts]) for k in geo_parts[0]}
        out["geo"] = geometry(model, merged, chunks)
    return out


@torch.no_grad()
def write_log(model: ttt.TTTModel, data: list, eta_base: float, g_clip: float, chunk: int,
              rule: Rule) -> dict:
    """Write-side state at the slow weights, chunk 0 of the probe set, under ``rule``.

    Residuals ``r = 1 - p(target)`` at the kv positions (target: the value), the
    value->W control positions and the W-marker positions; the chunk-0 inner
    gradient's joint norm and clip flag under the rule; the kv tokens' own write
    norm ``|eta * s * g_kv|`` (``s`` the rule's clip scale; for ``rp``/``rs`` the kv
    term includes the extra), the norm the rule adds (rp/nc) or removes (rs/ns), and
    for ``ns`` the fraction of sequences whose removal scale hit its cap.
    """
    r_kv, r_vw, r_wv, g0, clip, kvw, added, actual, capb = [], [], [], [], [], [], [], [], []
    scale_ratio, eff, fails, reff, caph = [], [], [], [], []
    eta = float(ttt.regime_etas(model, eta_base)[2].detach())
    for idx, y, meta in data:
        h = model.backbone(idx).detach()
        wp = meta["write_pos"]
        if write_chunks(wp, chunk) != {0}:
            raise ValueError("write_log expects every write in chunk 0")
        masks = position_masks(wp, y, idx.device)
        kv, vw = masks["kv"][:, :chunk], masks["vw"][:, :chunk]
        with torch.enable_grad():
            W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
            x = model.fast_input(h, idx) if model.frozen_emb is not None else None
            logits = model.read(ttt.fast_block(h[:, :chunk], model.head.ln, W,
                                               None if x is None else x[:, :chunk]))
            ce = ttt.token_ce(logits, y[:, :chunk])
            base = torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=True)
            g_kv = torch.autograd.grad(((kv * ce).sum(1) / chunk).sum(), W, retain_graph=True)
            extra, info = rule_extra(rule, ce, W, kv, vw, chunk, False)
        total = list(base) if extra is None else [g + e for g, e in zip(base, extra)]
        _, norm, clipped = ttt._clip_fast(total, g_clip)
        scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
        base_norm = _joint_norm(list(base))
        s_ce = torch.where(base_norm > g_clip, g_clip / base_norm.clamp_min(1e-24), torch.ones_like(norm))
        scale_ratio.append(scale / s_ce)
        # post-clip update change relative to CE: |s (g + e) - s_ce g|
        eff.append(eta * _joint_norm([_bcast(scale, a) * a - _bcast(s_ce, g) * g
                                      for a, g in zip(total, base)]))
        if info:
            fails.append(info["match_fail"].float())
            reff.append(info["r_eff"][kv > 0])
            if "cap_hit" in info:
                caph.append(info["cap_hit"])
        kv_term = list(g_kv) if rule.name not in ("rp", "rs") else [a + e for a, e in zip(g_kv, extra)]
        kvw.append(eta * scale * _joint_norm(kv_term))
        g0.append(norm)
        clip.append(clipped.float())
        added.append(info["added"] if info else torch.zeros_like(norm))
        actual.append(info["actual"] if info else torch.zeros_like(norm))
        if "cap_bound" in info:
            capb.append(info["cap_bound"].float())
        r = -torch.expm1(-ce)
        r_kv.append(kag._gather(r, wp).flatten())
        r_vw.append(r[vw > 0])  # true value->W transitions only
        r_wv.append(kag._gather(r, np.asarray(wp) - 2).flatten())
    rk = torch.cat(r_kv).float().cpu().double()  # MPS has no float64
    out = dict(r_kv_med=float(rk.median()), r_kv_mean=float(rk.mean()),
               r_kv_q10=float(torch.quantile(rk, .10)), r_kv_q90=float(torch.quantile(rk, .90)),
               r_kv_lt01=float((rk < .01).double().mean()), r_kv_lt02=float((rk < .02).double().mean()),
               r_kv_lt03=float((rk < .03).double().mean()),
               r_vw_med=float(torch.cat(r_vw).median()), r_wv_med=float(torch.cat(r_wv).median()),
               g0_med=float(torch.cat(g0).median()), clip_c0=float(torch.cat(clip).mean()),
               kv_write_med=float(torch.cat(kvw).median()),
               added_req_med=float(torch.cat(added).median()),
               added_actual_med=float(torch.cat(actual).median()))
    out["clip_scale_ratio_med"] = float(torch.cat(scale_ratio).median())
    out["eff_change_med"] = float(torch.cat(eff).median())
    if fails:
        out["match_fail"] = float(torch.cat(fails).mean())
        out["r_eff_med"] = float(torch.cat(reff).median())
    if caph:
        out["cap_hit"] = float(torch.cat(caph).mean())
    if capb:
        out["cap_bound"] = float(torch.cat(capb).mean())
    return {k: (round(v, 8) if math.isfinite(v) else None) for k, v in out.items()}


# --------------------------------------------------------------------------- per-event mediation


def corrupt_events(tokens: np.ndarray, meta: dict, probability: float, seed: int, step: int
                   ) -> tuple[np.ndarray, dict]:
    """``exp_ttt.corrupt_echo``'s copy-source channel at a given per-event probability.

    Each write event independently (probability ``probability``) gets its copy source
    at ``write_pos - 1`` replaced by another value; the kv label stays ``f(k)``. The
    draws are corrupt_echo's for a copy-noise arm with ``1 - q = probability``.
    """
    wp = meta["write_pos"]
    n_keys = wp.shape[1]
    rng = np.random.default_rng([seed, ttt.CORRUPTION_SEED, step])
    flags = rng.random(wp.shape) < probability
    out = tokens.copy()
    positions = np.full(wp.shape, -1, dtype=np.int64)
    rows, keys = np.nonzero(flags)
    value0 = 2 + n_keys
    positions[rows, keys] = wp[rows, keys] - 1
    original = tokens[rows, wp[rows, keys] + 1]
    out[rows, positions[rows, keys]] = value0 + (
        original - value0 + rng.integers(1, n_keys, len(rows))) % n_keys
    return out, dict(meta, corrupted=flags, corruption_pos=positions)


@torch.no_grad()
def kv_residual_w0(model: ttt.TTTModel, idx: torch.Tensor, y: torch.Tensor, write_pos: Any,
                   chunk: int) -> torch.Tensor:
    """(B, K) kv residuals ``1 - p_W0(v)`` of chunk 0, as the streaming write sees them."""
    if write_chunks(write_pos, chunk) != {0}:
        raise ValueError("kv_residual_w0 expects every write in chunk 0")
    h = model.backbone(idx)
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    logits = model.read(ttt.fast_block(h[:, :chunk], model.head.ln, model.per_sequence(idx.shape[0]),
                                       None if x is None else x[:, :chunk]))
    return kag._gather(-torch.expm1(-ttt.token_ce(logits, y[:, :chunk])), write_pos)


MEDIATION_R_MIN = 1e-6  # below this a residual is saturated: no drive ratio is attainable


def mediation_weights(r_clean: torch.Tensor, r_corr: torch.Tensor, flags: torch.Tensor,
                      cap: float) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Per-event kv weights (B, K) of the three mediation conditions and their validity.

    ``r_clean`` are the clean twin's residuals, ``r_corr`` the residuals in the stream
    actually evaluated (the corrupted sequence; for an uncorrupted event of that
    sequence, its own residual there). NAT 1. CLAMP: corrupted events
    ``min(r_clean / r_corr, cap)`` (their clean twin's drive), clean events 1. FLOOR:
    clean events ``clamp(r~ / r_corr, 1, cap)`` with ``r~`` the sequence's median
    corrupted-event residual (their drive raised to it in the same stream), corrupted
    events 1 (and 1 in a sequence without corrupted events). An event whose
    manipulation is undefined (non-finite residual, or the residual it is divided by
    below ``MEDIATION_R_MIN``) keeps weight 1 and is flagged ``invalid``; ``capped``
    flags events whose ratio exceeded ``cap``.
    """
    f = flags.bool()
    ones = torch.ones_like(r_corr)
    none = torch.zeros_like(f)
    fin = torch.isfinite(r_clean) & torch.isfinite(r_corr)
    ok = fin & (r_corr >= MEDIATION_R_MIN)
    den = r_corr.clamp_min(MEDIATION_R_MIN)
    ratio = r_clean / den
    clamp = torch.where(f & ok, ratio.clamp(max=cap), ones)
    rt = torch.nanmedian(torch.where(f & fin, r_corr, torch.full_like(r_corr, float("nan"))).cpu(), dim=1
                         ).values.to(r_corr.device)
    has = torch.isfinite(rt)[:, None]
    fratio = rt[:, None] / den
    floor = torch.where(~f & has & ok, fratio.clamp(1.0, cap), ones)
    valid = dict(NAT=dict(invalid=none, capped=none),
                 CLAMP=dict(invalid=f & ~ok, capped=f & ok & (ratio > cap)),
                 FLOOR=dict(invalid=~f & has & ~ok, capped=~f & has & ok & (fratio > cap)))
    return dict(NAT=ones, CLAMP=clamp, FLOOR=floor), valid


@torch.no_grad()
def mediation_eval(model: ttt.TTTModel, ctx: "Ctx") -> dict:
    """Recall of corrupted vs clean write events at fixed weights under NAT/CLAMP/FLOOR."""
    st, task, cs = ctx.st, ctx.task, ctx.cs
    acc: dict[str, list] = {k: [] for k in ("NAT", "CLAMP", "FLOOR")}
    clip: dict[str, list] = {k: [] for k in acc}
    wts: dict[str, list] = {k: [] for k in acc}
    drv: dict[str, list] = {k: [] for k in acc}
    bad: dict[str, list] = {k: [] for k in acc}
    capd: dict[str, list] = {k: [] for k in acc}
    nonfin: dict[str, int] = {k: 0 for k in acc}
    flags_all, qc_all, r0_all, rc_all = [], [], [], []
    for (idx_c, y_c, meta_c), (idx_0, y_0, meta_0) in ctx.med:
        wp = meta_c["write_pos"]
        r0 = kv_residual_w0(model, idx_0, y_0, wp, task.chunk)
        rc = kv_residual_w0(model, idx_c, y_c, wp, task.chunk)
        f = torch.as_tensor(meta_c["corrupted"], device=idx_c.device)
        qp = torch.as_tensor(np.asarray(meta_c["query_pos"]), device=idx_c.device)
        tgt = torch.as_tensor(np.asarray(meta_c["query_tok"]), device=idx_c.device)
        weights, valid = mediation_weights(r0, rc, f, cs.mediation_cap)
        for name, w in weights.items():
            cl: list = []
            nm: list = []
            logits, _, _ = chr_stream(model, idx_c, y_c, wp, ctx.eta, st.g_clip, task.chunk, CE,
                                      kv_weights=w, clip_out=cl, norm_out=nm)
            lq = logits.gather(1, qp[..., None].expand(-1, -1, logits.shape[-1]))
            acc[name].append((lq.argmax(-1) == tgt).float())
            clip[name].append(torch.cat(cl))
            wts[name].append(w)
            drv[name].append(w * rc)  # achieved pre-clip kv drive (residual units)
            bad[name].append(valid[name]["invalid"])
            capd[name].append(valid[name]["capped"])
            nonfin[name] += int((~torch.isfinite(torch.cat(nm))).sum())
        flags_all.append(f)
        qc_all.append(qp // task.chunk)
        r0_all.append(r0)
        rc_all.append(rc)
    f, qc = torch.cat(flags_all).bool(), torch.cat(qc_all)
    r0, rc = torch.cat(r0_all), torch.cat(rc_all)
    out: dict = dict(p=cs.mediation_p, n_corrupted=int(f.sum()), n_clean=int((~f).sum()),
                     r_clean_med=float(r0.median()), r_corr_med=float(rc[f].median()) if f.any() else None,
                     r_corr_clean_events_med=float(rc[~f].median()))
    for name in acc:
        hit, w, drive = torch.cat(acc[name]), torch.cat(wts[name]), torch.cat(drv[name])
        row = dict(acc_corr=float(hit[f].mean()), acc_clean=float(hit[~f].mean()),
                   clip_c0=float(torch.cat(clip[name]).mean()),
                   w_corr_med=float(w[f].median()), w_clean_med=float(w[~f].median()),
                   drive_corr_med=float(drive[f].median()) if f.any() else None,
                   drive_clean_med=float(drive[~f].median()),
                   n_invalid_events=int(torch.cat(bad[name]).sum()),
                   n_capped_events=int(torch.cat(capd[name]).sum()),
                   n_nonfinite_norms=nonfin[name])
        row["d"] = row["acc_corr"] - row["acc_clean"]
        for c in (cs.chunk_early, cs.chunk_late):
            sel = qc == c
            if bool((sel & f).any()) and bool((sel & ~f).any()):
                row[f"d{c}"] = float(hit[sel & f].mean() - hit[sel & ~f].mean())
        out[name] = row
    return out


# --------------------------------------------------------------------------- training


@dataclasses.dataclass
class ChrSettings:
    log_every: int
    probe_every: int
    cross_every: int
    fork_arms: list[str]
    fork_acc: float
    forks: dict[str, list[Rule]]
    cross_rules: dict[str, list[Rule]]
    final_rules: dict[str, list[Rule]]
    chunk_early: int
    chunk_late: int
    ckpt_steps: list[int]
    ckpt_dense_arms: list[str]
    ckpt_steps_sparse: list[int]
    branch_ckpt_every: int
    branch_ckpt_tags: list[str]
    mediation_p: float
    mediation_cap: float

    def rules(self, which: str, arm: str) -> list[Rule]:
        """The ``forks`` / ``cross_rules`` / ``final_rules`` of ``arm``."""
        table = getattr(self, which)
        return list(table.get(arm, table.get("*", [])))


def chr_settings(cfg: dict) -> ChrSettings:
    merged = {**CHR_DEFAULTS, **{k: v for k, v in cfg.items() if k in CHR_DEFAULTS}}
    cap = float(merged["rp_cap"])
    cs = ChrSettings(
        log_every=int(merged["log_every"]), probe_every=int(merged["probe_every"]),
        cross_every=int(merged["cross_every"]), fork_arms=list(merged["fork_arms"]),
        fork_acc=float(merged["fork_acc"]),
        forks=_rules_by_arm(merged["forks"], cap),
        cross_rules=_rules_by_arm(merged["cross_rules"], cap),
        final_rules=_rules_by_arm(merged["final_rules"], cap),
        chunk_early=int(merged["chunk_early"]), chunk_late=int(merged["chunk_late"]),
        ckpt_steps=[int(s) for s in merged["ckpt_steps"]],
        ckpt_dense_arms=[str(a) for a in merged["ckpt_dense_arms"]],
        ckpt_steps_sparse=[int(s) for s in (merged["ckpt_steps"] if merged["ckpt_steps_sparse"] is None
                                             else merged["ckpt_steps_sparse"])],
        branch_ckpt_every=int(merged["branch_ckpt_every"]),
        branch_ckpt_tags=[str(t) for t in merged["branch_ckpt_tags"]],
        mediation_p=float(merged["mediation_p"]), mediation_cap=float(merged["mediation_cap"]))
    if any(a not in FORK_ARMS_ALLOWED for a in cs.fork_arms):
        raise ValueError(f"fork arms must be in {FORK_ARMS_ALLOWED}")
    if cs.probe_every % cs.log_every or cs.cross_every % cs.log_every:
        raise ValueError("probe_every and cross_every must be multiples of log_every")
    for which in ("forks", "cross_rules", "final_rules"):
        for arm, rules in getattr(cs, which).items():
            if arm != "*" and arm not in CHR_ARMS:
                raise ValueError(f"{which}: unknown arm {arm}")
            if len({r.tag for r in rules}) != len(rules):
                raise ValueError(f"duplicate {which} for {arm}")
    for arm in cs.fork_arms:
        if not cs.rules("forks", arm):
            raise ValueError(f"fork arm {arm} has no fork rules")
    if not 0.0 <= cs.mediation_p < 1.0 or cs.mediation_cap < 1.0:
        raise ValueError("mediation_p must be in [0, 1) and mediation_cap >= 1")
    return cs


def _sha1(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def _emit(kind: str, payload: dict) -> None:
    print(f"{kind} " + json.dumps(payload, allow_nan=False, default=_json_default), flush=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _finite(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


@dataclasses.dataclass
class Ctx:
    """Everything a training segment needs besides the model and its optimizer."""

    task: ttt.Task
    arm: str
    eta: float
    st: ttt.Settings
    cs: ChrSettings
    dev: torch.device
    seed: int
    quick: list
    probe: list
    probe_corrupted: list | None
    ev: list
    ev_meta: dict
    ident: str
    med: list | None = None


def train_segment(model: ttt.TTTModel, opt: torch.optim.AdamW, rng: np.random.Generator,
                  ctx: Ctx, rule: Rule, branch: str, start: int, allow_fork: bool,
                  nonfinite0: int = 0) -> dict:
    """Train ``[start, steps)``; log, checkpoint and (main CE run only) fork on the way.

    The per-step arithmetic is ``exp_ttt.train_arm``'s for the arms used here
    (no outer mask, no lin selector, no Kamin, no kv eta, ``spike_skip`` 0). A branch
    inherits the main run's non-finite count at the fork (``nonfinite0``), so the
    divergence rule (> 5 % non-finite steps after step 20) is the main run's, continued.
    """
    st, cs, task, arm, dev = ctx.st, ctx.cs, ctx.task, ctx.arm, ctx.dev
    n_steps = int(task.steps)
    warmup = ttt.arm_warmup(arm, st)
    total = st.lr_total or n_steps
    if total < n_steps or st.spike_skip:
        raise ValueError("CHR needs lr_total >= steps and spike_skip 0")
    stats = ttt.GradStats()
    logs, probes, saved, forks = [], [], [], []
    nonfinite, diverged, times = 0, False, []
    fork_done = not allow_fork
    first_sha = None
    after_fork: dict | None = None
    diag: dict = {}
    late = (cs.chunk_early, cs.chunk_late)
    tag = f"{ctx.ident}|{branch}"
    cross_rules = [r for r in [CE, *cs.rules("cross_rules", arm)] if r != rule]
    wall0 = time.time()
    log_s = fork_s = 0.0  # wall time in logging / in the fork branches (phase budget)
    step = start - 1
    for step in range(start, n_steps):
        ttt.set_kamin_regime(model, arm, step, st)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * ttt.lr_scale(step, warmup, total, st.lr_floor)
        raw, meta = task.raw(rng, st.batch)
        if first_sha is None:
            first_sha = _sha1(raw)
        if after_fork is not None and "main_next_batch_sha1" not in after_fork:
            after_fork["main_next_batch_sha1"] = _sha1(raw)
        raw, meta = ttt.corrupt_echo(raw, meta, arm, ctx.seed, step, rf_probability=st.rf_probability)
        idx, y, meta = task.tensors(raw, meta, dev)
        ttt._sync(dev)
        t0 = time.perf_counter()
        loss = chr_arm_loss(model, arm, idx, y, ctx.eta, st.g_clip, task.chunk, stats, meta, rule, diag)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss).item():
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(ttt.clipped_params(model), st.grad_clip)
            rho_norm = (model.rho.grad.norm() if model.rho.grad is not None
                        else torch.zeros((), device=norm.device)).to(norm.device)
            finite = bool(torch.isfinite(torch.stack([norm.float(), rho_norm.float()])).all())
            if finite:
                opt.step()
        else:
            finite = False
        if not finite:
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            print(f"  WARNING {tag}: non-finite step {step + 1}", flush=True)
        ttt._sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
        done = step + 1
        if done % cs.log_every == 0 or done == n_steps:
            t_log = time.perf_counter()
            row = dict(cell_id=ctx.ident, branch=branch, rule=rule.tag, step=done,
                       loss=round(float(loss.detach()), 6),
                       quick=quick_log(model, ctx.quick, ctx.eta, st.g_clip, task.chunk, rule, late),
                       write=write_log(model, ctx.probe, ctx.eta, st.g_clip, task.chunk, rule))
            if diag.get("added"):
                clipped = torch.cat(diag["clipped"]).float()
                req = torch.cat(diag["added"]).detach().cpu().double()  # MPS has no float64
                act = torch.cat(diag["actual"]).detach().cpu().double()
                row["train_added_req_med"] = round(float(req.median()), 8)
                row["train_added_req_sum"] = round(float(req.sum()), 8)  # requested (rp/rs) norm
                row["train_added_actual_sum"] = round(float(act.sum()), 8)  # applied extra norm
                row["train_clip_c0"] = round(float(clipped.mean()), 6)
                row["train_n"] = int(clipped.numel())  # sequences x steps in this window
                row["train_clip_count"] = int(clipped.sum())
                row["train_match_fail_count"] = int(torch.cat(diag["match_fail"]).sum())
                if diag.get("cap_bound"):
                    cb = torch.cat(diag["cap_bound"]).float()
                    row["train_cap_bound"] = round(float(cb.mean()), 6)
                    row["train_cap_bound_count"] = int(cb.sum())
                if diag.get("cap_hit"):
                    row["train_cap_hit_mean"] = round(float(torch.cat(diag["cap_hit"]).mean()), 6)
                diag.clear()
            if done % cs.cross_every == 0 or done == n_steps:
                row["cross"] = {r.tag: quick_log(model, ctx.quick, ctx.eta, st.g_clip, task.chunk, r,
                                                 late, geometry_on=False) for r in cross_rules}
                if rule != CE:  # the CE-reference write at the same state (manipulation check)
                    row["write_ce"] = write_log(model, ctx.probe, ctx.eta, st.g_clip, task.chunk, CE)
            if done % cs.probe_every == 0 or done == n_steps:
                if ctx.probe_corrupted is not None:  # the noise arm's training-like residuals
                    row["write_corrupted"] = write_log(model, ctx.probe_corrupted, ctx.eta, st.g_clip,
                                                       task.chunk, rule)
                p = ttt.r4_probe(model, task, ctx.probe, ctx.eta, st.g_clip, arm=arm)
                if ctx.probe_corrupted is not None:
                    p.update(ttt.kv_w0_stats(model, task, ctx.probe_corrupted, arm))
                # the inherited probe measures CE-rule quantities whatever the branch trains with
                p = dict(cell_id=ctx.ident, branch=branch, step=done, train_rule=rule.tag,
                         measurement_rule="ce", **p)
                probes.append(p)
                _emit("CHR_PROBE", _finite(p))
            logs.append(row)
            _emit("CHR_LOG", _finite(row))
            log_s += time.perf_counter() - t_log
            if not fork_done and row["quick"]["acc"] >= cs.fork_acc and done < n_steps:
                fork_done = True
                t_fork = time.perf_counter()
                forks = run_forks(model, opt, rng, ctx, done, nonfinite0 + nonfinite)
                fork_s += time.perf_counter() - t_fork
                after_fork = {}
        if _save_here(cs, arm in cs.ckpt_dense_arms, branch, rule, start, done, n_steps):
            saved.append(ttt.save_checkpoint(model, f"{ctx.ident}__{branch}__step{done}"))
        if done >= 20 and (nonfinite0 + nonfinite) / done > 0.05:
            diverged = True
            print(f"  WARNING {tag}: diverged after {done} steps", flush=True)
            break
    if forks:
        nxt = (after_fork or {}).get("main_next_batch_sha1")
        for b in forks:
            b["main_batch_identical"] = nxt is not None and nxt == b.get("first_batch_sha1")
    return dict(logs=logs, probes=probes, saved=saved, forks=forks, nonfinite=nonfinite,
                diverged=diverged, first_batch_sha1=first_sha, steps_done=step + 1,
                segment_wall_s=round(time.time() - wall0, 1), log_wall_s=round(log_s, 1),
                fork_wall_s=round(fork_s, 1),
                ms_step=round(float(np.median(times[min(100, len(times) // 2):] or times or [0])), 3),
                grad=stats.summary())


def _save_here(cs: ChrSettings, dense: bool, branch: str, rule: Rule, start: int, done: int,
               n_steps: int) -> bool:
    """Main: the arm's step list + final. Branches: final; ``branch_ckpt_tags`` also every k."""
    if branch == "main":
        return done in (cs.ckpt_steps if dense else cs.ckpt_steps_sparse) or done == n_steps
    if done == n_steps:
        return True
    return (rule.tag in cs.branch_ckpt_tags and cs.branch_ckpt_every > 0
            and (done - start) % cs.branch_ckpt_every == 0)


def _state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _deep_equal(a: Any, b: Any) -> bool:
    """Recursive equality of (optimizer) state: tensors by ``torch.equal``, the rest by ``==``."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, z) for x, z in zip(a, b))
    return a == b


def run_forks(model: ttt.TTTModel, opt: torch.optim.AdamW, rng: np.random.Generator,
              ctx: Ctx, at: int, nonfinite0: int = 0) -> list[dict]:
    """Clone (model, AdamW, data stream) at ``at`` and run every configured branch.

    Branches run to the last step before the main run continues; each starts from
    the identical state and the identical future batch stream.
    """
    state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    rng_state = copy.deepcopy(rng.bit_generator.state)
    path = Path("checkpoints")
    path.mkdir(exist_ok=True)
    fork_file = path / f"{ctx.ident.replace('/', '__')}__fork_state_step{at}.pt"
    torch.save(dict(model={k: v.detach().cpu() for k, v in state.items()},
                    opt=_to_cpu(opt_state), rng=rng_state, step=at), fork_file)
    probe = rng_peek(rng_state, ctx.task, ctx.st.batch)
    rules = ctx.cs.rules("forks", ctx.arm)
    fork_write = {r.tag: write_log(model, ctx.probe, ctx.eta, ctx.st.g_clip, ctx.task.chunk, r)
                  for r in [CE, *rules]}
    late = (ctx.cs.chunk_early, ctx.cs.chunk_late)
    quick_at_fork = {r.tag: quick_log(model, ctx.quick, ctx.eta, ctx.st.g_clip, ctx.task.chunk, r, late,
                                      geometry_on=False) for r in [CE, *rules]}
    rec = dict(cell_id=ctx.ident, step=at, file=str(fork_file), next_batch_sha1=probe,
               nonfinite_at_fork=nonfinite0, write_at_fork=fork_write, quick_at_fork=quick_at_fork)
    if ctx.probe_corrupted is not None:  # the noise arm: the same on the training-like probe
        rec["write_at_fork_corrupted"] = {
            r.tag: write_log(model, ctx.probe_corrupted, ctx.eta, ctx.st.g_clip, ctx.task.chunk, r)
            for r in [CE, *rules]}
    _emit("CHR_FORK", _finite(rec))
    out = []
    for rule in rules:
        branch = f"fork-{rule.tag}" + ("-twin" if rule.name == "ce" else "")
        m2 = o2 = None
        try:
            m2 = ttt.build_model(ctx.task, ctx.st, ctx.dev, ctx.seed, ctx.arm)
            m2.load_state_dict(state)
            o2 = ttt.build_optim(m2, ctx.st)
            o2.load_state_dict(copy.deepcopy(opt_state))
            r2 = np.random.default_rng()
            r2.bit_generator.state = copy.deepcopy(rng_state)
            identical = (_state_equal(m2.state_dict(), model.state_dict())
                         and _deep_equal(o2.state_dict(), opt.state_dict()))
            res = train_segment(m2, o2, r2, ctx, rule, branch, at, allow_fork=False, nonfinite0=nonfinite0)
            res.update(branch=branch, rule=rule.tag, fork_step=at, state_identical=identical,
                       batch_identical=res["first_batch_sha1"] == probe)
            m2.eval()
            t_fin = time.perf_counter()
            res["final"] = final_eval(m2, ctx, rule, branch)
            res["final_wall_s"] = round(time.perf_counter() - t_fin, 1)
            out.append(_branch_summary(res))
            _emit("CHR_BRANCH", _finite(out[-1]))
        except Exception as exc:
            traceback.print_exc()
            print(f"CHR_BRANCH_ERROR {ctx.ident} {branch}", flush=True)
            out.append(dict(branch=branch, rule=rule.tag, fork_step=at, error=repr(exc)))
        finally:
            del m2, o2  # free the branch's model and optimizer before the next one
            if ctx.dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


def _to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def rng_peek(rng_state: dict, task: ttt.Task, batch: int) -> str:
    """SHA1 of the next raw batch a generator in ``rng_state`` would draw."""
    r = np.random.default_rng()
    r.bit_generator.state = copy.deepcopy(rng_state)
    return _sha1(task.raw(r, batch)[0])


def _branch_summary(res: dict) -> dict:
    keep = ("branch", "rule", "fork_step", "state_identical", "batch_identical", "nonfinite",
            "diverged", "steps_done", "ms_step", "segment_wall_s", "log_wall_s", "final_wall_s",
            "grad", "saved", "final",
            "first_batch_sha1")
    return {k: res[k] for k in keep if k in res}


@torch.no_grad()
def final_eval(model: ttt.TTTModel, ctx: Ctx, own: Rule, branch: str) -> dict:
    """Full-eval accuracy under the own rule, CE and the arm's eval-only rules, plus the
    intervention (own rule). Branches and main runs get the same rule set (common-rule
    contrasts between training rules)."""
    cs, st, task = ctx.cs, ctx.st, ctx.task
    late = (cs.chunk_early, cs.chunk_late)
    rules = [own] + [r for r in [CE, *cs.rules("final_rules", ctx.arm)] if r != own]
    out: dict = {}
    for r in rules:
        hits, cids, parts_all, tgts, gate, agree = [], [], [], [], 0.0, []
        for idx, y, meta in ctx.ev:
            logits, captured, tens = chr_stream(model, idx, y, meta["write_pos"], ctx.eta, st.g_clip,
                                                task.chunk, r, capture=late)
            qp = torch.as_tensor(np.asarray(meta["query_pos"]), device=idx.device)
            tgt = torch.as_tensor(np.asarray(meta["query_tok"]), device=idx.device)
            lq = logits.gather(1, qp[..., None].expand(-1, -1, logits.shape[-1]))
            hits.append((lq.argmax(-1) == tgt).float())
            cids.append(qp // task.chunk)
            if r == own:
                parts = read_parts(model, tens, captured, meta["query_pos"], task.chunk)
                rec_logits = model.read(parts["v"])
                gate = max(gate, float((rec_logits - lq).abs().max() / max(1.0, float(lq.abs().max()))))
                agree.append((rec_logits.argmax(-1) == lq.argmax(-1)).float())
                parts_all.append(parts)
                tgts.append(tgt)
        hit, cid = torch.cat(hits), torch.cat(cids)
        row = dict(acc=round(float(hit.mean()), 6),
                   **{f"acc{c}": round(float(hit[cid == c].mean()), 6)
                      for c in torch.unique(cid).tolist()})
        if r == own:
            merged = {k: torch.cat([p[k] for p in parts_all]) for k in parts_all[0]}
            row["geo"] = geometry(model, merged, late)
            row["decomp_gate_rel"] = gate
            row["decomp_pred_agree"] = float(torch.cat(agree).mean())
            row["decomp_gate_ok"] = gate <= 1e-3 and row["decomp_pred_agree"] == 1.0
            row["intervention"] = intervention(model, merged, torch.cat(tgts), cs.chunk_early,
                                               cs.chunk_late, RND_SEED + ctx.seed)
        out[r.tag] = row
    return out


def run_chr_cell(cfg: dict, cell: dict, dev: torch.device) -> dict:
    """Train, log and finalise one cell (its forks run inside the main training)."""
    st = ttt._settings(cfg, cell["task"])
    cs = chr_settings(cfg)
    task = ttt._task(st, cell["task"])
    arm, seed, eta = cell["arm"], int(cell["seed"]), float(cell["eta_base"])
    if arm not in CHR_ARMS:
        raise ValueError(f"CHR arm must be one of {CHR_ARMS}")
    ttt.check_arm_task(arm, task.mode)
    if task.steps != int(cell.get("steps", task.steps)):
        raise ValueError("per-cell steps are not supported in CHR")
    ident = ttt._cell_id(cell, task.steps)
    ev_tokens, ev_meta = ttt.fixed_set(task, st, "eval")
    ev = ttt.eval_batches(task, ev_tokens, ev_meta, dev, st.eval_batch)
    quick = ttt.eval_batches(task, *ttt.fixed_set(task, st, "quick"), dev, st.eval_batch)
    probe = ttt.eval_batches(task, *task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]), st.n_probe),
                             dev, st.eval_batch)
    probe_c = None
    if arm in ttt.NOISE_ARMS:
        raw, pm = task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]), st.n_probe)
        raw, pm = ttt.corrupt_echo(raw, pm, arm, ttt.CORRUPTION_EVAL_SEED, 2,
                                   rf_probability=st.rf_probability)
        probe_c = ttt.eval_batches(task, raw, pm, dev, st.eval_batch)
    med = None
    if cs.mediation_p > 0:
        raw_c, meta_c = corrupt_events(ev_tokens, ev_meta, cs.mediation_p, ttt.CORRUPTION_EVAL_SEED,
                                       MEDIATION_STEP)
        med = list(zip(ttt.eval_batches(task, raw_c, meta_c, dev, st.eval_batch), ev))
    ctx = Ctx(task, arm, eta, st, cs, dev, seed, quick, probe, probe_c, ev, ev_meta, ident, med)
    model = ttt.build_model(task, st, dev, seed, arm)
    if st.eta_lr != 0.0:
        raise ValueError("CHR runs at fixed eta (eta_lr = 0)")
    opt = ttt.build_optim(model, st)
    rng = np.random.default_rng([seed, ttt.TRAIN_SEED])
    start = time.time()
    cuda = dev.type == "cuda"
    if cuda:  # per-cell peaks (one cell at a time per worker process)
        torch.cuda.reset_peak_memory_stats(dev)
    res = train_segment(model, opt, rng, ctx, CE, "main", 0, allow_fork=arm in cs.fork_arms)
    mem = {}
    if cuda:
        mem = dict(cuda_peak_alloc_train_mib=round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
                   cuda_peak_reserved_train_mib=round(torch.cuda.max_memory_reserved(dev) / 2**20, 1))
        torch.cuda.reset_peak_memory_stats(dev)
    model.eval()
    t_fin = time.perf_counter()
    res["final"] = final_eval(model, ctx, CE, "main")
    t_med = time.perf_counter()
    res["mediation"] = mediation_eval(model, ctx) if ctx.med else None
    walls = dict(train_wall_s=res["segment_wall_s"], log_wall_s=res["log_wall_s"], fork_wall_s=res["fork_wall_s"],
                 final_wall_s=round(t_med - t_fin, 1), mediation_wall_s=round(time.perf_counter() - t_med, 1))
    if cuda:
        mem.update(cuda_peak_alloc_final_mib=round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
                   cuda_peak_reserved_final_mib=round(torch.cuda.max_memory_reserved(dev) / 2**20, 1))
    record = dict(cell_id=ident, task=task.name, arm=arm, seed=seed, eta_base=eta, device=str(dev),
                  steps=task.steps, wall_s=round(time.time() - start, 1), tf32=bool(st.tf32),
                  **{k: res[k] for k in ("nonfinite", "diverged", "first_batch_sha1", "ms_step",
                                         "grad", "saved", "final", "mediation", "forks", "steps_done")},
                  phase_wall_s=walls, memory=mem)
    record["forked"] = bool(res["forks"])
    _emit("CHR_CELL", _finite(record))
    return record


# --------------------------------------------------------------------------- sanity


def chr_sanity() -> dict:
    """CPU identities: CE paths bit-identical to exp_ttt; rp/nc(r_ref 0) and rs/ns(r_ref 1) == ce;
    nc/ns norm match; the event-weight stream reproduces ce (ones) and rp (rp weights)."""
    dev = torch.device("cpu")
    st = ttt._settings(dict(d_model=16, n_layers=2, n_heads=2, backbone_mult=2, fast_mult=2,
                            window=8, chunk=16, n_keys=4, T=96, n_eval=2, n_quick_eval=2,
                            eval_batch=2, n_probe=2, batch=2, steps=3, eta_lr=0.0))
    task = ttt._task(st, "echo")
    idx, y, meta = task.tensors(*task.raw(np.random.default_rng(5), 2), dev)
    out = {}

    def grads_of(fn, model):
        model.zero_grad(set_to_none=True)
        loss = fn(model)
        loss.backward()
        return loss.detach(), [p.grad.detach().clone() if p.grad is not None else None
                               for p in model.parameters()]

    def build():
        m = ttt.build_model(task, st, dev, 3, "e2e1emb")
        with torch.no_grad():
            m.head.W2.normal_(0, 0.3)
        return m

    a, b = build(), build()
    la, ga = grads_of(lambda m: ttt.arm_loss(m, "e2e1emb", idx, y, 10.0, st.g_clip, task.chunk,
                                             ttt.GradStats(), meta=meta), a)
    def inner(rule):
        def fn(m):
            h = m.backbone(idx)
            return chr_sequential_inner(m, h, y, idx, meta["write_pos"], ttt.regime_etas(m, 10.0),
                                        st.g_clip, task.chunk, ttt.GradStats(), rule)
        return fn

    def same_as_ref(lb, gb):
        return bool(torch.equal(la, lb)) and all(
            (x is None and z is None) or torch.equal(x, z) for x, z in zip(ga, gb))

    lc, gc = grads_of(lambda m: ttt.arm_loss(m, "e2e1emb_cn97", idx, y, 10.0, st.g_clip, task.chunk,
                                             ttt.GradStats(), meta=meta), build())
    out["cn97_ce_equals_e2e1emb"] = dict(passed=same_as_ref(lc, gc))
    for name, rule in (("inner_ce_equals_ttt", CE), ("rp0_equals_ce", Rule("rp", 0.0)),
                       ("nc0_equals_ce", Rule("nc", 0.0)), ("rs100_equals_ce", Rule("rs", 1.0)),
                       ("ns100_equals_ce", Rule("ns", 1.0))):
        out[name] = dict(passed=same_as_ref(*grads_of(inner(rule), b)))
    for name, rule in (("rp_changes_gradient", Rule("rp", .999)), ("rs_changes_gradient", Rule("rs", .5)),
                       ("ns_changes_gradient", Rule("ns", .5))):
        out[name] = dict(passed=not same_as_ref(*grads_of(inner(rule), b)))
    model = build()
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(2)]
    h = model.backbone(idx).detach()
    x = model.fast_input(h, idx)
    ce = ttt.token_ce(model.read(ttt.fast_block(h[:, :16], model.head.ln, W, x[:, :16])), y[:, :16])
    masks = position_masks(meta["write_pos"], y, dev)
    # an untrained model predicts v with p ~ 1/V, so only an r_min near 1 adds a term
    _, rp = rule_extra(Rule("rp", .999), ce, W, masks["kv"][:, :16], masks["vw"][:, :16], 16, True)
    extra, nc = rule_extra(Rule("nc", .999), ce, W, masks["kv"][:, :16], masks["vw"][:, :16], 16, True)
    rel = float(((_joint_norm(extra) - rp["added"]).abs() / rp["added"].clamp_min(1e-30)).max())
    out["nc_norm_match"] = dict(passed=rel <= 1e-5 and bool((rp["added"] > 0).all()), rel=rel)
    _, rs = rule_extra(Rule("rs", .5), ce, W, masks["kv"][:, :16], masks["vw"][:, :16], 16, True)
    extra, ns = rule_extra(Rule("ns", .5), ce, W, masks["kv"][:, :16], masks["vw"][:, :16], 16, True)
    rel = float(((_joint_norm(extra) - rs["added"]).abs() / rs["added"].clamp_min(1e-30)).max())
    out["ns_norm_match"] = dict(passed=rel <= 1e-5 and bool((rs["added"] > 0).all())
                                and not bool(ns["cap_bound"].any()), rel=rel)
    ref = ttt.eval_sequential(model, idx, y, ttt.regime_etas(model, 10.0), st.g_clip, task.chunk,
                              ttt.GradStats())
    mine, captured, tens = chr_stream(model, idx, y, meta["write_pos"], 10.0, st.g_clip, task.chunk, CE,
                                      capture=tuple(range(task.T // task.chunk)))
    out["stream_ce_equals_eval_sequential"] = dict(passed=bool(torch.equal(ref, mine)))
    wp = meta["write_pos"]
    ones = torch.ones(wp.shape)
    got, _, _ = chr_stream(model, idx, y, wp, 10.0, st.g_clip, task.chunk, CE, kv_weights=ones)
    out["event_weights_ones_equal_ce"] = dict(passed=bool(torch.equal(got, mine)))
    rule = Rule("rp", .999)
    _, w = rule_weights(-torch.log1p(-kv_residual_w0(model, idx, y, wp, task.chunk)), rule)
    rp_ref, _, _ = chr_stream(model, idx, y, wp, 10.0, st.g_clip, task.chunk, rule)
    got, _, _ = chr_stream(model, idx, y, wp, 10.0, st.g_clip, task.chunk, CE, kv_weights=w)
    diff = float((got - rp_ref).abs().max())
    out["event_weights_reproduce_rp"] = dict(passed=diff <= 1e-5 and not bool(torch.equal(rp_ref, mine)),
                                             max_abs=diff)
    parts = read_parts(model, tens, captured, meta["query_pos"], task.chunk)
    qp = torch.as_tensor(meta["query_pos"])
    lq = mine.gather(1, qp[..., None].expand(-1, -1, mine.shape[-1]))
    rec = model.read(parts["v"])
    diff = float((rec - lq).abs().max().detach())
    agree = float((rec.argmax(-1) == lq.argmax(-1)).float().mean())
    out["decomposition_identity"] = dict(passed=diff <= 1e-4 and agree == 1.0, max_abs=diff, pred_agree=agree)
    return out


# --------------------------------------------------------------------------- runner


def _worker(index: int, cfg: dict, dev_name: str, jobs: mp.Queue, results: mp.Queue, lock) -> None:
    sys.stdout = ttt._LockedLines(sys.stdout, lock)
    sys.stderr = ttt._LockedLines(sys.stderr, lock)
    ttt.configure_tf32(bool(ttt._settings(cfg).tf32), str(index))
    ttt._mark_process(f"{cfg.get('process_name', 'TTM-chr-01')}-w{index}")
    dev = ttt._device(dev_name)
    torch.set_num_threads(int(cfg.get("threads_per_worker", 2)))
    if dev.type == "mps" and cfg.get("mps_memory_fraction") is not None:
        torch.mps.set_per_process_memory_fraction(float(cfg["mps_memory_fraction"]))
    while True:
        cell = jobs.get()
        if cell is None:
            break
        deadline = cfg["_deadline"]
        ident = ttt._cell_id(cell, ttt._settings(cfg).steps)
        try:
            if time.time() >= deadline:
                print(f"!! budget guard: skipping {ident}", flush=True)
                results.put((cell, dict(cell_id=ident, skipped=True, reason="budget")))
                continue
            results.put((cell, run_chr_cell(cfg, cell, dev)))
        except Exception as exc:
            traceback.print_exc()
            rec = dict(cell_id=ident, arm=cell.get("arm"), seed=cell.get("seed"), error=repr(exc))
            print("CHR_CELL_ERROR " + json.dumps(rec), flush=True)
            results.put((cell, rec))
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# administrative settings a registered retry may change (the scientific fingerprint excludes them)
ADMIN_KEYS = ("cells", "process_name", "workers", "max_minutes", "hard_grace_minutes", "threads_per_worker")


def protocol(cfg: dict, dev: torch.device) -> dict:
    """Provenance: config hashes, optional ``plan_file`` hash, analyzer/code hashes, device and software."""
    root = Path(__file__).resolve().parents[2]
    cfg_file = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".toml") else None
    science = {k: v for k, v in cfg.items() if k not in ADMIN_KEYS}
    out = dict(protocol=str(cfg.get("protocol", "unnamed")),
               config_file=None if cfg_file is None else str(cfg_file),
               config_file_sha256=None if cfg_file is None else _file_sha256(cfg_file),
               config_resolved_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()
                                                     ).hexdigest(),
               science_sha256=hashlib.sha256(json.dumps(science, sort_keys=True, default=str).encode()
                                             ).hexdigest(),
               plan_sha256=_file_sha256(root / cfg["plan_file"]) if cfg.get("plan_file") else None,
               analyzer_sha256=_file_sha256(root / "scripts" / "analyze_chr.py"),
               exp_chr_sha256=_file_sha256(Path(__file__).resolve()),
               exp_ttt_sha256=_file_sha256(Path(ttt.__file__).resolve()),
               device=str(dev), torch=torch.__version__, python=sys.version.split()[0])
    if dev.type == "cuda":
        out.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    return out


def run(cfg: dict, seed: int) -> dict:
    """Entry point for ``experiment = "chr"``."""
    st = ttt._settings(cfg)
    ttt._mark_process(st.process_name)
    torch.set_num_threads(int(cfg.get("threads_per_worker", 2)))
    ttt.configure_tf32(st.tf32, "main")
    cs = chr_settings(cfg)
    dev = ttt._device(st.device)
    if cfg.get("require_cuda") and dev.type != "cuda":
        raise RuntimeError(f"protocol {cfg.get('protocol')} requires CUDA, got {dev}")
    for key in ("log_every", "probe_every", "cross_every"):
        if getattr(cs, key) <= 0:
            raise ValueError(f"{key} must be positive")
    print("CHR_PROTOCOL " + json.dumps(protocol(cfg, dev)), flush=True)
    print("CHR_SETTINGS " + json.dumps(dataclasses.asdict(cs), default=str), flush=True)
    sanity = {**chr_sanity(), **{f"r4_{k}": v for k, v in ttt.r4_sanity().items()}}
    print("CHR_SANITY " + json.dumps(_finite(sanity), default=str), flush=True)
    if not all(v["passed"] for v in sanity.values()):
        raise RuntimeError("CHR sanity failed")
    cells = list(cfg["cells"])
    ids = [ttt._cell_id(c, st.steps) for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cell ids")
    for c in cells:
        if c["arm"] not in CHR_ARMS or "eta_base" not in c:
            raise ValueError(f"bad CHR cell {c}")
        extra = set(c) - {"task", "seed", "arm", "eta_base"}
        if extra:
            raise ValueError(f"unsupported cell overrides {sorted(extra)} in {c}")
    print("CHR_MANIFEST " + json.dumps(dict(cells=ids, steps=st.steps, workers=st.workers)), flush=True)
    ctx = mp.get_context("spawn")
    jobs, results, lock = ctx.Queue(), ctx.Queue(), ctx.Lock()
    for c in cells:
        jobs.put(c)
    for _ in range(st.workers):
        jobs.put(None)
    t_start = time.time()
    shared = {**cfg, "_deadline": t_start + st.max_minutes * 60}
    # watchdog: the start-of-cell guard cannot stop a running cell, so hard-stop at
    # max_minutes + hard_grace_minutes; unfinished cells are reported as incomplete
    hard_deadline = t_start + (st.max_minutes + float(cfg.get("hard_grace_minutes", 60))) * 60
    procs = [ctx.Process(target=_worker, args=(i, shared, str(dev), jobs, results, lock))
             for i in range(st.workers)]
    for p in procs:
        p.start()
    collected: dict[str, dict] = {}
    watchdog = False
    while len(collected) < len(cells):
        try:
            cell, rec = results.get(timeout=30)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
            if time.time() > hard_deadline:
                watchdog = True
                print("!! CHR watchdog: hard deadline reached; terminating workers", flush=True)
                for p in procs:
                    p.terminate()
                break
            continue
        collected[ttt._cell_id(cell, st.steps)] = rec
    for p in procs:
        p.join(timeout=60)
    codes = [p.exitcode for p in procs]
    print("\n== CHR CELLS ==", flush=True)
    summary = {}
    for ident in ids:
        rec = collected.get(ident, dict(cell_id=ident, error="watchdog stop" if watchdog else
                                        "worker died before returning this cell"))
        fin = (rec.get("final") or {}).get("ce", {})
        branch_errors = sum(1 for b in rec.get("forks") or [] if "error" in b)
        status = ("skipped" if rec.get("skipped") else "error" if rec.get("error") else
                  "incomplete" if branch_errors else "diverged" if rec.get("diverged") else "ok")
        line = dict(cell_id=ident, status=status, acc=fin.get("acc"),
                    acc_early=fin.get(f"acc{cs.chunk_early}"), acc_late=fin.get(f"acc{cs.chunk_late}"),
                    forked=rec.get("forked"), n_branches=len(rec.get("forks") or []),
                    branch_errors=branch_errors, wall_s=rec.get("wall_s"),
                    phase_wall_s=rec.get("phase_wall_s"), memory=rec.get("memory"))
        summary[ident] = line
        print("CHR_SUMMARY " + json.dumps(_finite(line)), flush=True)
    done_status = ("ok", "diverged")  # a divergence is a result; errors / missing branches are not
    n_ok = sum(v["status"] in done_status for v in summary.values())
    missing = [i for i, v in summary.items() if v["status"] not in done_status]
    complete = n_ok == len(ids) and not watchdog
    print(f"CHR_DONE cells={len(ids)} ok={n_ok} complete={complete} watchdog={watchdog} "
          f"worker_exit={codes} not_ok={missing}", flush=True)
    if not complete:
        print(f"!! CHR INCOMPLETE: {len(missing)} of {len(ids)} cells not ok: {missing}", flush=True)
    return dict(device=str(dev), torch=torch.__version__, complete=complete, not_ok=missing,
                summary=summary)
