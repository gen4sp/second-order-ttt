"""RTW toy wave: gated writes, SO/FO mixtures, and residual-floor controls.

The module deliberately keeps the inherited ``exp_ttt`` recurrence untouched.  The
new routes are expressed as small wrappers around the same model and data objects,
which makes the A0/SO anchors directly comparable and keeps the production run
contract ``python -m tttmem.run config.toml``.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as ttt
from .floor_rules import complement_ce

FAST_KEYS = ttt.FAST_KEYS
ARMS = ("A0", "SO", "G-hi", "G-lo", "G-scalar", "G-sg", "G-hi-c",
        "M-p125", "M-p25", "M-p50", "M-late", "M-early", "RP-all", "NC-all")
GATE_ARMS = ("G-hi", "G-lo", "G-scalar", "G-sg", "G-hi-c")
SOMIX_ARMS = ("M-p125", "M-p25", "M-p50", "M-late", "M-early")
PF_ARMS = ("RP-all", "NC-all")
SEEDS = tuple(range(300, 324))
SCREEN_SEEDS = tuple(range(300, 312))
ETA_BASE = 40.0
G_CLIP = 10.0
R_REF = 0.03
TRAIN_SEED = 20240101
EVAL_SEED = ttt.EVAL_SEED
DEFAULT_ARTIFACTS = Path("outputs/rtw")


# Inner-loop telemetry must not retain a tensor for every token and every chunk.
# Each key has a device-side sum/count and a bounded quantile reservoir for the
# current step.  ``_flush_stats`` emits one scalar row at each quick probe and
# resets the accumulators, so memory is O(number of quick probes).
_RESERVOIR = 4096   # ring of per-update samples; covers a 25-step quick window
_PER_UPDATE = 16    # strided values kept from each update (chunk call)


def _stat_update(stats: dict, key: str, value: torch.Tensor | float | int) -> None:
    if isinstance(value, torch.Tensor):
        v = value.detach().reshape(-1)
    else:
        v = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if v.numel() == 0:
        return
    acc = stats.setdefault("_acc", {}).get(key)
    # Keep reductions on the training device.  The quantile sample takes up to
    # _PER_UPDATE strided values from *every* update into a ring buffer, so the
    # median/q90 of a quick window represent all of its steps and chunks rather
    # than the last update only.
    stride = max(1, v.numel() // _PER_UPDATE)
    sample = v[::stride][:_PER_UPDATE].clone()
    if acc is None:
        acc = {"sum": v.sum(), "count": int(v.numel()), "seen": int(v.numel()),
               "reservoir": sample, "ptr": 0}
        stats["_acc"][key] = acc
        return
    acc["sum"] = acc["sum"] + v.sum()
    acc["count"] += int(v.numel())
    acc["seen"] += int(v.numel())
    room = _RESERVOIR - acc["reservoir"].numel()
    if room > 0:
        acc["reservoir"] = torch.cat((acc["reservoir"], sample[:room]))
        sample = sample[room:]
    if sample.numel():
        pos = (acc["ptr"] + torch.arange(sample.numel(), device=sample.device)) % _RESERVOIR
        acc["reservoir"][pos] = sample
        acc["ptr"] = (acc["ptr"] + int(sample.numel())) % _RESERVOIR


def _flush_stats(stats: dict) -> dict:
    """Materialise one bounded quick-step row and clear device accumulators."""
    row = {}
    for key, acc in stats.pop("_acc", {}).items():
        r = acc["reservoir"].detach().float()
        finite = torch.isfinite(r)
        r = r[finite]
        if r.numel():
            row[key] = {
                "n": int(acc["count"]),
                "mean": float((acc["sum"] / max(1, acc["count"])).detach().cpu()),
                "median": float(r.median().cpu()),
                "q90": float(torch.quantile(r, .9).cpu()),
            }
        else:
            row[key] = {"n": int(acc["count"])}
    stats.setdefault("series", []).append(row)
    return row


def _joint_norm(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    """Joint fast-tensor norm for tensors with a leading sequence dimension."""
    sq = None
    for g, key in zip(grads, FAST_KEYS):
        tail = ttt.MATRIX_NDIM[key]
        v = g.reshape(*g.shape[:-tail], -1).square().sum(-1)
        sq = v if sq is None else sq + v
    return sq.clamp_min(1e-24).sqrt()


def _scale_like(scale: torch.Tensor, g: torch.Tensor, key: str) -> torch.Tensor:
    tail = ttt.MATRIX_NDIM[key]
    return g * scale.reshape(*g.shape[:-tail], *([1] * tail))


def gate_beta(h: torch.Tensor, u: torch.Tensor | None, c: torch.Tensor | None,
              s_max: float = math.log(1000.0), *, scalar: bool = False,
              stop_h: bool = False, override: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the bounded exponential gate with unit derivative at zero.

    ``override`` is used by T0's fixed global/kv-oracle replay and is deliberately
    allowed to carry a gradient with respect to log beta.
    """
    if override is not None:
        return override
    if c is None:
        raise ValueError("gate bias is required")
    z = h.detach() if stop_h else h
    z = F.layer_norm(z, (z.shape[-1],), weight=None, bias=None)
    if scalar:
        s = c.reshape(1, 1, 1).expand(z.shape[0], z.shape[1], 1)
    else:
        if u is None:
            raise ValueError("per-token gate needs u")
        s = torch.einsum("btd,d->bt", z, u).unsqueeze(-1) + c
    return torch.exp(s_max * torch.tanh(s / s_max)).squeeze(-1)


def _gate_params(model, arm: str) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Install zero-initialised gain parameters on a model exactly once."""
    if not hasattr(model, "rtw_u"):
        d = model.head.ln.normalized_shape[0]
        model.register_parameter("rtw_u", torch.nn.Parameter(torch.zeros(d, device=model.rho.device,
                                                                          dtype=model.rho.dtype)))
        model.register_parameter("rtw_c", torch.nn.Parameter(torch.zeros((), device=model.rho.device,
                                                                          dtype=model.rho.dtype)))
    return model.rtw_u, model.rtw_c


def _gate_rule(arm: str) -> tuple[float, bool, bool, str]:
    """Return (cap, scalar, stop_h, clip) for a GATE arm."""
    if arm not in GATE_ARMS:
        raise ValueError(f"unknown GATE arm {arm}")
    return (math.log(2.0) if arm == "G-lo" else math.log(1000.0),
            arm == "G-scalar", arm == "G-sg",
            "current" if arm == "G-hi-c" else "base")


def _write_mask(meta: dict, T: int, device: torch.device) -> torch.Tensor:
    wp = torch.as_tensor(meta["write_pos"], device=device, dtype=torch.long)
    m = torch.zeros((wp.shape[0], T), device=device, dtype=torch.bool)
    return m.scatter(1, wp.clamp(0, T - 1), True)


def detached_gate_gradients(model, h: torch.Tensor, idx: torch.Tensor, y: torch.Tensor,
                            W: list[torch.Tensor], beta: torch.Tensor,
                            x_override: torch.Tensor | None = None,
                            factor_override: tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor] | None = None
                            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(ungated G, beta-weighted G)`` from detached token factors.

    The read head is used only to obtain the detached error ``e``.  Every factor in
    ``G_t`` (``e``, ``x``, ``gelu'`` and ``W2``) is detached before the live beta
    weighting.  Consequently a backward pass can add gain/backbone paths through
    beta while never creating a hidden SO path through ``ln_f`` or ``unembed``.
    """
    if factor_override is None:
        x_live = x_override if x_override is not None else (model.fast_input(h, idx) if model.frozen_emb is not None else model.head.ln(h))
        out = ttt.fast_block(h, model.head.ln, W, x_live if model.frozen_emb is not None else None)
        ce = ttt.token_ce(model.read(out), y)
        e = torch.autograd.grad(ce.sum(), out, retain_graph=True)[0].detach()
        x = x_live.detach()
        W2 = W[2].detach()
        u = torch.einsum("btd,bmd->btm", x, W[0].detach()) + W[1].detach()[:, None]
        a = F.gelu(u).detach()
        gp = (0.5 * (1 + torch.erf(u / math.sqrt(2))) +
              u * torch.exp(-u.square() / 2) / math.sqrt(2 * math.pi)).detach()
        e = e.detach()
    else:
        # T0 fixed-factor replay: freeze exactly the factors recorded on the
        # unperturbed trajectory, while the outer read still uses the perturbed W.
        e, x, a, gp, W2 = (q.detach() for q in factor_override)
    eps = torch.einsum("btd,bdm->btm", e, W2).detach()
    r = eps * gp
    inv_t = 1.0 / max(1, y.shape[1])
    raw = [torch.einsum("btm,btd->bmd", r, x) * inv_t, r.sum(1) * inv_t,
           torch.einsum("btd,btm->bdm", e, a) * inv_t, e.sum(1) * inv_t]
    weighted_r = beta.unsqueeze(-1) * r
    weighted_e = beta.unsqueeze(-1) * e
    weighted = [torch.einsum("btm,btd->bmd", weighted_r, x) * inv_t, weighted_r.sum(1) * inv_t,
                torch.einsum("btd,btm->bdm", weighted_e, a) * inv_t, weighted_e.sum(1) * inv_t]
    return raw, weighted


def _clip_rule(base: list[torch.Tensor], gated: list[torch.Tensor], g_clip: float,
               variant: str) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Apply none/base/current clipping; the scale is always detached."""
    if variant == "none":
        norm = _joint_norm(gated).detach()
        return gated, norm, norm > g_clip
    source = base if variant == "base" else gated
    norm = _joint_norm(source).detach()
    scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
    return [_scale_like(scale, g, k) for g, k in zip(gated, FAST_KEYS)], norm, norm > g_clip


def gate_inner_loss(model, idx: torch.Tensor, y: torch.Tensor, meta: dict,
                    *, arm: str = "G-hi", eta_base: float = ETA_BASE,
                    g_clip: float = G_CLIP, chunk: int = 64,
                    clip_variant: str | None = None,
                    beta_override: torch.Tensor | None = None,
                    stats: dict | None = None) -> torch.Tensor:
    """Training-time GATE outer CE with the production base-clip rule."""
    u, c = _gate_params(model, arm)
    h = model.backbone(idx)
    x_full = model.fast_input(h, idx) if model.frozen_emb is not None else None
    W = model.per_sequence(idx.shape[0])
    T = idx.shape[1]
    cap, scalar, stop_h, default_clip = _gate_rule(arm)
    variant = clip_variant or default_clip
    total = torch.zeros((), device=idx.device, dtype=h.dtype)
    stats = stats if stats is not None else {}
    write = _write_mask(meta, T, idx.device)
    for lo in range(0, T, chunk):
        sl = slice(lo, lo + chunk)
        if beta_override is None:
            beta = gate_beta(h[:, sl], u, c, cap, scalar=scalar, stop_h=stop_h)
        else:
            beta = beta_override[:, sl] if beta_override.ndim == 2 else beta_override.expand(idx.shape[0], sl.stop - sl.start)
        raw, weighted = detached_gate_gradients(model, h[:, sl], idx[:, sl], y[:, sl], W, beta,
                                                 None if x_full is None else x_full[:, sl])
        gated, norm, clipped = _clip_rule(raw, weighted, g_clip, variant)
        _, base_norm, base_clipped = ttt._clip_fast(raw, g_clip)
        _stat_update(stats, "clip", clipped)
        _stat_update(stats, "clip_base", base_clipped)
        _stat_update(stats, "ratio", _joint_norm(weighted) / _joint_norm(raw).clamp_min(1e-24))
        bw, bo = beta[write[:, sl]], beta[~write[:, sl]]
        if bw.numel(): _stat_update(stats, "beta_write", bw)
        if bo.numel(): _stat_update(stats, "beta_other", bo)
        out = ttt.fast_block(h[:, sl], model.head.ln, W, None if x_full is None else x_full[:, sl])
        ce = ttt.token_ce(model.read(out), y[:, sl])
        total = total + ce.mean()
        W = [w - eta_base * g for w, g in zip(W, gated)]
    return total / max(1, T // chunk)


def somix_mask(seed: int, step: int, batch: int, p: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """Counter-based paired subset mask used by all SOMIX arms."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    rng = np.random.default_rng([int(seed), int(step), 0x534F4D49])
    perm = rng.permutation(batch)
    k = int(round(p * batch))
    out = np.zeros(batch, dtype=bool); out[perm[:k]] = True
    return torch.from_numpy(out).to(device)


def _somix_fraction(arm: str, step: int) -> float:
    return {"M-p125": .125, "M-p25": .25, "M-p50": .5,
            "M-late": 1.0 if step >= 600 else 0.0,
            "M-early": 1.0 if step < 400 else 0.0}[arm]


def somix_loss(model, idx: torch.Tensor, y: torch.Tensor, seed: int, step: int,
               arm: str, *, eta_base: float = ETA_BASE, g_clip: float = G_CLIP,
               chunk: int = 64, stats: dict | None = None) -> torch.Tensor:
    """One-backbone-pass mixed outer loss, weighted by subset size."""
    h = model.backbone(idx)
    p = _somix_fraction(arm, step)
    mask = somix_mask(seed, step, len(idx), p, idx.device)
    stats = stats if stats is not None else {}
    stats["mask"] = mask.detach().cpu().tolist()
    total = torch.zeros((), device=idx.device, dtype=h.dtype)
    for want_so in (False, True):
        sel = mask if want_so else ~mask
        if not bool(sel.any()):
            continue
        sub_h, sub_i, sub_y = h[sel], idx[sel], y[sel]
        gs = ttt.GradStats()
        loss = ttt.sequential_inner(model, sub_h, sub_y, model.per_sequence(int(sel.sum())),
                                    model.etas(eta_base), g_clip, want_so, gs, chunk,
                                    idx=sub_i)
        total = total + loss * (sel.to(loss.dtype).mean())
        norm = torch.cat([n.reshape(-1) for n in gs.norms]).mean() if gs.norms else torch.zeros((), device=idx.device)
        _stat_update(stats, "so_norm" if want_so else "fo_norm", norm)
    return total


def _nc_control_mask(write_pos: Any, active: torch.Tensor, seed: int, step: int, chunk_id: int,
                     offset: int = 0) -> torch.Tensor:
    """Draw class-matched non-active positions using a counter-based RNG."""
    B, T = active.shape
    wp = np.asarray(write_pos)
    active_np = active.detach().cpu().numpy()  # one transfer per chunk
    out = np.zeros((B, T), dtype=bool)
    for row in range(B):
        for cls in (0, 1):
            klass = np.zeros(T, dtype=bool)
            local = wp[row] - int(offset)
            local = local[(local >= 0) & (local < T)]
            if cls == 1:
                klass[local] = True
            else:
                klass[:] = True; klass[local] = False
            a = np.flatnonzero(active_np[row] & klass)
            n = np.flatnonzero((~active_np[row]) & klass)
            k = min(len(a), len(n))
            if k:
                rng = np.random.default_rng([seed, step, row, chunk_id, cls])
                out[row, rng.choice(n, size=k, replace=False)] = True
    return torch.from_numpy(out).to(active.device)


def pf_inner_loss(model, idx: torch.Tensor, y: torch.Tensor, meta: dict, *, arm: str,
                  eta_base: float = ETA_BASE, g_clip: float = G_CLIP, chunk: int = 64,
                  seed: int = 0, step: int = 0, stats: dict | None = None) -> torch.Tensor:
    """PF-T recurrence. RP uses all positions; NC matches active classes."""
    if arm not in PF_ARMS:
        raise ValueError(arm)
    h = model.backbone(idx)
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    W = model.per_sequence(len(idx)); total = torch.zeros((), device=idx.device, dtype=h.dtype)
    stats = stats if stats is not None else {}
    for c, lo in enumerate(range(0, idx.shape[1], chunk)):
        sl = slice(lo, lo + chunk)
        logits = model.read(ttt.fast_block(h[:, sl], model.head.ln, W, None if x is None else x[:, sl]))
        ce = ttt.token_ce(logits, y[:, sl]); base = list(torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=True))
        r = -torch.expm1(-ce.detach()); active = r < R_REF
        coeff = (R_REF - r).clamp_min(0).detach()
        ex_obj = (coeff * complement_ce(logits, y[:, sl])).sum() / chunk
        g_ex = list(torch.autograd.grad(ex_obj, W, retain_graph=True))
        if arm == "NC-all":
            ctl_mask = _nc_control_mask(meta["write_pos"], active, seed, step, c, lo)
            ctl_obj = (ctl_mask * ce).sum() / chunk
            g_ctl = list(torch.autograd.grad(ctl_obj, W, retain_graph=True))
            exn, ctn = _joint_norm(g_ex).detach(), _joint_norm(g_ctl).detach()
            kappa = torch.where(ctn > 1e-12, exn / ctn.clamp_min(1e-12), torch.zeros_like(ctn))
            extra = [_scale_like(kappa, g, k) for g, k in zip(g_ctl, FAST_KEYS)]
            _stat_update(stats, "match", ctl_mask.sum(1) / active.sum(1).clamp_min(1))
            _stat_update(stats, "nc_fail", ((ctn < 1e-12) |
                                              ((kappa * ctn - exn).abs() > 1e-4 * exn.clamp_min(1e-12))))
        else:
            extra = g_ex
        raw = [a + b for a, b in zip(base, extra)]
        clipped, norm, hit = ttt._clip_fast(raw, g_clip)
        _stat_update(stats, "clip", hit)
        _stat_update(stats, "ratio", _joint_norm(g_ex) / _joint_norm(base).clamp_min(1e-24))
        _stat_update(stats, "induced_clip", hit & ~ttt._clip_fast(base, g_clip)[2])
        if arm == "NC-all":
            _, ce_hit = ttt._clip_fast(base, g_clip)[1:]
            _stat_update(stats, "clip_ce", ce_hit)
        wp = torch.as_tensor(meta["write_pos"], device=idx.device)
        cls = torch.zeros((len(idx), idx.shape[1]), dtype=torch.bool, device=idx.device)
        cls.scatter_(1, wp.clamp(0, idx.shape[1]-1), True)
        _stat_update(stats, "active_write", active[cls[:, sl]])
        _stat_update(stats, "active_other", active[~cls[:, sl]])
        total = total + ce.mean(); W = [w - eta_base * g for w, g in zip(W, clipped)]
    return total / max(1, idx.shape[1] // chunk)


def _settings(cfg: dict) -> tuple[ttt.Settings, ttt.Task]:
    st = ttt._settings(cfg)
    task = ttt.Task(name="echo", V=2 + 3 * int(cfg.get("n_keys", 16)), T=int(cfg.get("T", 512)),
                    window=int(cfg.get("window", 64)), chunk=int(cfg.get("chunk", 64)),
                    steps=int(cfg.get("steps", 1500)), n_eval=int(cfg.get("n_eval", 256)),
                    n_val=int(cfg.get("n_val", 64)), mode="echo", n_keys=int(cfg.get("n_keys", 16)),
                    n_layers=int(cfg.get("n_layers", 4)), prefix_min=0, prefix_max=0)
    return st, task


def _optim(model, st: ttt.Settings):
    gain = [p for n, p in model.named_parameters() if n.startswith("rtw_")]
    normal = [p for n, p in model.named_parameters() if not n.startswith("rtw_") and n not in ("rho", "rho_kvW2")]
    decay = [p for p in normal if p.ndim >= 2]; plain = [p for p in normal if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": st.weight_decay, "base_lr": st.lr},
              {"params": plain, "weight_decay": 0., "base_lr": st.lr},
              {"params": [model.rho], "weight_decay": 0., "base_lr": 0.}]
    if gain:
        groups.append({"params": gain, "weight_decay": 0., "base_lr": .1, "gain_group": True})
    return torch.optim.AdamW(groups, lr=st.lr, betas=(st.beta1, st.beta2))


def _metric_rows(model, task, data, arm: str, st, *, own: bool) -> dict:
    rows = []
    for idx, y, meta in data:
        if own and arm in GATE_ARMS:
            rows.append(_gate_eval(model, idx, y, meta, st, arm))
        elif own and arm in PF_ARMS:
            rows.append(_pf_eval(model, idx, y, meta, st, arm))
        else:
            base = "e2e2emb" if arm == "SO" else "e2e1emb"
            rows.append(ttt.eval_batch_logits(model, task, base, idx, y, ETA_BASE, G_CLIP,
                                               task.chunk, ttt.GradStats(), learned=False, meta=meta))
    ce = torch.cat([ttt.token_ce(z, y) for z, (_, y, _) in zip(rows, data)])
    pred = torch.cat([z.argmax(-1) for z in rows])
    meta = {k: np.concatenate([m[k] for _, _, m in data]) for k in ("query_pos", "query_tok")}
    pos = torch.as_tensor(meta["query_pos"], device=pred.device, dtype=torch.long)
    tok = torch.as_tensor(meta["query_tok"], device=pred.device)
    return {"outer_loss": float(ce.mean()), "binding_acc": float((pred.gather(1, pos) == tok).float().mean())}


def _metric(model, task, data, arm: str, st) -> dict:
    """Return own-rule and plain CE reads for every route."""
    own = _metric_rows(model, task, data, arm, st, own=True)
    ce = _metric_rows(model, task, data, arm, st, own=False)
    return {"own": own, "ce": ce,
            "acc_own": own["binding_acc"], "acc_ce": ce["binding_acc"],
            "retained_own": own["binding_acc"] >= .90,
            "retained_ce": ce["binding_acc"] >= .90}


def _gate_eval(model, idx, y, meta, st, arm: str):
    """Eval-time GATE recurrence under the arm's own cap/scalar/clip rule."""
    u, c = _gate_params(model, arm)
    cap, scalar, stop_h, clip = _gate_rule(arm)
    h = model.backbone(idx).detach()
    x_full = model.fast_input(h, idx) if model.frozen_emb is not None else None
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    out = []
    write = _write_mask(meta, idx.shape[1], idx.device)
    for lo in range(0, idx.shape[1], st.chunk):
        sl = slice(lo, lo + st.chunk)
        beta = gate_beta(h[:, sl], u.detach(), c.detach(), cap, scalar=scalar,
                         stop_h=stop_h).detach()
        with torch.enable_grad():
            raw, weighted = detached_gate_gradients(
                model, h[:, sl], idx[:, sl], y[:, sl], W, beta,
                None if x_full is None else x_full[:, sl])
            g, _, _ = _clip_rule(raw, weighted, st.g_clip, clip)
            z = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                          None if x_full is None else x_full[:, sl]))
        out.append(z.detach())
        W = [(w - ETA_BASE * q).detach().requires_grad_(True) for w, q in zip(W, g)]
    return torch.cat(out, 1)


def _pf_eval(model, idx, y, meta, st, arm: str):
    """Eval-time PF-T recurrence under RP/NC's own rule."""
    h = model.backbone(idx).detach()
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    outs = []
    for c, lo in enumerate(range(0, idx.shape[1], st.chunk)):
        sl = slice(lo, lo + st.chunk)
        with torch.enable_grad():
            z = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                          None if x is None else x[:, sl]))
            ce = ttt.token_ce(z, y[:, sl])
            base = list(torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=True))
            r = -torch.expm1(-ce.detach()); active = r < R_REF
            coeff = (R_REF - r).clamp_min(0).detach()
            extra_obj = (coeff * complement_ce(z, y[:, sl])).sum() / st.chunk
            extra = list(torch.autograd.grad(extra_obj, W, retain_graph=True))
            if arm == "NC-all":
                # Eval control is keyed only by (row, chunk); the seed is a
                # stable stream identifier and does not depend on training step.
                ctl = _nc_control_mask(meta["write_pos"], active, 0, 0, c, lo)
                ctl_obj = (ctl * ce).sum() / st.chunk
                g_ctl = list(torch.autograd.grad(ctl_obj, W, retain_graph=True))
                exn, ctn = _joint_norm(extra).detach(), _joint_norm(g_ctl).detach()
                extra = [_scale_like(torch.where(ctn > 1e-12, exn / ctn.clamp_min(1e-12),
                                                   torch.zeros_like(ctn)), g, k)
                         for g, k in zip(g_ctl, FAST_KEYS)]
            raw = [a + b for a, b in zip(base, extra)]
            clipped, _, _ = ttt._clip_fast(raw, st.g_clip)
        outs.append(z.detach())
        W = [(w - ETA_BASE * q).detach().requires_grad_(True) for w, q in zip(W, clipped)]
    return torch.cat(outs, 1)


def _cell(cfg: dict, seed: int, arm: str, dev: torch.device) -> dict:
    st, task = _settings(cfg)
    ttt.configure_tf32(bool(cfg.get("tf32", False)), f"rtw-{seed}")
    ttt._reset_peak(dev)
    model_arm = "e2e2emb" if arm == "SO" else "e2e1emb"
    model = ttt.build_model(task, st, dev, seed, model_arm)
    if arm in GATE_ARMS:
        _gate_params(model, arm)
    opt = _optim(model, st)
    rng = np.random.default_rng([seed, TRAIN_SEED])
    stats: dict[str, Any] = {}
    steps = int(cfg.get("steps", 1500)); times = []; logs = []
    qraw, qmeta = ttt.fixed_set(task, st, "quick")
    qdata = ttt.eval_batches(task, qraw, qmeta, dev, int(cfg.get("eval_batch", 32)))
    nonfinite_steps = 0; first_nonfinite = None
    for step in range(steps):
        raw, meta = task.raw(rng, int(cfg.get("batch", 32)))
        idx, y, meta = task.tensors(raw, meta, dev)
        t0 = time.perf_counter(); opt.zero_grad(set_to_none=True)
        if arm == "A0":
            loss = ttt.arm_loss(model, "e2e1emb", idx, y, ETA_BASE, G_CLIP,
                                task.chunk, ttt.GradStats(), meta=meta)
        elif arm == "SO":
            loss = ttt.arm_loss(model, "e2e2emb", idx, y, ETA_BASE, G_CLIP,
                                task.chunk, ttt.GradStats(), meta=meta)
        elif arm in GATE_ARMS:
            loss = gate_inner_loss(model, idx, y, meta, arm=arm, stats=stats)
        elif arm in SOMIX_ARMS:
            loss = somix_loss(model, idx, y, seed, step, arm, stats=stats)
        else:
            loss = pf_inner_loss(model, idx, y, meta, arm=arm, seed=seed,
                                 step=step, stats=stats)
        finite = bool(torch.isfinite(loss).item())
        if finite:
            loss.backward()
            clipped_params = [p for n, p in model.named_parameters()
                              if n not in ("rho", "rho_kvW2")]
            grad_norm = torch.nn.utils.clip_grad_norm_(clipped_params, st.grad_clip)
            finite = bool(torch.isfinite(grad_norm).item()) and all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in clipped_params)
        if not finite:
            nonfinite_steps += 1
            first_nonfinite = first_nonfinite or step + 1
            opt.zero_grad(set_to_none=True)
            print(f"  WARNING RTW arm={arm} seed={seed}: skipped non-finite step {step + 1}", flush=True)
        scale = ttt.lr_scale(step, int(cfg.get("warmup", 200)),
                             int(cfg.get("lr_total", 3000)), float(cfg.get("lr_floor", .1)))
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] if group.get("gain_group") else group["base_lr"] * scale
        if finite:
            opt.step()
        ttt._sync(dev)
        times.append((time.perf_counter() - t0) * 1000)
        done = step + 1
        if done % int(cfg.get("log_every", 5)) == 0 or done == steps:
            logs.append({"step": done, "loss": float(loss.detach()),
                         "ms_step": float(np.median(times[-min(25, len(times)):]))})
            print(f"RTW arm={arm} seed={seed} step={done} loss={float(loss.detach()):.5f} "
                  f"ms={logs[-1]['ms_step']:.2f}", flush=True)
        if done % int(cfg.get("quick_every", 25)) == 0:
            srow = _flush_stats(stats)
            qmetric = _metric(model, task, qdata, arm, st)
            logs.append({"step": done, "quick": {"acc": qmetric["acc_own"]},
                         "quick_loss_own": qmetric["own"]["outer_loss"],
                         "quick_binding_acc": qmetric["acc_own"],
                         "quick_binding_acc_ce": qmetric["acc_ce"], "stats": srow})
    if stats.get("_acc"):
        _flush_stats(stats)
    evraw, evmeta = ttt.fixed_set(task, st, "eval")
    ev = ttt.eval_batches(task, evraw, evmeta, dev, int(cfg.get("eval_batch", 32)))
    metric = _metric(model, task, ev, arm, st)
    # This is the same first-quick-read definition used by POOL's trigger detector.
    from .triggers import detect
    events = detect([x for x in logs if "quick" in x])
    acquired = "t_acq" in events
    t_acq = events["t_acq"].event_step if acquired else None
    result = {"arm": arm, "seed": seed, "steps": steps,
              "retained": metric["retained_own"], "retained_own": metric["retained_own"],
              "retained_ce": metric["retained_ce"], "ever_acquired": acquired,
              "t_acq": t_acq,
              "ms_step": float(np.median(times[max(1, min(100, len(times) - 1)):])) if times else None,
              "logs": logs, "metric": metric, "peak_memory_mb": ttt._peak_mem(dev),
              "nonfinite_steps": nonfinite_steps, "first_nonfinite": first_nonfinite,
              "diverged": nonfinite_steps > 0, "stats": _stats_summary(stats)}
    if arm in GATE_ARMS:
        series = result["stats"].get("series", [])
        bw = [r["beta_write"]["median"] for r in series if r.get("beta_write")]
        bo = [r["beta_other"]["median"] for r in series if r.get("beta_other")]
        result["beta_write_median"] = float(np.median(bw)) if bw else None
        result["beta_other_median"] = float(np.median(bo)) if bo else None
        post = [r for r in logs if "quick" in r and t_acq is not None and r["step"] > t_acq]
        if not post:
            post = [r for r in logs if "quick" in r][-20:]
            result["beta_localisation_fallback"] = True
        take = len(post) if post else len(bw)
        result["beta_localisation"] = (float(np.median(bw[-take:]) / max(np.median(bo[-take:]), 1e-24))
                                        if bw and bo else None)
    return result


def _jsonable(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, dict): return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x, list): return [_jsonable(v) for v in x]
    if isinstance(x, (np.generic,)): return x.item()
    return x


def _stats_summary(stats: dict) -> dict:
    """Return only bounded per-quick-step telemetry."""
    out = {"series": list(stats.get("series", []))}
    if "mask" in stats:
        out["mask"] = stats["mask"]
    rows = out["series"]
    for key in ("beta_write", "beta_other", "ratio", "clip", "clip_base", "induced_clip",
                "clip_ce", "active_write", "active_other", "match", "nc_fail",
                "so_norm", "fo_norm"):
        vals = [r[key] for r in rows if key in r and "mean" in r[key]]
        if vals:
            out[key] = {"n": int(sum(v["n"] for v in vals)),
                        "mean": float(np.mean([v["mean"] for v in vals])),
                        "median": float(np.median([v["median"] for v in vals])),
                        "q90": float(np.median([v["q90"] for v in vals]))}
            if key == "nc_fail":
                out[key]["count"] = int(round(sum(v["mean"] * v["n"] for v in vals)))
    return out


def _cell_worker(index: int, cfg: dict, seed: int, arm: str) -> tuple[str, dict]:
    """Spawn-safe worker entrypoint; the parent owns result-file writes."""
    threads = min(4, max(1, int(cfg.get("threads_per_worker", 2))))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    base_name = str(cfg.get("process_name", "TTM-rtw-01-w0"))
    ttt._mark_process(base_name if index == 0 else base_name.rsplit("-w", 1)[0] + f"-w{index}")
    dev = ttt._device(str(cfg.get("device", "auto")))
    key = f"{arm}__s{seed:03d}"
    return key, _cell(cfg, seed, arm, dev)


def _cell_worker_args(args: tuple) -> tuple[str, dict]:
    return _cell_worker(*args)


def _write_cell(file: Path, key: str, rec: dict) -> None:
    """Persist one finished cell atomically and print its progress line."""
    tmp = file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(rec), indent=2, sort_keys=True)); tmp.replace(file)
    print("RTW_CELL_DONE " + json.dumps({"cell": key, "retained_own": rec.get("retained_own"),
                                          "retained_ce": rec.get("retained_ce"),
                                          "ever_acquired": rec.get("ever_acquired"),
                                          "ms_step": rec.get("ms_step"),
                                          "peak_memory_mb": rec.get("peak_memory_mb"),
                                          "nonfinite_steps": rec.get("nonfinite_steps")}), flush=True)


def run(cfg: dict, seed: int = 0) -> dict:
    """Run selected RTW cells; reject all seeds outside the registered range."""
    torch.set_num_threads(min(4, max(1, int(cfg.get("threads_per_worker", 2)))))
    try:
        torch.set_num_interop_threads(min(4, max(1, int(cfg.get("threads_per_worker", 2)))))
    except RuntimeError:
        # PyTorch permits this setting only before the first parallel op.
        pass
    seeds = [int(x) for x in cfg.get("seeds", [seed])]
    cells = cfg.get("cells")
    if cells is None:
        cells = [{"arm": a, "seed": s} for a in cfg.get("arms", ARMS) for s in seeds]
    bad = [int(c["seed"]) for c in cells if int(c["seed"]) not in SEEDS]
    if bad: raise ValueError(f"rtw seeds must lie in 300..323, got {bad}")
    dev = ttt._device(str(cfg.get("device", "auto"))); ttt._mark_process(str(cfg.get("process_name", "TTM-rtw-01-w0")))
    root=Path(cfg.get("output_root", "outputs/rtw-output")); root.mkdir(parents=True, exist_ok=True)
    out={"experiment":"rtw","device":str(dev),"arms":list(ARMS),"cells":{}}
    pending=[]
    for i,c in enumerate(cells):
        arm=str(c["arm"]); s=int(c["seed"])
        if arm not in ARMS: raise ValueError(f"unknown RTW arm {arm}")
        key=f"{arm}__s{s:03d}"; folder=root/key; folder.mkdir(parents=True,exist_ok=True); file=folder/"result.json"
        if cfg.get("resume",True) and file.exists(): out["cells"][key]=json.loads(file.read_text()); continue
        pending.append((i,key,arm,s,file))
    workers=max(1,int(cfg.get("workers",1)))
    if workers > 1 and len(pending) > 1:
        # CUDA workers are independent processes so each owns its model and
        # allocator; spawn also avoids inheriting a live CUDA context.
        ctx=mp.get_context("spawn")
        files={key:file for _,key,_,_,file in pending}
        with ctx.Pool(processes=min(workers,len(pending))) as pool:
            # Write each cell as soon as it finishes, so an interrupted run keeps every
            # finished cell and `resume` restarts only the unfinished ones.
            for key,rec in pool.imap_unordered(_cell_worker_args, [(j,cfg,s,a) for j,_,a,s,_ in pending]):
                _write_cell(files[key], key, rec); out["cells"][key]=rec
    else:
        for i,key,arm,s,file in pending:
            rec=_cell(cfg,s,arm,dev); _write_cell(file, key, rec); out["cells"][key]=rec
    summary={}
    for arm in ARMS:
        rs=[r for r in out["cells"].values() if r["arm"]==arm]
        summary[arm]={"n":len(rs),"retained":sum(bool(r.get("retained_own", r.get("retained"))) for r in rs),
                     "retained_own":sum(bool(r.get("retained_own", r.get("retained"))) for r in rs),
                     "retained_ce":sum(bool(r.get("retained_ce")) for r in rs),
                     "ever_acquired":sum(bool(r.get("ever_acquired")) for r in rs),
                     "ms_step":float(np.nanmean([r["ms_step"] for r in rs])) if rs else None,
                     "peak_memory_mb":float(np.nanmax([r.get("peak_memory_mb",0.) for r in rs])) if rs else None}
    out["summary"]=summary
    if cfg.get("pod_smoke"):
        for arm,row in summary.items():
            if row["n"]: print(f"RTW_PODSMOKE arm={arm} ms_step={row['ms_step']:.3f} peak_memory_mb={row['peak_memory_mb']:.1f}",flush=True)
        total_s=sum(float(r.get("steps",0))*float(r.get("ms_step",0.)) / 1000. for r in out["cells"].values() if np.isfinite(r.get("ms_step",float("nan"))))
        projected=total_s/max(1,workers); out["pod_smoke"]={"workers":workers,"sum_serial_s":total_s,"projected_wave_s":projected}
        print(f"RTW_PODSMOKE projected_wave_s={projected:.3f} workers={workers}",flush=True)
    (root/"summary.json").write_text(json.dumps(_jsonable(out),indent=2,sort_keys=True)); print("RTW_SUMMARY "+json.dumps(summary,sort_keys=True),flush=True)
    return out


def _state_paths(kind: str, seeds=(200,201,202,203)) -> list[Path]:
    root=Path("outputs/pool-output/mains")
    arm="e2e1emb" if kind=="FO" else "e2e2emb"
    return [root/f"echo__s{s}__{arm}__eta40__st1500/weights-step1500.pt" for s in seeds]


def _load_state(task, st, path: Path, seed: int):
    model=ttt.build_model(task,st,torch.device("cpu"),seed,"e2e1emb").double(); payload=torch.load(path,map_location="cpu",weights_only=False)
    model.load_state_dict(payload,strict=True); return model


def _fixed_t0_batches(task, n=8, b=4):
    rng=np.random.default_rng([EVAL_SEED,77]); return [task.tensors(*task.raw(rng,b),torch.device("cpu")) for _ in range(n)]


def _group_name(name: str) -> str:
    if name=="tok.weight": return "token_embedding"
    if name.startswith("blocks."):
        i=name.split(".")[1]; return f"backbone_{i}_attention" if any(x in name for x in ("ln1","qkv","proj")) else f"backbone_{i}_mlp"
    if name.startswith("head.ln."): return "read_head"
    if name.startswith("head.") or name in ("rho",):
        return f"fast_W0_{name.split('.', 1)[1]}" if name.startswith("head.") else "eta"
    if name.startswith("ln_f") or name.startswith("unembed"): return "read_head"
    if name=="frozen_emb": return "frozen_code"
    return "other"


def _meta_grads(model, arm, idx, y, meta):
    model.zero_grad(set_to_none=True); loss=ttt.arm_loss(model,arm,idx,y,ETA_BASE,G_CLIP,64,ttt.GradStats(),meta=meta)
    gs=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True); return {n:(g.detach().clone() if g is not None else torch.zeros_like(p)) for (n,p),g in zip(model.named_parameters(),gs)},float(loss.detach())


def _gate_replay(model, idx, y, meta, beta, clip, *, factor_cache=None,
                 record_factors=False, h_override=None, x_override=None):
    """Fixed-gain replay, optionally reusing the unperturbed factor trajectory."""
    h = model.backbone(idx).detach() if h_override is None else h_override.detach()
    x_full = ((model.fast_input(h, idx) if x_override is None else x_override)
              if model.frozen_emb is not None else None)
    W = model.per_sequence(len(idx)); out = []; clips = []; factors = []
    b = beta if torch.is_tensor(beta) else torch.tensor(float(beta), dtype=h.dtype)
    for ci, lo in enumerate(range(0, idx.shape[1], 64)):
        hi = min(lo + 64, idx.shape[1]); sl = slice(lo, hi)
        width = hi - lo
        bb = b.expand(len(idx), width) if b.ndim == 0 else b[:, sl]
        override = None if factor_cache is None else factor_cache[ci]
        with torch.enable_grad():
            if override is None:
                # Compute and retain the factors that FO deliberately drops.
                xx = x_full[:, sl] if x_full is not None else model.head.ln(h[:, sl])
                W2 = W[2].detach()
                uu = torch.einsum("btd,bmd->btm", xx.detach(), W[0].detach()) + W[1].detach()[:, None]
                aa = F.gelu(uu).detach()
                gg = (0.5 * (1 + torch.erf(uu / math.sqrt(2))) +
                      uu * torch.exp(-uu.square() / 2) / math.sqrt(2 * math.pi)).detach()
                tmp = ttt.fast_block(h[:, sl], model.head.ln, W,
                                     None if x_full is None else x_full[:, sl])
                ee = torch.autograd.grad(ttt.token_ce(model.read(tmp), y[:, sl]).sum(), tmp,
                                         retain_graph=True)[0].detach()
                override_now = (ee, xx.detach(), aa, gg, W2)
            else:
                override_now = override
            raw, weighted = detached_gate_gradients(
                model, h[:, sl], idx[:, sl], y[:, sl], W, bb,
                None if x_full is None else x_full[:, sl], factor_override=override_now)
            g, _, hit = _clip_rule(raw, weighted, 10., clip)
            z = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                          None if x_full is None else x_full[:, sl]))
        out.append(z)
        clips.append(hit.detach())
        if record_factors:
            factors.append(tuple(q.detach().clone() for q in override_now))
        W = [w - ETA_BASE * q for w, q in zip(W, g)]
    result = (torch.cat(out, 1), torch.cat(clips))
    return (result[0], result[1], factors) if record_factors else result


def _flat_rel(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    num = torch.sqrt(sum((x - y).square().sum() for x, y in zip(a, b)))
    den = torch.sqrt(sum(y.square().sum() for y in b)).clamp_min(1e-30)
    return float((num / den).detach())


def _per_sequence_grad(model, idx, y, *, second: bool) -> list[torch.Tensor]:
    h = model.backbone(idx)
    loss = ttt.sequential_inner(model, h, y, model.per_sequence(len(idx)),
                                model.etas(ETA_BASE), G_CLIP, second,
                                ttt.GradStats(), 64, idx=idx)
    gs = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return [(g if g is not None else torch.zeros_like(p)).detach()
            for p, g in zip(model.parameters(), gs)]


def _mixed_direct_grad(model, idx, y, mask) -> list[torch.Tensor]:
    h = model.backbone(idx); total = torch.zeros((), dtype=h.dtype)
    for second in (False, True):
        sel = mask if second else ~mask
        if bool(sel.any()):
            sub = ttt.sequential_inner(model, h[sel], y[sel], model.per_sequence(int(sel.sum())),
                                       model.etas(ETA_BASE), G_CLIP, second,
                                       ttt.GradStats(), 64, idx=idx[sel])
            total = total + sub * sel.to(sub.dtype).mean()
    gs = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
    return [(g if g is not None else torch.zeros_like(p)).detach()
            for p, g in zip(model.parameters(), gs)]


def run_t0(output: Path | None = None) -> dict:
    """Run the eval-only calibration and write ``t0.json``/``t0.md``."""
    ttt._mark_process("TTM-rtw-02-t0")
    torch.set_num_threads(2)
    output = Path(output or DEFAULT_ARTIFACTS / "t0")
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"d_model":128,"n_layers":4,"n_heads":4,"backbone_mult":4,"fast_mult":4,
           "window":64,"chunk":64,"T":512,"steps":1500,"batch":32,"n_keys":16,
           "n_eval":256,"n_val":64,"n_quick_eval":64,"eval_batch":32,"eta_lr":0.,
           "lr_total":3000,"warmup":200,"lr_floor":.1,"lr":1e-3,"weight_decay":.01,
           "grad_clip":1.,"g_clip":10.,"device":"cpu"}
    st, task = _settings(cfg)
    paths = {"FO": _state_paths("FO"), "SO": _state_paths("SO")}
    states = []
    for kind, ps in paths.items():
        for j, path in enumerate(ps):
            if path.exists():
                states.append((kind, 200 + j, path))
    inits = [("INIT", s, None) for s in range(300, 304)]
    # (a) use a 64-sequence replay.
    gate_batch = _fixed_t0_batches(task, 1, 64)[0]
    gate = {"replay": [], "gradient_check": []}
    gate_states = [(f"{kind}_s{seed}", _load_state(task, st, path, seed))
                   for kind, seed, path in states]
    gate_states.extend((f"INIT_s{seed}", ttt.build_model(task, st, torch.device("cpu"), seed,
                                                           "e2e1emb").double())
                        for _, seed, _ in inits)
    idx, y, meta = gate_batch
    wm = _write_mask(meta, task.T, torch.device("cpu"))
    pos = torch.as_tensor(meta["query_pos"]); tok = torch.as_tensor(meta["query_tok"])
    for state_name, model in gate_states:
        h_cached = model.backbone(idx).detach()
        ce_logits = ttt.eval_batch_logits(model, task, "e2e1emb", idx, y, ETA_BASE, G_CLIP,
                                          task.chunk, ttt.GradStats(), learned=False, meta=meta)
        ce = ttt.token_ce(ce_logits, y)
        ce_read = {"outer_loss": float(ce.mean()),
                   "binding_acc": float((ce_logits.argmax(-1).gather(1, pos) == tok).double().mean())}
        for cls, vals in (("global", [1, 2, 10, 100]), ("kv", [10, 100, 1000])):
            for val in vals:
                beta = (torch.full((len(idx), task.T), float(val), dtype=torch.float64)
                        if cls == "global" else
                        torch.where(wm, torch.tensor(float(val)), torch.tensor(1.)).to(torch.float64))
                for clip in ("none", "base", "current"):
                    z, hits = _gate_replay(model, idx, y, meta, beta, clip, h_override=h_cached)
                    c = ttt.token_ce(z, y)
                    gate["replay"].append({"state": state_name, "class": cls, "beta": val,
                        "clip": clip, "outer_loss": float(c.mean().detach()),
                        "binding_acc": float((z.argmax(-1).gather(1, pos) == tok).double().mean()),
                        "clip_binding": float(hits.double().mean()), "ce_read": ce_read})
        # Autograd follows the FO recurrence.  FD is evaluated twice: with the
        # recorded (e,x,a,gelu',W2) factors and with the full factor path.
        for clip in ("none", "base", "current"):
            _, _, factors = _gate_replay(model, idx, y, meta,
                                         torch.ones((), dtype=torch.float64), clip,
                                         record_factors=True, h_override=h_cached)
            for cls in ("global", "kv"):
                ell = torch.tensor(0., dtype=torch.float64, requires_grad=True)
                b = (torch.exp(ell) * torch.ones((len(idx), task.T), dtype=torch.float64)
                     if cls == "global" else 1 + (torch.exp(ell) - 1) * wm)
                z, _ = _gate_replay(model, idx, y, meta, b, clip, factor_cache=factors,
                                    h_override=h_cached)
                ag = torch.autograd.grad(ttt.token_ce(z, y).mean(), ell, allow_unused=True)[0]
                aut = float(ag.detach() if ag is not None else 0.)
                # Richardson-extrapolated five-point central FD keeps curvature
                # and fp64 roundoff below the 1e-6 tolerance.
                eps = 1e-2; vals_fixed = {}; vals_true = {}
                for sign in (-2, -1, -0.5, 0.5, 1, 2):
                    q = math.exp(sign * eps)
                    bb = (q * torch.ones((len(idx), task.T), dtype=torch.float64)
                          if cls == "global" else 1 + (q - 1) * wm)
                    zz, _ = _gate_replay(model, idx, y, meta, bb, clip, factor_cache=factors,
                                         h_override=h_cached)
                    zt, _ = _gate_replay(model, idx, y, meta, bb, clip, h_override=h_cached)
                    vals_fixed[sign] = float(ttt.token_ce(zz, y).mean().detach())
                    vals_true[sign] = float(ttt.token_ce(zt, y).mean().detach())
                fd_fixed_h = (vals_fixed[-2] - 8 * vals_fixed[-1] +
                              8 * vals_fixed[1] - vals_fixed[2]) / (12 * eps)
                fd_fixed_half = (vals_fixed[-1] - 8 * vals_fixed[-0.5] +
                                 8 * vals_fixed[0.5] - vals_fixed[1]) / (6 * eps)
                fd_true_h = (vals_true[-2] - 8 * vals_true[-1] +
                             8 * vals_true[1] - vals_true[2]) / (12 * eps)
                fd_true_half = (vals_true[-1] - 8 * vals_true[-0.5] +
                                8 * vals_true[0.5] - vals_true[1]) / (6 * eps)
                fd_fixed = (16 * fd_fixed_half - fd_fixed_h) / 15
                fd_true = (16 * fd_true_half - fd_true_h) / 15
                gate["gradient_check"].append({"state": state_name, "class": cls, "clip": clip,
                    "autograd": aut, "fixed_factor_fd": fd_fixed,
                    "true_derivative": fd_true,
                    "relative_error_fixed": abs(aut - fd_fixed) / max(abs(fd_fixed), 1e-30),
                    "true_to_fo_ratio": fd_true / fd_fixed if abs(fd_fixed) > 1e-30 else None})
    # (b) dropped-term locator; eight fixed batches per endpoint state.
    batches = _fixed_t0_batches(task, 8, 4)
    locator = {}
    for kind, seed, path in states:
        model = _load_state(task, st, path, seed); rows = []
        for idx, y, meta in batches:
            fo, _ = _meta_grads(model, "e2e1emb", idx, y, meta)
            so, _ = _meta_grads(model, "e2e2emb", idx, y, meta)
            for n in fo:
                d = so[n] - fo[n]
                rows.append((_group_name(n), float(d.norm()), float(torch.dot(d.flatten(), fo[n].flatten()) /
                              (d.norm() * fo[n].norm()).clamp_min(1e-30)), d, fo[n]))
        groups = {}
        for gname in sorted(set(r[0] for r in rows)):
            rr = [r for r in rows if r[0] == gname]
            norms = np.asarray([r[1] for r in rr]); cos = np.asarray([r[2] for r in rr])
            groups[gname] = {"norm_mean": float(norms.mean()), "norm_sd": float(norms.std(ddof=1)),
                             "cos_mean": float(cos.mean()), "cos_sd": float(cos.std(ddof=1)),
                             "share_of_total": float(norms.mean() / max(sum(norms.mean() for gg in groups.values()), 1e-30))}
        # Recompute total once all group rows exist; this avoids order-dependent shares.
        den = sum(float(np.mean([r[1] for r in rows if r[0] == gg])) for gg in groups)
        for gg in groups:
            groups[gg]["share_of_total"] = groups[gg]["norm_mean"] / max(den, 1e-30)
        locator[f"{kind}_s{seed}"] = groups
    # (c) exact SOMIX decomposition plus per-mask dispersion and averaged-mask error.
    identity = {"rows": []}
    if states:
        model = _load_state(task, st, states[0][2], states[0][1])
        for bi, (idx, y, meta) in enumerate(batches):
            fo_seq = [_per_sequence_grad(model, idx[i:i+1], y[i:i+1], second=False) for i in range(len(idx))]
            so_seq = [_per_sequence_grad(model, idx[i:i+1], y[i:i+1], second=True) for i in range(len(idx))]
            fo = [sum((g[i] for g in fo_seq), torch.zeros_like(fo_seq[0][i])) / len(idx)
                  for i in range(len(fo_seq[0]))]
            so = [sum((g[i] for g in so_seq), torch.zeros_like(so_seq[0][i])) / len(idx)
                  for i in range(len(so_seq[0]))]
            for pfrac in (.125, .25, .5):
                disp, exact = [], []
                mixed_sum = [torch.zeros_like(q) for q in fo]
                for mi in range(16):
                    mask = somix_mask(200, bi * 16 + mi, len(idx), pfrac, "cpu")
                    composed = []
                    for j in range(len(fo)):
                        q = sum((so_seq[i][j] if bool(mask[i]) else fo_seq[i][j]
                                 for i in range(len(idx))), torch.zeros_like(fo_seq[0][j])) / len(idx)
                        composed.append(q); mixed_sum[j] += q
                    if bi == 0:
                        direct = _mixed_direct_grad(model, idx, y, mask)
                        exact.append(_flat_rel(direct, composed))
                    target = [fo[j] + pfrac * (so[j] - fo[j]) for j in range(len(fo))]
                    disp.append(_flat_rel(composed, target))
                avg = [q / 16 for q in mixed_sum]
                target = [fo[j] + pfrac * (so[j] - fo[j]) for j in range(len(fo))]
                identity["rows"].append({"batch": bi, "p": pfrac, "masks": 16,
                    "exact_decomposition_error_max": float(max(exact)) if exact else None,
                    "dispersion_mean": float(np.mean(disp)),
                    "dispersion_sd": float(np.std(disp, ddof=1)),
                    "mask_averaged_error": _flat_rel(avg, target)})
    result = {"part_a_gate": gate, "part_b_locator": locator, "part_c_identity": identity,
              "states_used": {**{k: len(v) for k, v in paths.items()}, "INIT": len(inits)},
              "fixed_batches": len(batches), "gate_batch_sequences": len(gate_batch[0])}
    (output / "t0.json").write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    md = ["# RTW T0 calibration", "", "## GATE calibration", "",
          "|state|class|beta|clip|loss|binding acc|clip binding|", "|---|---|---:|---|---:|---:|---:|"]
    for r in gate["replay"]:
        md.append(f"|{r['state']}|{r['class']}|{r['beta']}|{r['clip']}|{r['outer_loss']:.6g}|{r['binding_acc']:.4f}|{r['clip_binding']:.4f}|")
    md += ["", "## Fixed-factor gradient check", "", "|state|class|clip|autograd|fixed FD|true FD|rel error|true/FO|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in gate["gradient_check"]:
        md.append(f"|{r['state']}|{r['class']}|{r['clip']}|{r['autograd']:.6g}|{r['fixed_factor_fd']:.6g}|{r['true_derivative']:.6g}|{r['relative_error_fixed']:.3g}|{r['true_to_fo_ratio']:.6g}|")
    md += ["", "## SO − FO locator", "", json.dumps(locator, sort_keys=True), "", "## SOMIX exact decomposition", "", json.dumps(identity, sort_keys=True)]
    (output / "t0.md").write_text("\n".join(md) + "\n")
    print("T0_JSON " + json.dumps({"path": str(output / "t0.json"), "states": result["states_used"]}), flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--t0",action="store_true"); p.add_argument("--output",default=None); a=p.parse_args(argv)
    if a.t0: run_t0(Path(a.output) if a.output else None)
    else: raise SystemExit("use --t0 or run through tttmem.run")


if __name__ == "__main__": main()
