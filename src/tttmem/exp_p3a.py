"""P3-a (SOQ-D, claim D): toy write-rule control at a fixed address, load and schedule.

Cells: inner write rule {ce, lin, own} x outer order {FO, SO} x task {echo, echo_ctrl}.
Every cell uses the frozen key code (the ``e2e1emb``/``e2e2emb`` model; both build the
same initialisation from the seed), fixed eta 40, the per-sequence fast-gradient clip 10
and the POOL schedule (1500 steps, batch 32, AdamW 1e-3, cosine over 3000, warmup 200,
grad clip 1).  Only the inner loss whose gradient writes the fast head changes:

- ``ce``:  whole-model next-token CE (bit-identical to the inherited e2e1emb/e2e2emb path);
- ``lin``: the inherited prediction-independent linear token loss
           ``mean_j z_j - z_y`` at every position (bit-identical to ``e2e1emb_lin``);
- ``own``: the fast head's own residual in normalised output space,
           ``1/2 ||n(r(x; W)) - sg(n(U[y]))||^2`` with ``r = W2 gelu(W1 x + b1) + b2`` the
           fast MLP's own output (no ``h``), ``U[y]`` the detached unembedding row of the
           next token and ``n`` the parameter-free LayerNorm (zero mean, unit RMS, eps 1e-5).
           Its error ``n(r) - n(U[y])`` is a delta-rule error on the fast head's own output:
           it contains neither the attention path's prediction nor the readout softmax.
           The normalisation bounds the loss (0 <= l <= 2d); the unnormalised squared
           error diverges at eta 40 in pilot runs.

The outer loss is always the whole-model CE at each chunk's pre-update weights.  FO
detaches the inner gradient (``create_graph=False``), SO keeps it in the graph.  Each
non-CE rule has one multiplier on its inner loss, chosen on CAL seeds so that the median
post-clip fast-gradient norm over the first K training steps matches CE's, then frozen
before any main cell starts (logged as ``P3A_FREEZE``).

The production run is resumable per cell: each finished cell is written atomically and a
restart skips it.  A V0 parity check against the inherited recurrences runs first on the
run's device and aborts with an ``ABORT P3A_PARITY`` line if it fails.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as ttt
from . import p3a_pool
from .triggers import detect

RULES = ("ce", "lin", "own")
ORDERS = ("FO", "SO")
TASKS = ("echo", "echo_ctrl")
NONCE_RULES = ("lin", "own")
MAIN_SEEDS = tuple(range(330, 336))
CAL_SEEDS = tuple(range(340, 343))
SMOKE_SEEDS = tuple(range(0, 10))
ETA_BASE = 40.0
G_CLIP = 10.0
BIND = 0.90
OWN_EPS = 1e-5
# CAL round-0 multipliers, set from smoke-seed pilots (init and 200 steps); the CAL rounds
# move them, and the frozen multiplier is the tested round closest to the target.
CAL_START = {"lin": 1.0, "own": 0.01}
CAL_STATISTIC = ("median over (training steps 1..K, all chunks, all sequences, all CAL seeds, "
                 "both orders, both tasks) of the post-clip per-sequence joint fast-gradient "
                 "norm min(||grad_W m*L_c||, g_clip); the update norm is eta times this")
# keys that do not change the science of a cell (resume and hashing ignore them)
ADMIN_KEYS = {"process_name", "device", "workers", "threads_per_worker", "output_root", "resume",
              "max_hours", "estimate_wall_s", "cutoff_margin_s", "stall_s", "parity",
              "parity_batch", "profile_steps", "mps_memory_fraction", "frozen_multipliers",
              "seeds", "cal_seeds", "smoke", "run_analysis"}


# --------------------------------------------------------------------------- config


def make_task(cfg: dict, name: str) -> ttt.Task:
    """The POOL echo geometry for ``echo`` or ``echo_ctrl``."""
    if name not in TASKS:
        raise ValueError(f"unknown P3-a task {name}")
    n_keys = int(cfg.get("n_keys", 16))
    T, chunk = int(cfg.get("T", 512)), int(cfg.get("chunk", 64))
    if T % chunk:
        raise ValueError(f"T={T} is not a multiple of chunk={chunk}")
    return ttt.Task(name=name, V=ttt.EchoVocab(n_keys).size, T=T,
                    window=int(cfg.get("window", 64)), chunk=chunk,
                    steps=int(cfg.get("steps", 1500)), n_eval=int(cfg.get("n_eval", 256)),
                    n_val=int(cfg.get("n_val", 64)), mode=name, n_keys=n_keys,
                    n_layers=int(cfg.get("n_layers", 4)), prefix_min=0, prefix_max=0)


def science_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in sorted(cfg) if k not in ADMIN_KEYS and not k.startswith("_")}


def science_sha(cfg: dict) -> str:
    blob = json.dumps(science_config(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def check_design(cfg: dict) -> None:
    """Seed registry guard and design sanity, before any model is built."""
    smoke = bool(cfg.get("smoke", False))
    main = [int(s) for s in cfg.get("seeds", [])]
    cal = [int(s) for s in cfg.get("cal_seeds", [])]
    allowed_main, allowed_cal = ((SMOKE_SEEDS, SMOKE_SEEDS) if smoke else (MAIN_SEEDS, CAL_SEEDS))
    bad = [s for s in main if s not in allowed_main] + [s for s in cal if s not in allowed_cal]
    if bad:
        raise ValueError(f"P3-a seeds outside the registered ranges (main 330-335, CAL 340-342"
                         f"{'; smoke 0-9' if smoke else ''}): {bad}")
    if set(main) & set(cal):
        raise ValueError("CAL and main seeds must be disjoint")
    if len(set(main)) != len(main) or len(set(cal)) != len(cal):
        raise ValueError("duplicate seeds")
    if not main:
        raise ValueError("no main seeds")
    if not cal and not cfg.get("frozen_multipliers"):
        raise ValueError("CAL seeds or frozen_multipliers are required")
    for key, allowed in (("rules", RULES), ("orders", ORDERS), ("tasks", TASKS)):
        vals = list(cfg.get(key, allowed))
        if not vals or any(v not in allowed for v in vals) or len(set(vals)) != len(vals):
            raise ValueError(f"invalid {key}: {vals}")
    if "ce" not in cfg.get("rules", RULES):
        raise ValueError("the CE rule is the norm reference and must be present")
    if float(cfg.get("eta_lr", 0.0)) != 0.0:
        raise ValueError("P3-a uses fixed eta: eta_lr must be 0")
    if int(cfg.get("cal_steps", 200)) > int(cfg.get("steps", 1500)):
        raise ValueError("cal_steps K cannot exceed the run length")


# --------------------------------------------------------------------------- rules


def fast_residual(W: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The fast MLP's own output ``r = W2 gelu(W1 x + b1) + b2`` (no residual ``h``)."""
    W1, b1, W2, b2 = W
    u = torch.einsum("...bd,...md->...bm", x, W1) + ttt._bias(b1)
    return torch.einsum("...bm,...dm->...bd", F.gelu(u), W2) + ttt._bias(b2)


def unit_norm(v: torch.Tensor) -> torch.Tensor:
    """Parameter-free LayerNorm over the hidden axis: zero mean, unit RMS (eps 1e-5)."""
    return F.layer_norm(v, v.shape[-1:], eps=OWN_EPS)


def own_tokens(model: ttt.TTTModel, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token own-residual loss ``1/2 ||n(r) - sg(n(U[y]))||^2`` (normalised delta rule)."""
    target = unit_norm(F.embedding(y, model.unembed.weight).detach())
    return 0.5 * (unit_norm(r) - target).square().sum(-1)


def chunk_terms(model: ttt.TTTModel, rule: str, h: torch.Tensor, x: torch.Tensor,
                W: list[torch.Tensor], y: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logits, outer CE tokens, inner tokens)`` of one chunk at weights ``W``.

    The CE and lin branches use exactly the inherited ops (``fast_block`` then ``read``),
    which keeps them bit-identical to ``exp_ttt.sequential_inner``/``eval_sequential``.
    """
    if rule == "own":
        r = fast_residual(W, x)
        z = model.read(h + r)
        return z, ttt.token_ce(z, y), own_tokens(model, r, y)
    z = model.read(ttt.fast_block(h, model.head.ln, W, x))
    ce = ttt.token_ce(z, y)
    if rule == "ce":
        return z, ce, ce
    if rule == "lin":
        return z, ce, ttt.lin_token(z, y)
    raise ValueError(f"unknown P3-a rule {rule}")


def p3a_loss(model: ttt.TTTModel, idx: torch.Tensor, y: torch.Tensor, rule: str, so: bool,
             mult: float, *, eta_base: float = ETA_BASE, g_clip: float = G_CLIP,
             chunk: int = 64, tel: "Telemetry | None" = None) -> torch.Tensor:
    """Outer training loss: the chunked inner recurrence under ``rule``.

    ``W_{c+1} = W_c - eta * clip(grad_W [mult * mean_t inner_t])``; the outer loss is the
    mean CE of every chunk at ``W_c``.  ``so`` keeps the inner gradient in the graph.
    """
    if rule not in RULES:
        raise ValueError(f"unknown P3-a rule {rule}")
    if model.frozen_emb is None:
        raise ValueError("P3-a cells use the frozen key code")
    h = model.backbone(idx)
    x = model.fast_input(h, idx)
    etas = ttt.regime_etas(model, eta_base)
    W = model.per_sequence(idx.shape[0])
    n_chunk = idx.shape[1] // chunk
    total = 0.0
    for c in range(n_chunk):
        sl = slice(c * chunk, (c + 1) * chunk)
        _, ce, inner = chunk_terms(model, rule, h[:, sl], x[:, sl], W, y[:, sl])
        inner_c = inner.mean(1)
        if mult != 1.0:
            inner_c = inner_c * mult
        grads = torch.autograd.grad(inner_c.sum(), W, create_graph=so, retain_graph=True)
        grads, norm, _ = ttt._clip_fast(list(grads), g_clip)
        if tel is not None:
            tel.add(norm)
        total = total + ce.mean(1).mean()
        W = [w - eta * g for w, eta, g in zip(W, etas, grads)]
    return total / n_chunk


@torch.no_grad()
def eval_logits(model: ttt.TTTModel, idx: torch.Tensor, y: torch.Tensor, rule: str, mult: float,
                *, eta: float = ETA_BASE, g_clip: float = G_CLIP, chunk: int = 64,
                norms: list | None = None) -> torch.Tensor:
    """Streaming first-order test-time recurrence under the cell's own rule and multiplier."""
    h = model.backbone(idx).detach()
    x = model.fast_input(h, idx)
    etas = ttt._fixed_etas(model, eta)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(idx.shape[0])]
    out = []
    with torch.enable_grad():
        for c in range(idx.shape[1] // chunk):
            sl = slice(c * chunk, (c + 1) * chunk)
            z, _, inner = chunk_terms(model, rule, h[:, sl], x[:, sl], W, y[:, sl])
            out.append(z.detach())
            inner_c = inner.mean(1)
            if mult != 1.0:
                inner_c = inner_c * mult
            grads = torch.autograd.grad(inner_c.sum(), W)
            grads, norm, _ = ttt._clip_fast(list(grads), g_clip)
            if norms is not None:
                norms.append(norm.detach())
            W = [(w - e * g).detach().requires_grad_(True) for w, e, g in zip(W, etas, grads)]
    return torch.cat(out, dim=1)


def read_metrics(model: ttt.TTTModel, data: list, rule: str, mult: float, *, eta: float,
                 g_clip: float, chunk: int, per_seq: bool = False) -> dict:
    """Binding accuracy at each key's first query (the FRS echo metric) and CEs."""
    hits, qces, ces, norms = [], [], [], []
    for idx, y, meta in data:
        z = eval_logits(model, idx, y, rule, mult, eta=eta, g_clip=g_clip, chunk=chunk,
                        norms=norms)
        ce = ttt.token_ce(z, y)
        pos = torch.as_tensor(np.asarray(meta["query_pos"]), device=z.device, dtype=torch.long)
        tok = torch.as_tensor(np.asarray(meta["query_tok"]), device=z.device, dtype=torch.long)
        hits.append((z.argmax(-1).gather(1, pos) == tok).to(torch.float64))
        qces.append(ce.gather(1, pos).to(torch.float64))
        ces.append(ce.mean(1).to(torch.float64))
    hit = torch.cat(hits).cpu().numpy()
    nrm = torch.cat([n.reshape(-1) for n in norms]).double().cpu().numpy()
    out = dict(acc=float(hit.mean()), query_ce=float(torch.cat(qces).mean()),
               ce=float(torch.cat(ces).mean()),
               eval_norm_median=float(np.median(nrm)),
               eval_clip_frac=float((nrm > g_clip).mean()))
    if per_seq:
        out["acc_per_seq"] = [round(float(v), 6) for v in hit.mean(1)]
    return out


# --------------------------------------------------------------------------- telemetry


class Telemetry:
    """Per-step inner-gradient norms without per-chunk host syncs.

    The pre-clip norms of the first ``K`` steps are kept in full on the device (they are
    the CAL statistic); afterwards only device-side window sums survive, flushed once per
    quick read.
    """

    def __init__(self, K: int, n_chunk: int, batch: int, g_clip: float, dev: torch.device):
        self.K, self.n_chunk, self.batch, self.g_clip = K, n_chunk, batch, g_clip
        self.buf = torch.full((K, n_chunk, batch), float("nan"), device=dev)
        self.step = 0
        self._cur: list[torch.Tensor] = []
        self.win = torch.zeros(3, device=dev, dtype=torch.float64)
        self.tot = torch.zeros(2, device=dev, dtype=torch.float64)
        self.win_n = self.tot_n = 0

    def add(self, norm: torch.Tensor) -> None:
        self._cur.append(norm.detach())

    def end_step(self) -> None:
        if not self._cur:
            return
        n = torch.stack(self._cur).float()
        self._cur = []
        if n.shape != (self.n_chunk, self.batch):
            raise ValueError(f"telemetry shape {tuple(n.shape)} != {(self.n_chunk, self.batch)}")
        if self.step < self.K:
            self.buf[self.step] = n
        post = n.clamp_max(self.g_clip).double()
        clip = (n > self.g_clip).double()
        vec = torch.stack([post.sum(), clip.sum(), post[0].sum()])
        self.win += vec
        self.tot += vec[:2]
        self.win_n += n.numel()
        self.tot_n += n.numel()
        self.step += 1

    def flush_window(self) -> dict:
        v = self.win.cpu().numpy()
        n, n0 = max(1, self.win_n), max(1, self.win_n // self.n_chunk)
        self.win.zero_()
        self.win_n = 0
        return dict(post_mean=float(v[0] / n), clip_frac=float(v[1] / n), post_mean_c0=float(v[2] / n0))

    def first_k(self) -> tuple[dict, np.ndarray]:
        k = min(self.step, self.K)
        pre = self.buf[:k].cpu().numpy().astype(np.float32)
        post = np.minimum(pre, self.g_clip)
        summ = dict(K=k, n=int(pre.size),
                    post_median=float(np.median(post)) if pre.size else None,
                    post_median_c0=float(np.median(post[:, 0])) if pre.size else None,
                    pre_median=float(np.median(pre)) if pre.size else None,
                    clip_frac=float((pre > self.g_clip).mean()) if pre.size else None)
        return summ, pre

    def totals(self) -> dict:
        v = self.tot.cpu().numpy()
        n = max(1, self.tot_n)
        return dict(post_mean=float(v[0] / n), clip_frac=float(v[1] / n), n=int(self.tot_n))


# --------------------------------------------------------------------------- qualifiers


def _flat(grads, n_seq: int) -> torch.Tensor:
    return torch.cat([g.reshape(n_seq, -1) for g in grads], dim=1)


def write_direction(model: ttt.TTTModel, batch: tuple, rule: str, mult: float, chunk: int) -> dict:
    """Chunk-0 write at ``W0``: the rule's fast gradient vs the whole-model CE gradient.

    Per-sequence cosines over the joint (W1, b1, W2, b2) gradient, for all chunk-0 tokens
    and for the write-position tokens only, plus the norm ratio rule/CE (with the rule's
    multiplier).  The cosine is scale-free; for the CE rule it is 1 by construction.
    """
    idx, y, meta = batch
    n = idx.shape[0]
    with torch.no_grad():
        h = model.backbone(idx)
        x = model.fast_input(h, idx)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(n)]
    sl = slice(0, chunk)
    wp = torch.as_tensor(np.asarray(meta["write_pos"]), device=idx.device, dtype=torch.long)
    kv = torch.zeros(n, idx.shape[1], dtype=torch.bool, device=idx.device)
    kv.scatter_(1, wp.clamp(0, idx.shape[1] - 1), True)
    kv = kv[:, sl].to(h.dtype)
    out: dict[str, Any] = {}
    with torch.enable_grad():
        _, ce, inner = chunk_terms(model, rule, h[:, sl], x[:, sl], W, y[:, sl])
        objs = dict(rule_all=mult * inner.mean(1).sum(), ce_all=ce.mean(1).sum(),
                    rule_kv=mult * ((inner * kv).sum(1) / chunk).sum(),
                    ce_kv=((ce * kv).sum(1) / chunk).sum())
        names = list(objs)
        flats = {}
        for i, name in enumerate(names):
            g = torch.autograd.grad(objs[name], W, retain_graph=i < len(names) - 1)
            flats[name] = _flat(g, n)
    for part in ("all", "kv"):
        a, b = flats[f"rule_{part}"], flats[f"ce_{part}"]
        na, nb = a.norm(dim=1), b.norm(dim=1)
        cos = (a * b).sum(1) / (na * nb).clamp_min(1e-30)
        ok = (na > 0) & (nb > 0)
        out[f"cos_{part}"] = float(cos[ok].median()) if bool(ok.any()) else None
        out[f"norm_ratio_{part}"] = float((na[ok] / nb[ok]).median()) if bool(ok.any()) else None
        out[f"rule_norm_{part}"] = float(na.median())
        out[f"ce_norm_{part}"] = float(nb.median())
    return out


def second_order_term(model: ttt.TTTModel, batch: tuple, rule: str, mult: float, *,
                      eta: float, g_clip: float, chunk: int) -> dict:
    """Size of the inner-Jacobian (second-order) path of the outer gradient.

    ``rel = ||g_SO - g_FO|| / ||g_FO||`` over every clipped slow parameter, at the current
    state on a fixed batch, for this cell's rule and multiplier; reported separately for
    the slow fast-head weights ``W0`` and for the rest (backbone, readout).  This is the
    cheap Jacobian qualifier: it costs one FO and one SO outer gradient.
    """
    idx, y, _ = batch
    names = [n for n, _ in model.named_parameters() if n not in ("rho", "rho_kvW2")]
    params = [p for n, p in model.named_parameters() if n not in ("rho", "rho_kvW2")]
    grads = {}
    for order, so in (("FO", False), ("SO", True)):
        loss = p3a_loss(model, idx, y, rule, so, mult, eta_base=eta, g_clip=g_clip, chunk=chunk)
        g = torch.autograd.grad(loss, params, allow_unused=True)
        grads[order] = [torch.zeros_like(p) if gi is None else gi.detach() for p, gi in zip(params, g)]
    out = {}
    groups = dict(all=lambda n: True, head=lambda n: n.startswith("head."),
                  rest=lambda n: not n.startswith("head."))
    for gname, keep in groups.items():
        fo = torch.cat([g.reshape(-1) for n, g in zip(names, grads["FO"]) if keep(n)])
        so = torch.cat([g.reshape(-1) for n, g in zip(names, grads["SO"]) if keep(n)])
        nf = float(fo.norm())
        out[f"rel_{gname}"] = float((so - fo).norm()) / max(nf, 1e-30)
        out[f"cos_{gname}"] = float((so * fo).sum()) / max(nf * float(so.norm()), 1e-30)
        out[f"fo_norm_{gname}"] = nf
    return out


# --------------------------------------------------------------------------- one cell


def cell_key(spec: dict) -> str:
    return f"{spec['task']}__{spec['rule']}__{spec['order']}__s{int(spec['seed']):03d}"


def train_cell(cfg: dict, spec: dict, dev: torch.device) -> tuple[dict, dict]:
    """Train one cell; return its JSON record and its arrays (first-K norms)."""
    t_start = time.time()
    st = ttt._settings(cfg)
    task = make_task(cfg, spec["task"])
    rule, so = spec["rule"], spec["order"] == "SO"
    mult, seed, steps = float(spec["mult"]), int(spec["seed"]), int(spec["steps"])
    if rule == "ce" and mult != 1.0:
        raise ValueError("the CE rule has multiplier 1 by definition")
    if not (math.isfinite(mult) and mult > 0):
        raise ValueError(f"multiplier must be finite and positive, got {mult}")
    main = spec["phase"] == "main"
    eta, g_clip, chunk = float(cfg.get("eta_base", ETA_BASE)), float(st.g_clip), task.chunk
    K = min(int(cfg.get("cal_steps", 200)), steps)
    quick_every = int(cfg.get("quick_every", 25))
    qual_every = int(cfg.get("qual_every", 100))
    log_every = int(cfg.get("log_every", 5))
    total = int(st.lr_total or steps)
    if total < steps:
        raise ValueError("lr_total shorter than the run")
    ttt._reset_peak(dev)
    model = ttt.build_model(task, st, dev, seed, "e2e2emb" if so else "e2e1emb")
    opt = ttt.build_optim(model, st)
    rng = np.random.default_rng([seed, ttt.TRAIN_SEED])
    tel = Telemetry(K, task.T // chunk, st.batch, g_clip, dev)
    quick = None
    if main:
        qraw, qmeta = ttt.fixed_set(task, st, "quick")
        quick = ttt.eval_batches(task, qraw, qmeta, dev, st.eval_batch)
    reads = dict(eta=eta, g_clip=g_clip, chunk=chunk)
    losses, quick_rows, qual_rows, times, outer = [], [], [], [], []
    nonfinite, first_nonfinite, diverged, first_sha = 0, None, False, None
    done = 0
    for step in range(steps):
        scale = ttt.lr_scale(step, st.warmup, total, st.lr_floor)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        raw, meta = task.raw(rng, st.batch)
        if first_sha is None:
            first_sha = hashlib.sha1(raw[:, :task.T].tobytes()).hexdigest()
        idx, y, meta = task.tensors(raw, meta, dev)
        ttt._sync(dev)
        t0 = time.perf_counter()
        loss = p3a_loss(model, idx, y, rule, so, mult, eta_base=eta, g_clip=g_clip,
                        chunk=chunk, tel=tel)
        opt.zero_grad(set_to_none=True)
        finite = bool(torch.isfinite(loss))  # one host sync per step
        if finite:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(ttt.clipped_params(model), st.grad_clip)
            rho = (model.rho.grad.norm() if model.rho.grad is not None
                   else torch.zeros((), device=norm.device)).to(norm.device)
            finite = bool(torch.isfinite(torch.stack([norm.float(), rho.float()])).all())
            if finite:
                opt.step()
                outer.append(norm.detach())
        if not finite:
            nonfinite += 1
            first_nonfinite = first_nonfinite or step + 1
            opt.zero_grad(set_to_none=True)
            print(f"  WARNING P3A {cell_key(spec)}: skipped non-finite step {step + 1}", flush=True)
        tel.end_step()
        ttt._sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
        done = step + 1
        if done % log_every == 0 or done == steps:
            losses.append([done, float(loss.detach())])
        if main and (done % quick_every == 0 or done == steps):
            own = read_metrics(model, quick, rule, mult, **reads)
            ce_read = own if rule == "ce" else read_metrics(model, quick, "ce", 1.0, **reads)
            quick_rows.append(dict(step=done, acc_own=own["acc"], acc_ce=ce_read["acc"],
                                   qce_own=own["query_ce"], qce_ce=ce_read["query_ce"],
                                   ce_own=own["ce"], loss=float(loss.detach()),
                                   train=tel.flush_window()))
            if done % (quick_every * 8) == 0 or done == steps:
                print(f"P3A {cell_key(spec)} step={done} loss={float(loss.detach()):.4f} "
                      f"acc_own={own['acc']:.3f} acc_ce={ce_read['acc']:.3f} "
                      f"ms={float(np.median(times[-quick_every:])):.1f}", flush=True)
        if main and (done % qual_every == 0 or done == steps):
            qual_rows.append(dict(step=done,
                                  write=write_direction(model, quick[0], rule, mult, chunk),
                                  so_term=second_order_term(model, quick[0], rule, mult, **reads)))
        if done >= 20 and nonfinite / done > 0.05:
            diverged = True
            print(f"  WARNING P3A {cell_key(spec)}: diverged after {done} steps", flush=True)
            break
    first_k, pre = tel.first_k()
    lo = min(100, max(1, len(times) // 2))
    outer_np = torch.stack(outer).float().cpu().numpy() if outer else np.zeros(0)
    rec: dict[str, Any] = dict(
        key=cell_key(spec), **spec, steps_done=done, eta_base=eta, g_clip=g_clip,
        science_sha256=science_sha(cfg), device=str(dev), torch=torch.__version__,
        first_batch_sha1=first_sha, ms_step=float(np.median(times[lo:] or times)),
        peak_memory_mb=ttt._peak_mem(dev), nonfinite_steps=nonfinite,
        first_nonfinite=first_nonfinite, diverged=diverged, first_k=first_k,
        train_tel=tel.totals(), loss_log=losses,
        outer_grad=dict(median=float(np.median(outer_np)) if outer_np.size else None,
                        max=float(outer_np.max()) if outer_np.size else None,
                        clip_frac=float((outer_np > st.grad_clip).mean()) if outer_np.size else None))
    if main:
        ev = ttt.eval_batches(task, *ttt.fixed_set(task, st, "eval"), dev, st.eval_batch)
        own = read_metrics(model, ev, rule, mult, per_seq=True, **reads)
        ce_read = own if rule == "ce" else read_metrics(model, ev, "ce", 1.0, per_seq=True, **reads)
        events = detect([dict(step=r["step"], quick=dict(acc=r["acc_own"])) for r in quick_rows],
                        bind=BIND, grid=quick_every)
        last = [r["acc_own"] for r in quick_rows[-2:]]
        last_ce = [r["acc_ce"] for r in quick_rows[-2:]]
        defined = (not diverged and all(math.isfinite(v) for v in (own["acc"], own["query_ce"])))
        rec.update(quick=quick_rows, qualifiers=qual_rows, final=dict(own=own, ce=ce_read),
                   t_acq=events["t_acq"].event_step if "t_acq" in events else None,
                   t_loss=events["t_loss"].event_step if "t_loss" in events else None,
                   ever_acquired="t_acq" in events, defined=defined,
                   retained=bool(defined and len(last) == 2 and min(last) >= BIND
                                 and own["acc"] >= BIND),
                   retained_ce=bool(defined and len(last_ce) == 2 and min(last_ce) >= BIND
                                    and ce_read["acc"] >= BIND))
    rec["wall_s"] = round(time.time() - t_start, 2)
    return rec, dict(first_k_preclip=pre)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_cell(folder: Path, rec: dict, arrays: dict) -> None:
    """Arrays first, then the JSON record: a present ``result.json`` means a whole cell."""
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "first_k.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(folder / "first_k.npz")
    _atomic_write(folder / "result.json", (json.dumps(rec, sort_keys=True) + "\n").encode())


def init_worker(cfg: dict) -> None:
    """Per-process thread caps, name, device options; runs once in each pool child."""
    threads = min(4, max(1, int(cfg.get("threads_per_worker", 2))))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    index = p3a_pool.worker_index()
    if index:
        ttt._mark_process(f"{cfg.get('process_name', 'TTM-s3a-01')}-w{index}")
    dev = ttt._device(str(cfg.get("device", "auto")))
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(float(cfg.get("mps_memory_fraction") or 0.25))
    ttt.configure_tf32(bool(cfg.get("tf32", False)), f"w{index}")


def run_cell(job: dict) -> dict:
    """Pool entry point: train one cell and persist it (unless past the start cutoff)."""
    if job.get("cutoff") is not None and time.time() > float(job["cutoff"]):
        return dict(status="deadline", key=job["key"])
    cfg = job["cfg"]
    dev = ttt._device(str(cfg.get("device", "auto")))
    rec, arrays = train_cell(cfg, job["spec"], dev)
    write_cell(Path(job["folder"]), rec, arrays)
    return dict(status="done", key=job["key"], **{k: rec.get(k) for k in (
        "ms_step", "peak_memory_mb", "nonfinite_steps", "diverged", "retained", "ever_acquired",
        "wall_s", "steps_done")}, acc_own=(rec.get("final") or {}).get("own", {}).get("acc"),
        post_median=rec["first_k"]["post_median"])


# --------------------------------------------------------------------------- V0 parity


def _rel(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    num = math.sqrt(sum(float((x - z).detach().double().square().sum()) for x, z in zip(a, b)))
    den = math.sqrt(sum(float(z.detach().double().square().sum()) for z in b))
    return num / max(den, 1e-30)


def own_closed_form(model: ttt.TTTModel, x: torch.Tensor, W: list[torch.Tensor],
                    y: torch.Tensor, mult: float = 1.0) -> list[torch.Tensor]:
    """Analytic per-sequence gradient of ``mult * mean_t 1/2 ||n(r_t) - n(U[y_t])||^2``.

    With ``n = (r - mean r) / s``, ``s = sqrt(var r + eps)`` and ``e = n - t``, the loss
    gradient at ``r`` is the LayerNorm backward ``(e - mean e - n mean(n e)) / s``; the rest
    is the chain rule through ``r = W2 gelu(W1 x + b1) + b2``.
    """
    W1, b1, W2, b2 = W
    b = x.shape[1]
    u = torch.einsum("btd,bmd->btm", x, W1) + b1[:, None]
    a = F.gelu(u)
    r = torch.einsum("btm,bdm->btd", a, W2) + b2[:, None]
    tu = F.embedding(y, model.unembed.weight)
    tu = tu - tu.mean(-1, keepdim=True)
    t = tu / torch.sqrt(tu.square().mean(-1, keepdim=True) + OWN_EPS)
    rc = r - r.mean(-1, keepdim=True)
    sd = torch.sqrt(rc.square().mean(-1, keepdim=True) + OWN_EPS)
    n = rc / sd
    e = n - t
    g = (e - e.mean(-1, keepdim=True) - n * (n * e).mean(-1, keepdim=True)) / sd
    e = mult * g / b
    gp = 0.5 * (1 + torch.erf(u / math.sqrt(2))) + u * torch.exp(-u.square() / 2) / math.sqrt(2 * math.pi)
    eps = torch.einsum("btd,bdm->btm", e, W2) * gp
    return [torch.einsum("btm,btd->bmd", eps, x), eps.sum(1),
            torch.einsum("btd,btm->bdm", e, a), e.sum(1)]


def parity_check(cfg: dict, dev: torch.device, *, tol: float | None = None) -> dict:
    """V0: the new recurrence against the inherited paths, on the run's own device.

    - CE FO/SO training loss and all outer gradients vs ``exp_ttt.arm_loss`` (e2e1emb/e2e2emb);
    - lin FO/SO vs ``exp_ttt.sequential_inner`` with the all-position selector;
    - own: autograd vs the closed-form normalised delta-rule gradient; the target
      (unembedding row) receives no gradient from the inner loss;
    - eval: CE and lin logits vs ``eval_batch_logits`` / ``eval_sequential``;
    - the multiplier scales the pre-clip inner gradient linearly.
    """
    st = ttt._settings(cfg)
    task = make_task(cfg, "echo")
    tol = float(cfg.get("parity_tol", 1e-5) if tol is None else tol)
    B = int(cfg.get("parity_batch", 4))
    raw, meta = task.raw(np.random.default_rng([ttt.EVAL_SEED, 77]), B)
    idx, y, meta = task.tensors(raw, meta, dev)
    eta, g_clip, chunk = float(cfg.get("eta_base", ETA_BASE)), float(st.g_clip), task.chunk
    rows: dict[str, float] = {}

    def params(m):
        return [p for n, p in m.named_parameters() if n != "rho"]

    for order, arm in (("FO", "e2e1emb"), ("SO", "e2e2emb")):
        so = order == "SO"
        model = ttt.build_model(task, st, dev, 0, arm)
        ref = ttt.arm_loss(model, arm, idx, y, eta, g_clip, chunk, ttt.GradStats(), meta=meta)
        g_ref = torch.autograd.grad(ref, params(model))
        new = p3a_loss(model, idx, y, "ce", so, 1.0, eta_base=eta, g_clip=g_clip, chunk=chunk)
        g_new = torch.autograd.grad(new, params(model))
        new, ref = new.detach(), ref.detach()
        rows[f"ce_{order}_loss"] = abs(float(new) - float(ref)) / max(abs(float(ref)), 1e-30)
        rows[f"ce_{order}_grad"] = _rel(list(g_new), list(g_ref))
        h = model.backbone(idx)
        sel = torch.ones(B, task.T, dtype=torch.bool, device=dev)
        ref = ttt.sequential_inner(model, h, y, model.per_sequence(B), ttt.regime_etas(model, eta),
                                   g_clip, so, ttt.GradStats(), chunk, idx, sel=sel, rule="lin")
        g_ref = torch.autograd.grad(ref, params(model))
        new = p3a_loss(model, idx, y, "lin", so, 1.0, eta_base=eta, g_clip=g_clip, chunk=chunk)
        g_new = torch.autograd.grad(new, params(model))
        new, ref = new.detach(), ref.detach()
        rows[f"lin_{order}_loss"] = abs(float(new) - float(ref)) / max(abs(float(ref)), 1e-30)
        rows[f"lin_{order}_grad"] = _rel(list(g_new), list(g_ref))
    model = ttt.build_model(task, st, dev, 0, "e2e1emb")
    with torch.no_grad():
        h = model.backbone(idx)
        x = model.fast_input(h, idx)
    sl = slice(0, chunk)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(B)]
    for mult in (1.0, 2.5):
        _, _, inner = chunk_terms(model, "own", h[:, sl], x[:, sl], W, y[:, sl])
        g_auto = torch.autograd.grad(mult * inner.mean(1).sum(), W)
        rows[f"own_closed_form_m{mult:g}"] = _rel(list(g_auto), own_closed_form(model, x[:, sl], W, y[:, sl], mult))
    _, _, inner = chunk_terms(model, "own", h[:, sl], x[:, sl], W, y[:, sl])
    g_u = torch.autograd.grad(inner.sum(), model.unembed.weight, allow_unused=True)[0]
    rows["own_target_grad_is_none"] = 0.0 if g_u is None else 1.0
    g1 = torch.autograd.grad(chunk_terms(model, "lin", h[:, sl], x[:, sl], W, y[:, sl])[2].mean(1).sum(), W)
    g2 = torch.autograd.grad(3.0 * chunk_terms(model, "lin", h[:, sl], x[:, sl], W, y[:, sl])[2].mean(1).sum(), W)
    rows["multiplier_linear"] = _rel(list(g2), [3.0 * g for g in g1])
    model = ttt.build_model(task, st, dev, 0, "e2e1emb")
    ref = ttt.eval_batch_logits(model, task, "e2e1emb", idx, y, eta, g_clip, chunk,
                                ttt.GradStats(), learned=False, meta=meta)
    new = eval_logits(model, idx, y, "ce", 1.0, eta=eta, g_clip=g_clip, chunk=chunk)
    rows["eval_ce"] = _rel([new], [ref])
    sel = torch.ones(B, task.T, dtype=torch.bool, device=dev)
    ref = ttt.eval_sequential(model, idx, y, ttt._fixed_etas(model, eta), g_clip, chunk,
                              ttt.GradStats(), sel, rule="lin")
    new = eval_logits(model, idx, y, "lin", 1.0, eta=eta, g_clip=g_clip, chunk=chunk)
    rows["eval_lin"] = _rel([new], [ref])
    own_tol = max(tol, 1e-4) if torch.get_default_dtype() == torch.float32 else tol
    fails = {k: v for k, v in rows.items()
             if not math.isfinite(v) or v > (own_tol if k.startswith("own_closed") else tol)}
    return dict(ok=not fails, tol=tol, own_tol=own_tol, rows=rows, fails=fails, device=str(dev))


def profile(cfg: dict, dev: torch.device, steps: int) -> dict:
    """Single-process ms/step per (rule, order) on the run's device, full geometry."""
    st = ttt._settings(cfg)
    task = make_task(cfg, "echo")
    rng = np.random.default_rng([ttt.EVAL_SEED, 78])
    raw, meta = task.raw(rng, st.batch)
    idx, y, _ = task.tensors(raw, meta, dev)
    out = {}
    for rule in cfg.get("rules", RULES):
        for order in cfg.get("orders", ORDERS):
            model = ttt.build_model(task, st, dev, 0, "e2e2emb" if order == "SO" else "e2e1emb")
            times = []
            for i in range(steps + 2):
                ttt._sync(dev)
                t0 = time.perf_counter()
                loss = p3a_loss(model, idx, y, rule, order == "SO", 1.0,
                                eta_base=float(cfg.get("eta_base", ETA_BASE)), g_clip=st.g_clip,
                                chunk=task.chunk)
                loss.backward()
                model.zero_grad(set_to_none=True)
                ttt._sync(dev)
                if i >= 2:
                    times.append((time.perf_counter() - t0) * 1e3)
            out[f"{rule}_{order}"] = float(np.median(times))
    return out


# --------------------------------------------------------------------------- calibration


def next_multiplier(history: list[tuple[float, float]], target: float,
                    lo: float = 0.25, hi: float = 4.0) -> float:
    """Log-space secant step toward ``median(m) = target``.

    ``history`` holds ``(m_i, median_i)``.  With one point the median is taken as
    proportional to ``m`` (slope 1); with two, the log-log slope of the last two points,
    clamped to ``[lo, hi]``.
    """
    m, med = history[-1]
    if not (med > 0 and target > 0 and m > 0):
        raise ValueError(f"non-positive CAL values m={m} median={med} target={target}")
    a = 1.0
    if len(history) >= 2:
        m0, med0 = history[-2]
        if med0 > 0 and m0 > 0 and abs(math.log(m) - math.log(m0)) > 1e-12:
            a = (math.log(med) - math.log(med0)) / (math.log(m) - math.log(m0))
            a = min(hi, max(lo, a)) if math.isfinite(a) else 1.0
    return float(m * (target / med) ** (1.0 / a))


def cal_start(cfg: dict, rule: str) -> float:
    """Round-0 multiplier of a non-CE rule (config ``cal_start`` overrides ``CAL_START``)."""
    return float(dict(CAL_START, **cfg.get("cal_start", {}))[rule])


def closest_round(history: list[tuple[float, float]], target: float) -> float:
    """The tested multiplier whose CAL median is closest to ``target`` in log ratio.

    Freezing a tested value (not the secant's next, untested step) keeps the frozen
    multiplier inside the range whose first-K behaviour CAL has observed; the bounded
    rules' norm-vs-multiplier curves are flat, so an extrapolated step can be large.
    """
    ok = [(m, med) for m, med in history if m > 0 and med > 0 and math.isfinite(med)]
    if not ok or not target > 0:
        raise ValueError(f"no usable CAL rounds for target {target}: {history}")
    return float(min(ok, key=lambda p: (abs(math.log(p[1] / target)), p[0]))[0])


def pooled_post(folders: list[Path], g_clip: float) -> np.ndarray:
    vals = []
    for folder in folders:
        with np.load(folder / "first_k.npz") as z:
            pre = z["first_k_preclip"]
        vals.append(np.minimum(pre[np.isfinite(pre)], g_clip).reshape(-1))
    return np.concatenate(vals) if vals else np.zeros(0)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def _check_resumed(rec: dict, spec: dict, cfg: dict, path: Path) -> None:
    same = all(rec.get(k) == v for k, v in spec.items()) and rec.get("science_sha256") == science_sha(cfg)
    if not same:
        raise ValueError(f"stale P3-a cell output does not match this run: {path}")


def run_specs(pool: p3a_pool.CellPool, cfg: dict, specs: list[dict], folder: Path, *,
              cutoff: float | None, stall_s: float | None) -> dict:
    """Run every spec without a matching ``result.json``; return per-key status."""
    status, jobs = {}, []
    for spec in specs:
        key = cell_key(spec)
        sub = folder / key
        file = sub / "result.json"
        if file.exists() and cfg.get("resume", True):
            _check_resumed(_json_load(file), spec, cfg, file)
            status[key] = "resumed"
            print(f"P3A_RESUME {spec['phase']} {key}", flush=True)
            continue
        jobs.append(dict(key=key, spec=spec, cfg=cfg, folder=str(sub), cutoff=cutoff))
    for key, res, err in pool.run(jobs, stall_s=stall_s):
        if err is not None:
            status[key] = "error"
            print(f"P3A_CELL_ERROR {key}\n{err}", flush=True)
        elif res.get("status") == "deadline":
            status[key] = "deadline"
            print(f"P3A_CELL_DEADLINE {key}", flush=True)
        else:
            status[key] = "done"
            print("P3A_CELL_DONE " + json.dumps(res, sort_keys=True), flush=True)
    return status


def cal_specs(cfg: dict, round_index: int, mults: dict) -> list[dict]:
    rules = [r for r in cfg.get("rules", RULES) if round_index == 0 or r != "ce"]
    K = int(cfg.get("cal_steps", 200))
    return [dict(phase="cal", round=round_index, task=t, rule=r, order=o, seed=int(s),
                 mult=1.0 if r == "ce" else float(mults[r]), steps=K)
            for s in cfg["cal_seeds"] for t in cfg.get("tasks", TASKS) for r in rules
            for o in cfg.get("orders", ORDERS)]


def calibrate(cfg: dict, pool: p3a_pool.CellPool, root: Path, *, cutoff: float | None,
              stall_s: float | None) -> dict:
    """CAL rounds -> frozen multipliers, persisted before any main cell starts."""
    cal = root / "cal"
    mult_file, state_file = cal / "multipliers.json", cal / "state.json"
    nonce = [r for r in cfg.get("rules", RULES) if r != "ce"]
    g_clip = float(cfg.get("g_clip", G_CLIP))
    if cfg.get("frozen_multipliers"):
        frozen = {r: float(cfg["frozen_multipliers"][r]) for r in nonce}
        record = dict(source="config", ce=1.0, **frozen)
        _atomic_write(mult_file, (json.dumps(record, sort_keys=True, indent=1) + "\n").encode())
        return record
    if mult_file.exists() and cfg.get("resume", True):
        record = _json_load(mult_file)
        if record.get("science_sha256") != science_sha(cfg):
            raise ValueError("frozen multipliers belong to a different science config")
        print("P3A_FREEZE_RESUMED " + json.dumps({r: record[r] for r in nonce}), flush=True)
        return record
    state = _json_load(state_file) if state_file.exists() and cfg.get("resume", True) else {}
    if state and state.get("science_sha256") != science_sha(cfg):
        raise ValueError("CAL state belongs to a different science config")
    state.setdefault("science_sha256", science_sha(cfg))
    state.setdefault("rounds", [])
    rounds = int(cfg.get("cal_rounds", 3))

    def history(rule):
        return [(rd["m"][rule], rd["median"][rule]) for rd in state["rounds"] if "median" in rd]

    for r in range(rounds):
        if cutoff is not None and time.time() > cutoff:
            raise TimeoutError("deadline reached before the CAL multipliers were frozen")
        rd = state["rounds"][r] if r < len(state["rounds"]) else None
        if rd is not None and "median" in rd:
            continue
        if rd is None:
            m = {rule: cal_start(cfg, rule) if r == 0
                 else next_multiplier(history(rule), state["target"]) for rule in nonce}
            rd = dict(round=r, m=m)
            state["rounds"].append(rd)
            _atomic_write(state_file, (json.dumps(state, sort_keys=True, indent=1) + "\n").encode())
        specs = cal_specs(cfg, r, rd["m"])
        status = run_specs(pool, cfg, specs, cal / f"r{r}", cutoff=None, stall_s=stall_s)
        bad = {k: v for k, v in status.items() if v not in ("done", "resumed")}
        if bad:
            raise RuntimeError(f"CAL round {r} incomplete: {bad}")
        folders = {rule: [cal / f"r{r}" / cell_key(s) for s in specs if s["rule"] == rule]
                   for rule in {s["rule"] for s in specs}}
        if r == 0:
            ce_vals = pooled_post(folders["ce"], g_clip)
            state["target"] = float(np.median(ce_vals))
            state["target_n"] = int(ce_vals.size)
        rd["median"], rd["n"] = {}, {}
        for rule in nonce:
            vals = pooled_post(folders[rule], g_clip)
            rd["median"][rule] = float(np.median(vals))
            rd["n"][rule] = int(vals.size)
        rd["ratio_target_over_median"] = {k: state["target"] / v for k, v in rd["median"].items()}
        ms = [_json_load(f / "result.json")["ms_step"] for fs in folders.values() for f in fs]
        rd["ms_step_median"] = float(np.median(ms))
        _atomic_write(state_file, (json.dumps(state, sort_keys=True, indent=1) + "\n").encode())
        print("P3A_CAL_ROUND " + json.dumps(dict(round=r, target=state["target"], m=rd["m"],
                                                 median=rd["median"],
                                                 ms_step_median=rd["ms_step_median"]),
                                            sort_keys=True), flush=True)
    frozen = {rule: closest_round(history(rule), state["target"]) for rule in nonce}
    record = dict(source="cal", ce=1.0, **frozen, K=int(cfg.get("cal_steps", 200)),
                  freeze_rule="tested CAL round with the smallest |log(median / target)|",
                  statistic=CAL_STATISTIC, target=state["target"], target_n=state.get("target_n"),
                  rounds=state["rounds"], cal_seeds=list(cfg["cal_seeds"]),
                  science_sha256=science_sha(cfg),
                  frozen_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    blob = (json.dumps(record, sort_keys=True, indent=1) + "\n").encode()
    _atomic_write(mult_file, blob)
    print("P3A_FREEZE " + json.dumps(dict(**frozen, target=state["target"],
                                          sha256=hashlib.sha256(blob).hexdigest()),
                                     sort_keys=True), flush=True)
    return record


def main_specs(cfg: dict, mults: dict) -> list[dict]:
    """Seed-major order: a cut run still holds complete seeds."""
    steps = int(cfg.get("steps", 1500))
    return [dict(phase="main", task=t, rule=r, order=o, seed=int(s),
                 mult=1.0 if r == "ce" else float(mults[r]), steps=steps)
            for s in cfg["seeds"] for t in cfg.get("tasks", TASKS)
            for r in cfg.get("rules", RULES) for o in cfg.get("orders", ORDERS)]


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - a source snapshot may not be a git checkout
        return None


# --------------------------------------------------------------------------- entry point


def run(cfg: dict, seed: int = 0) -> dict:
    """Parity -> CAL (frozen multipliers) -> main cells -> summary and analysis."""
    t0 = time.time()
    check_design(cfg)
    threads = min(4, max(1, int(cfg.get("threads_per_worker", 2))))
    torch.set_num_threads(threads)
    ttt._mark_process(str(cfg.get("process_name", "TTM-s3a-01")))
    dev = ttt._device(str(cfg.get("device", "auto")))
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(float(cfg.get("mps_memory_fraction") or 0.25))
    ttt.configure_tf32(bool(cfg.get("tf32", False)), "parent")
    root = Path(cfg.get("output_root", "outputs/soq/p3a"))
    root.mkdir(parents=True, exist_ok=True)
    max_hours = float(cfg.get("max_hours", 1.4))
    cutoff = t0 + max_hours * 3600 - float(cfg.get("cutoff_margin_s", 420))
    stall_s = float(cfg.get("stall_s", 1800))
    manifest = dict(experiment="p3a", science_sha256=science_sha(cfg), config=cfg,
                    git_commit=_git_commit(), torch=torch.__version__, python=platform.python_version(),
                    device=str(dev), device_name=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else
                                                  platform.processor()),
                    started=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    _atomic_write(root / f"run_manifest_{int(t0)}.json",
                  (json.dumps(manifest, sort_keys=True, indent=1, default=str) + "\n").encode())
    print("P3A_START " + json.dumps(dict(science_sha256=manifest["science_sha256"], device=str(dev),
                                         device_name=manifest["device_name"], root=str(root),
                                         git_commit=manifest["git_commit"], max_hours=max_hours),
                                    sort_keys=True), flush=True)
    if cfg.get("parity", True):
        par = parity_check(cfg, dev)
        _atomic_write(root / "parity.json", (json.dumps(par, sort_keys=True, indent=1) + "\n").encode())
        if not par["ok"]:
            print("ABORT P3A_PARITY " + json.dumps(par["fails"], sort_keys=True), flush=True)
            raise RuntimeError(f"P3-a V0 parity failed: {par['fails']}")
        print("P3A_PARITY ok " + json.dumps({k: f"{v:.2e}" for k, v in par["rows"].items()},
                                            sort_keys=True), flush=True)
    if int(cfg.get("profile_steps", 0)) > 0:
        prof = profile(cfg, dev, int(cfg["profile_steps"]))
        print("P3A_PROFILE single_process_ms_step " + json.dumps(prof, sort_keys=True), flush=True)
    workers = max(1, int(cfg.get("workers", 1)))
    with p3a_pool.CellPool(workers, "tttmem.exp_p3a:run_cell", "tttmem.exp_p3a:init_worker",
                           (cfg,)) as pool:
        mults = calibrate(cfg, pool, root, cutoff=cutoff, stall_s=stall_s)
        specs = main_specs(cfg, mults)
        if mults.get("rounds"):
            ms = mults["rounds"][-1].get("ms_step_median")
            if ms:
                waves = math.ceil(len(specs) / workers)
                proj = waves * int(cfg.get("steps", 1500)) * ms / 1e3 * 1.12
                print(f"P3A_PROJECTION main_wall_s={proj:.0f} (CAL ms/step {ms:.1f} at "
                      f"workers={workers}, {waves} waves, +12% eval)", flush=True)
        status = run_specs(pool, cfg, specs, root / "main", cutoff=cutoff, stall_s=stall_s)
    counts = {s: sum(v == s for v in status.values()) for s in ("done", "resumed", "error", "deadline")}
    summary = dict(experiment="p3a", wall_s=round(time.time() - t0, 1), counts=counts,
                   multipliers={r: mults[r] for r in cfg.get("rules", RULES)},
                   n_main=len(specs), status=status)
    _atomic_write(root / f"summary_{int(t0)}.json",
                  (json.dumps(summary, sort_keys=True, indent=1) + "\n").encode())
    print("P3A_SUMMARY " + json.dumps(dict(wall_s=summary["wall_s"], counts=counts,
                                           multipliers=summary["multipliers"]), sort_keys=True),
          flush=True)
    verdict = None
    if cfg.get("run_analysis", True):
        from . import p3a_analysis
        result = p3a_analysis.analyze(root)
        out = root / "analysis"
        _atomic_write(out / "analysis.json", (json.dumps(result, sort_keys=True, indent=1) + "\n").encode())
        _atomic_write(out / "analysis.md", p3a_analysis.render_md(result).encode())
        verdict = result["verdict"]
        print("P3A_VERDICT " + json.dumps(verdict, sort_keys=True), flush=True)
    complete = counts["error"] == 0 and counts["deadline"] == 0
    return dict(complete=complete, counts=counts, multipliers=summary["multipliers"],
                verdict=verdict, wall_s=summary["wall_s"], output_root=str(root))
