"""Evaluation driver: barrier modes and controls on a fixed document set.

Modes (all use the same eligible documents and probe targets):
- ``adapt``        barrier layout, the given inner rule writes everywhere (primary L)
- ``noadapt``      barrier layout, no inner updates (G = L_noadapt - L)
- ``frozenP``      writes in S1/S2 only
- ``fP_focal``     frozen-P + focal-write suppression
- ``fP_nonfocal``  frozen-P + count-matched non-focal suppression (M = fP_focal - fP_nonfocal)
- ``fP_focal_refclip``  focal suppression with unsuppressed frozen-P clip scales
- ``focal`` / ``nonfocal``  the same suppressions with P writes on (diagnostics)
- ``heldout``      no barrier, full-document mean NLL (Delta_ho; run it on the ``ho`` split)
The prefix is shared across the barrier modes of one batch; modes whose inner rule
needs no logits evaluate the head at probe positions only.
"""

from __future__ import annotations

import numpy as np
import torch

from .meta import InnerCfg, stream_eval
from .model import TTTE2E
from .probe import BARRIERS, PROBE_START, DocProbes, boundary_valid, find_probes

BARRIER_MODES = ("adapt", "noadapt", "frozenP", "fP_focal", "fP_nonfocal", "fP_focal_refclip",
                 "focal", "nonfocal")


def prepare_probes(docs: np.ndarray, C: int, excluded=None, seed: int = 0,
                   bos: int = 128000, barriers=BARRIERS,
                   seeds: list[int] | np.ndarray | None = None) -> list[DocProbes]:
    """docs: [N, T + 1] token ids. Probe sets are text-only and computed once."""
    keys = list(range(len(docs))) if seeds is None else list(seeds)
    if len(keys) != len(docs):
        raise ValueError("seeds must have one entry per document")
    return [find_probes(d[:-1], d[1:], C, valid=d[1:] != bos, excluded=excluded,
                        barriers=barriers, seed=seed + int(keys[i]))
            for i, d in enumerate(docs)]


def adapted_loss(model: TTTE2E, docs: np.ndarray, rule: str, *, batch: int = 8,
                 inner_lr: float = 1.0, inner_clip: float = 1.0, device="cuda",
                 fused_attn: bool = True, barriers=None, r_ref: float = 0.03,
                 data_seed: int = 0, classes: np.ndarray | None = None):
    """Per-document NLL sum and count over every loss-valid target (barrier layout, the rule
    writing everywhere): the SOQ-N3B / P2-3 calibration probe of the from-scratch guard."""
    C, bos = model.cfg.chunk, model.cfg.bos_id
    N, T1 = docs.shape
    bars = list(BARRIERS if barriers is None else barriers)
    bvalid = torch.from_numpy(boundary_valid(T1 - 1, C, bars)).to(device)
    sums, counts = np.zeros(N), np.zeros(N)
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        arr = torch.from_numpy(docs[sl].astype(np.int64)).to(device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        bmask = (tgt != bos) & bvalid
        cls = None if classes is None else torch.from_numpy(np.asarray(classes[sl])).to(device)
        inner = InnerCfg(rule=rule, lr=inner_lr, clip=inner_clip, r_ref=r_ref, seed=data_seed,
                         classes=cls, row_ids=torch.arange(b0, sl.stop, device=device))
        r = stream_eval(model, ids, tgt, bmask, inner, barriers=bars, fused_attn=fused_attn)
        sums[sl] = (r["nll"] * bmask).sum(1).double().cpu().numpy()
        counts[sl] = bmask.sum(1).double().cpu().numpy()
    return sums, counts


def eval_docs(model: TTTE2E, docs: np.ndarray, probes: list[DocProbes], rule: str, *,
              modes=("adapt", "noadapt", "focal", "heldout"), batch: int = 8,
              inner_lr: float = 1.0, inner_clip: float = 1.0, device="cuda",
              fused_attn: bool = True, barriers: tuple | list | None = None,
              classes: np.ndarray | None = None, r_ref: float = 0.03,
              data_seed: int = 0, inner_extra: dict | None = None,
              kappa: list | None = None) -> dict:
    """Returns per-document arrays: ``probe_nll[mode]`` [N, K_i] as lists, ``doc[mode]``
    (probe mean, or full-document mean for ``heldout``), source NLL and clip statistics.
    For ``adapt`` and ``frozenP``, ``src_tok`` / ``src_cot`` hold, per document, the write-time
    NLL and cotangent norm at each focal source position (``p.focal`` order).
    ``inner_extra`` (G0) adds InnerCfg fields such as the token multipliers; ``rule_stats``
    then reports, per frozen-P mode, the write-token mean multiplier and the clip-binding
    fraction over the writing chunks.
    ``kappa`` (SOQ-N3B: one :class:`~tttmem.p2.probe.KappaSet` per document) gathers, inside
    the ``fP_focal`` / ``fP_nonfocal`` forwards, the centred logits and log-probabilities of
    each document's candidates at its query positions: ``kappa_z[mode]`` / ``kappa_lp[mode]``
    hold one [Q_i, V_i] float64 array per document."""
    cfg = model.cfg
    C, bos = cfg.chunk, cfg.bos_id
    inner = InnerCfg(rule=rule, lr=inner_lr, clip=inner_clip)
    N, T1 = docs.shape
    T = T1 - 1
    bars = list(BARRIERS if barriers is None else barriers)
    bvalid = torch.from_numpy(boundary_valid(T, C, bars)).to(device)
    out = {m: np.full(N, np.nan) for m in modes}
    tok = {m: [None] * N for m in modes if m != "heldout"}
    src = {m: np.full(N, np.nan) for m in modes if m in ("adapt", "frozenP")}
    src_tok = {m: [None] * N for m in src}
    src_cot = {m: [None] * N for m in src}
    class_means = {"adapt": {k: np.full(N, np.nan) for k in range(4)},
                   "noadapt": {k: np.full(N, np.nan) for k in range(4)}}
    clip_norm, upd_norm, clip_scale = [], [], []
    stats = {}
    kmodes = [m for m in ("fP_focal", "fP_nonfocal") if m in modes] if kappa is not None else []
    if kappa is not None and len(kappa) != N:
        raise ValueError("kappa needs one KappaSet per document")
    hsum, hcount = np.full(N, np.nan), np.full(N, np.nan)
    bind_doc: dict = {}
    kz = {m: [None] * N for m in kmodes}
    klp = {m: [None] * N for m in kmodes}
    run_modes = list(dict.fromkeys(modes))
    if "fP_focal_refclip" in run_modes:
        run_modes = ["frozenP"] + [m for m in run_modes if m != "frozenP"]
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        arr = torch.from_numpy(docs[sl].astype(np.int64)).to(device, non_blocking=True)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        base = tgt != bos
        bmask = base & bvalid
        B = ids.shape[0]
        pmask = torch.zeros(B, T, dtype=torch.bool, device=device)
        focal = torch.zeros_like(pmask)
        nonfocal = torch.zeros_like(pmask)
        for i, p in enumerate(probes[sl]):
            if p is None:  # held-out-only documents carry no probe set
                continue
            pmask[i, torch.from_numpy(p.pos).to(device)] = True
            focal[i] = torch.from_numpy(p.focal).to(device)
            nonfocal[i] = torch.from_numpy(p.nonfocal).to(device)
        cache: dict = {}
        reference_scales = None
        for m in run_modes:
            cls = None if classes is None else torch.from_numpy(np.asarray(classes[sl])).to(device)
            rule_inner = InnerCfg(rule=rule, lr=inner_lr, clip=inner_clip, r_ref=r_ref,
                                  seed=data_seed, classes=cls,
                                  row_ids=torch.arange(b0, min(N, b0 + batch), device=device),
                                  **(inner_extra or {}))
            kw = dict(barriers=bars, prefix_cache=cache, nll_positions=pmask | focal,
                      fused_attn=fused_attn, return_cot_norm=m in src)
            if m == "adapt":
                r = stream_eval(model, ids, tgt, bmask, rule_inner, **kw)
                if r["inner_norm"] is not None:
                    clip_norm.append(r["inner_norm"].cpu())
                    upd_norm.append(r["update_norm"].cpu())
                    clip_scale.append(r["clip_scale"].cpu())
            elif m == "noadapt":
                r = stream_eval(model, ids, tgt, bmask, rule_inner, adapt=False, **kw)
            elif m == "focal":
                r = stream_eval(model, ids, tgt, bmask, rule_inner, inner_mask=~focal, **kw)
            elif m == "nonfocal":
                r = stream_eval(model, ids, tgt, bmask, rule_inner, inner_mask=~nonfocal, **kw)
            elif m in ("frozenP", "fP_focal", "fP_nonfocal", "fP_focal_refclip"):
                im = torch.ones_like(bmask)
                im[:, PROBE_START * C:] = False
                if m in ("fP_focal", "fP_focal_refclip"):
                    im &= ~focal
                elif m == "fP_nonfocal":
                    im &= ~nonfocal
                if m == "fP_focal_refclip":
                    kw["clip_scales"] = reference_scales
                if m in kmodes:
                    ks = kappa[sl]
                    Q = max([len(k.pos) for k in ks] + [1])
                    V = max([len(k.cand) for k in ks] + [1])
                    gpos = np.full((B, Q), -1, dtype=np.int64)
                    gid = np.zeros((B, Q, V), dtype=np.int64)
                    for i, k in enumerate(ks):
                        gpos[i, :len(k.pos)] = k.pos
                        gid[i, :len(k.pos), :len(k.cand)] = k.cand
                    kw["gather_positions"] = torch.from_numpy(gpos).to(device)
                    kw["gather_ids"] = torch.from_numpy(gid).to(device)
                r = stream_eval(model, ids, tgt, bmask, rule_inner, inner_mask=im, **kw)
                if m in kmodes:
                    gz = r["gather_centred"].double().cpu().numpy()
                    gl = r["gather_logp"].double().cpu().numpy()
                    for i, k in enumerate(ks):
                        kz[m][b0 + i] = gz[i, :len(k.pos), :len(k.cand)]
                        klp[m][b0 + i] = gl[i, :len(k.pos), :len(k.cand)]
                    del kw["gather_positions"], kw["gather_ids"]
                if m == "frozenP":
                    reference_scales = r["clip_scale"]
                st = stats.setdefault(m, {"mult_sum": 0.0, "mult_count": 0, "bind": 0, "writes": 0})
                if "mult_mean" in r:
                    st["mult_sum"] += r["mult_mean"] * r["mult_count"]
                    st["mult_count"] += r["mult_count"]
                cs = r["clip_scale"][:PROBE_START]
                st["bind"] += int((cs < 1).sum().item())
                st["writes"] += int(cs.numel())
                bind_doc.setdefault(m, np.full(N, np.nan))[sl] = \
                    (cs < 1).double().mean(0).cpu().numpy()
            elif m == "heldout":
                # H_LM is the no-adaptation loss in the same segment-blocked
                # training layout.  Keep this call deliberately free of the
                # sparse probe mask: every barrier-valid target contributes.
                rn = stream_eval(model, ids, tgt, bmask, rule_inner, adapt=False,
                                 barriers=bars, prefix_cache=cache, fused_attn=fused_attn)
                no = rn["nll"]
                full = (no * bmask).sum(1) / bmask.sum(1)
                out[m][sl] = full.double().cpu().numpy()
                out.setdefault("heldout_noadapt", np.full(N, np.nan))[sl] = full.double().cpu().numpy()
                hsum[sl] = (no * bmask).sum(1).double().cpu().numpy()
                hcount[sl] = bmask.sum(1).double().cpu().numpy()
                if classes is not None:
                    cc = cls.cpu().numpy()
                    cc = np.where(cc == 0, 1, cc)
                    for k in range(4):
                        selk = cc == k
                        q = no.cpu().numpy()
                        class_means["noadapt"][k][sl] = np.divide((q * selk).sum(1), selk.sum(1),
                            out=np.full(B, np.nan), where=selk.sum(1) > 0)
                continue
            else:
                raise ValueError(m)
            if m not in out:  # frozen-P may be needed only as a clip reference
                continue
            nll = r["nll"].cpu().numpy()
            cot = r["cot_norm"].cpu().numpy() if m in src else None
            for i, p in enumerate(probes[sl]):
                v = nll[i, p.pos]
                tok[m][b0 + i] = v
                out[m][b0 + i] = float(v.mean()) if len(v) else np.nan
                if m in src:
                    src_tok[m][b0 + i] = nll[i, p.focal]
                    src_cot[m][b0 + i] = cot[i, p.focal]
                    if p.focal.any():
                        src[m][b0 + i] = float(nll[i, p.focal].mean())
    res = {"doc": out, "tok": tok, "src_nll": src, "src_tok": src_tok, "src_cot": src_cot,
           "class_means": class_means,
           "rule_stats": {m: {"mult_mean": (st["mult_sum"] / st["mult_count"]
                                            if st["mult_count"] else float("nan")),
                              "clip_bind_frac": (st["bind"] / st["writes"]
                                                 if st["writes"] else float("nan"))}
                          for m, st in stats.items()}}
    if kmodes:
        res["kappa_z"], res["kappa_lp"] = kz, klp
    if bind_doc:  # per-document clip-binding fraction over the frozen-P write chunks (SOQ-N3B)
        res["clip_bind_doc"] = bind_doc
    if "heldout" in modes:  # per-document sums and counts of the H_LM read (SOQ-N3B)
        res["heldout_sum"], res["heldout_count"] = hsum, hcount
    if clip_norm:
        clipped = (torch.cat(clip_norm, 1) > inner_clip).double()
        if inner_clip <= 0:
            clipped.zero_()
        res["clip_frac"] = clipped.mean(1).numpy()                  # per update index
        res["doc_clip_frac"] = clipped.mean(0).numpy()              # per document
        res["clip_scale"] = torch.cat(clip_scale, 1).mean(1).numpy() # per update index
        res["update_norm"] = torch.cat(upd_norm, 1).mean((0, 1)).numpy()
    return res
