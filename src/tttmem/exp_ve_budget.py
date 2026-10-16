"""E1-VE-D2: the write budget behind the A failure, and how much stronger the C write must be.

Eval only. E1-VE-D found that with the C copy edge blocked the C
write of v1 is readable at revisit, but its share of the A margin
``z[v1] - z[v0]`` (2.6 to 9.3 logits) is smaller than the E binding's (5.2 to
8.4), and that a beta = .5 erase or add at the A write rescues half the cells.
On the same 60 checkpoints and 256 episodes this module measures:

* **budget**: for every chunk, the joint pre-clip norm of the fast gradient and
  the clip scale ``min(1, g_clip / norm)``; for chunks 0 (E) and C, the norm of
  each fast tensor's update, of the whole chunk and of the location's own
  writes only (the E or C write of each A or X location, same clip scale);
* **trajectory**: the A and X margins, and the shares of the chunk updates and
  of the own writes in them, read at the R query positions with the weights
  after each chunk ``k >= C``. The probe is the stream's own ``fast_block``; at
  ``k = chunk(q) - 1`` it is the stream's read of query ``q`` (checked);
* **beta sweep**: ``erase_v0`` and ``add_v1`` at the A writes for every beta in
  ``betas``, and ``erase_rand`` at the largest registered beta, in every cell
  whose base ``acc_A`` is below ``sweep_below``. For a CE write ``add_v1`` at
  beta weights the A-write CE term by ``1 + beta`` before clipping.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import exp_ttt as tt
from . import exp_ve as ve
from . import exp_ve_diag as dg

BETAS = (0.5, 1.0, 1.5, 2.0, 3.0)
S2_BETA = 2.0          # registered dose for S2 / S2c
R0_ACC_TOL = 0.005     # beta = .5 must reproduce E1-VE-D within this (acc) ...
R0_MARGIN_TOL = 0.01   # ... and this (median margin, logits)
OWN_TYPES = ("A", "X")


@dataclass
class BudgetSettings:
    process_name: str = "TTM-vd2-01"
    device: str = "cpu"
    threads: int = 8
    ckpt_dir: str = ""
    reference_csv: str = ""     # wave-1 per-cell finals (gate G0)
    diag_csv: str = ""          # E1-VE-D per-cell records (gate R0, E/C ratio for S3)
    betas: list = field(default_factory=lambda: list(BETAS))
    sweep_below: float = 0.90
    n_eval: int = 256
    eval_batch: int = 32
    ve: dict = field(default_factory=dict)
    cells: list = field(default_factory=list)


def settings(cfg: dict) -> BudgetSettings:
    known = {f.name for f in dataclasses.fields(BudgetSettings)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown [ve_budget] keys: {unknown}")
    return BudgetSettings(**cfg)


def _setup(model, lay, cell, tok, meta):
    idx, y = tok[:, :-1], tok[:, 1:]
    sel = ve.event_selector(lay, meta, len(idx)) if cell["rule"] == "linev" else None
    with torch.no_grad():
        h = model.backbone(idx, edge_mask=ve.edge_mask_for(cell, lay, idx.device))
    etas = [e.detach() for e in model.etas(float(cell["eta"]))]
    return idx, y, sel, h, etas


def clip_scale(norm: torch.Tensor, g_clip: float) -> torch.Tensor:
    """The per-sequence factor ``exp_ttt._clip_fast`` applies."""
    return torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))


def own_masks(lay: ve.Layout, meta: dict, idx: torch.Tensor, t: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, chunk) masks of the E writes (chunk 0) and C writes (chunk C) of type-``t`` locations."""
    cc = dg.c_chunk(lay)
    lo = cc * lay.chunk
    if int(meta["e_write"].max()) >= lay.chunk or int(meta["c_write"].min()) < lo \
            or int(meta["c_write"].max()) >= lo + lay.chunk:
        raise ValueError("E writes must lie in chunk 0 and C writes in chunk C")
    is_t = meta["c_type"] == t
    loc_c = idx.gather(1, meta["c_write"])
    loc_e = idx.gather(1, meta["e_write"])
    e_hit = ((loc_e[:, :, None] == loc_c[:, None, :]) & is_t[:, None, :]).any(-1)
    m_e = torch.zeros(idx.shape[0], lay.chunk, dtype=torch.bool, device=idx.device)
    m_e.scatter_(1, meta["e_write"], e_hit)
    m_c = torch.zeros_like(m_e)
    m_c.scatter_(1, meta["c_write"] - lo, is_t)
    return m_e, m_c


def masked_update(model, rule, h, x, y, sel, reference, W, etas, scale, c, chunk, mask):
    """``-eta * scale * grad`` of the chunk-``c`` inner loss restricted to ``mask``.

    The inner loss is the stream's (``inner.mean(1)`` over the chunk); the clip
    scale is the whole chunk's, so the masked updates of a partition of the
    chunk sum to the stream's update.
    """
    sl = slice(c * chunk, (c + 1) * chunk)
    Wg = [w.detach().requires_grad_(True) for w in W]
    with torch.enable_grad():
        logits = model.read(tt.fast_block(h[:, sl], model.head.ln, Wg,
                                          None if x is None else x[:, sl]))
        inner = tt.inner_token_loss(rule, logits, y[:, sl],
                                    sel=None if sel is None else sel[:, sl],
                                    reference=None if reference is None else reference[:, sl])
        grads = torch.autograd.grad((inner * mask).mean(1).sum(), Wg)
    return [(-eta * g * scale.view(-1, *([1] * (g.ndim - 1)))).detach()
            for eta, g in zip(etas, grads)]


def _margin(z: torch.Tensor, meta: dict, t: int) -> torch.Tensor:
    """``z[new] - z[old]`` at every type-``t`` query (A or X), flattened."""
    is_t = meta["query_type"] == t
    z1 = z.gather(-1, meta["query_tok"][..., None]).squeeze(-1)
    z0 = z.gather(-1, meta["query_old"].clamp_min(0)[..., None]).squeeze(-1)
    return (z1 - z0)[is_t]


def _norms(dW: list[torch.Tensor]) -> list[list[float]]:
    """Per-sequence Frobenius norm of each fast tensor, as lists."""
    return [d.reshape(d.shape[0], -1).norm(dim=1).tolist() for d in dW]


def budget_stream(model: tt.TTTModel, lay: ve.Layout, vst: ve.VeSettings, cell: dict,
                  tok: torch.Tensor, meta: dict) -> dict:
    """Base stream of one batch with the write budget and the probe trajectory."""
    idx, y, sel, h, etas = _setup(model, lay, cell, tok, meta)
    rule = ve.RULES[cell["rule"]]
    n_chunk = lay.T // lay.chunk
    cc = dg.c_chunk(lay)
    arrive: dict[int, list[torch.Tensor]] = {}
    dW: dict[int, list[torch.Tensor]] = {}
    norm_c: dict[int, torch.Tensor] = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        arrive[c] = [w.detach() for w in W]
        norm_c[c] = norm.detach()
        if c in (0, cc):
            dW[c] = [(-eta * g).detach() for eta, g in zip(etas, grads)]

    logits = tt.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, tt.GradStats(), sel,
                                h=h, rule=rule, observer=observer)
    zq = ve._gather_last(logits, meta["query_pos"])
    out: dict[str, Any] = dict(q=dg._queries(zq, meta))
    out["norm"] = {c: norm_c[c].tolist() for c in range(n_chunk)}
    out["scale"] = {c: clip_scale(norm_c[c], vst.g_clip).tolist() for c in range(n_chunk)}
    out["dW_norm"] = {c: _norms(dW[c]) for c in (0, cc)}

    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    reference = tt.reference_logits(model, h, idx, arrive[0]) if rule == "aice" else None
    own: dict[tuple[str, str], list[torch.Tensor]] = {}
    with torch.no_grad():
        p_y = logits.softmax(-1).gather(-1, y[..., None]).squeeze(-1)
    for t in OWN_TYPES:
        ti = ve.TYPES.index(t)
        m_e, m_c = own_masks(lay, meta, idx, ti)
        for name, c, m in (("E", 0, m_e), ("C", cc, m_c)):
            own[(t, name)] = masked_update(model, rule, h, x, y, sel, reference, arrive[c], etas,
                                           clip_scale(norm_c[c], vst.g_clip), c, lay.chunk, m)
            out[f"own_dW_norm_{name}_{t}"] = _norms(own[(t, name)])
            sl = slice(c * lay.chunk, (c + 1) * lay.chunk)
            out[f"resid_{name}_{t}"] = (1 - p_y[:, sl][m]).tolist()

    # probe the R query positions with the weights after each chunk k >= C
    pos = meta["query_pos"]
    hq = ve._gather_last(h, pos)
    xq = None if x is None else ve._gather_last(x, pos)
    qchunk = pos // lay.chunk
    removals = [("E", dW[0], OWN_TYPES), ("C", dW[cc], OWN_TYPES)]
    for t in OWN_TYPES:
        removals += [("Eown", own[(t, "E")], (t,)), ("Cown", own[(t, "C")], (t,))]
    is_a = meta["query_type"] == ve.A
    maxdiff = 0.0
    traj: dict[str, dict[int, list[float]]] = {}
    with torch.no_grad():
        for k in range(cc, n_chunk - 1):
            Wk = arrive[k + 1]
            z_full = model.read(tt.fast_block(hq, model.head.ln, Wk, xq))
            mine = qchunk == k + 1
            if bool(mine.any()):
                maxdiff = max(maxdiff, float((z_full - zq)[mine].abs().max()))
            full = {t: _margin(z_full, meta, ve.TYPES.index(t)) for t in OWN_TYPES}
            for t in OWN_TYPES:
                traj.setdefault(f"margin_{t}", {})[k] = full[t].tolist()
            hit = (z_full.argmax(-1) == meta["query_tok"]).float()
            traj.setdefault("acc_A", {})[k] = hit[is_a].tolist()
            for name, d, types in removals:
                z = model.read(tt.fast_block(hq, model.head.ln,
                                             [w - e for w, e in zip(Wk, d)], xq))
                for t in types:
                    share = full[t] - _margin(z, meta, ve.TYPES.index(t))
                    traj.setdefault(f"share_{name}_{t}", {})[k] = share.tolist()
    out["probe_maxdiff"] = maxdiff
    out["traj"] = traj
    return out


def sweep_stream(model: tt.TTTModel, lay: ve.Layout, vst: ve.VeSettings, cell: dict,
                 tok: torch.Tensor, meta: dict, cond: str, beta: float,
                 gen: torch.Generator) -> dict:
    """One batch with an A-write injection; per-type hits, A margins, clip scale at chunk C."""
    idx, y, sel, h, etas = _setup(model, lay, cell, tok, meta)
    cc = dg.c_chunk(lay)
    seen: dict[str, torch.Tensor] = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        if c == cc:
            seen["scale"] = clip_scale(norm.detach(), vst.g_clip)

    logits = tt.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, tt.GradStats(), sel,
                                h=h, rule=ve.RULES[cell["rule"]],
                                inner_transform=dg.injection(cond, lay, meta, y, beta, gen),
                                observer=observer)
    zq = ve._gather_last(logits, meta["query_pos"])
    return dict(q=dg._queries(zq, meta), margin=dg._margin_A(zq, meta).tolist(),
                scale_C=seen["scale"].tolist())


# --------------------------------------------------------------------------- pooling


def _med(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def _cat(batches: list[dict], key: str) -> list[float]:
    return sum((b[key] for b in batches), [])


def summarize_budget(batches: list[dict], n_chunk: int, cc: int) -> dict:
    out: dict[str, Any] = {}
    qs = [b["q"] for b in batches]
    tot = {k: sum(q[k] for q in qs) for k in qs[0]}
    for t in ve.TYPES:
        out[f"base.acc_{t}"] = tot[f"hit_{t}"] / (tot[f"n_{t}"] or 1.0)
    for c in range(n_chunk):
        norms = sum((b["norm"][c] for b in batches), [])
        scales = sum((b["scale"][c] for b in batches), [])
        out[f"norm_c{c}"] = _med(norms)
        out[f"clip_frac_c{c}"] = float(np.mean([s < 1 for s in scales]))
        out[f"clip_scale_c{c}"] = float(np.mean(scales))
    out["clip_frac_E"], out["clip_frac_C"] = out["clip_frac_c0"], out[f"clip_frac_c{cc}"]
    for c, tag in ((0, "E"), (cc, "C")):
        for i, key in enumerate(tt.FAST_KEYS):
            out[f"dW_{tag}_{key}"] = _med(sum((b["dW_norm"][c][i] for b in batches), []))
    for t in OWN_TYPES:
        for tag in ("E", "C"):
            for i, key in enumerate(tt.FAST_KEYS):
                out[f"own_dW_{tag}_{t}_{key}"] = _med(
                    sum((b[f"own_dW_norm_{tag}_{t}"][i] for b in batches), []))
            out[f"resid_{tag}_{t}"] = _med(_cat(batches, f"resid_{tag}_{t}"))
    out["probe_maxdiff"] = max(b["probe_maxdiff"] for b in batches)
    for key in batches[0]["traj"]:
        for k in batches[0]["traj"][key]:
            vals = sum((b["traj"][key][k] for b in batches), [])
            out[f"k{k}.{key}"] = float(np.mean(vals)) if key == "acc_A" else _med(vals)
    return out


def summarize_sweep(parts: dict[tuple[str, float], list[dict]]) -> dict:
    out: dict[str, Any] = {}
    for (cond, beta), batches in parts.items():
        tag = f"{cond}@{beta:g}"
        qs = [b["q"] for b in batches]
        tot = {k: sum(q[k] for q in qs) for k in qs[0]}
        for t in ve.TYPES:
            out[f"{tag}.acc_{t}"] = tot[f"hit_{t}"] / (tot[f"n_{t}"] or 1.0)
        out[f"{tag}.stale_A"] = tot["stale_A"] / (tot["n_A"] or 1.0)
        out[f"{tag}.margin_A"] = _med(_cat(batches, "margin"))
        out[f"{tag}.scale_C"] = float(np.mean(_cat(batches, "scale_C")))
    return out


def sweep_conditions(betas: list[float]) -> list[tuple[str, float]]:
    conds = [(c, float(b)) for b in betas for c in ("erase_v0", "add_v1")]
    return conds + [("erase_rand", S2_BETA)]


def run_cell(cell: dict, st: BudgetSettings, vst: ve.VeSettings, lay: ve.Layout,
             dev: torch.device) -> dict:
    t0 = time.time()
    model = dg.load_model(vst, lay, cell, st.ckpt_dir, dev)
    data = ve.fixed_set(lay, st.n_eval, ve.EVAL_SEED, dev, st.eval_batch)
    n_chunk = lay.T // lay.chunk
    rec: dict[str, Any] = dict(cell_id=ve.cell_id(cell), **cell)
    rec.update(summarize_budget([budget_stream(model, lay, vst, cell, tok, meta)
                                 for tok, meta in data], n_chunk, dg.c_chunk(lay)))
    rec["swept"] = rec["base.acc_A"] < st.sweep_below
    if rec["swept"]:
        parts = {}
        for cond, beta in sweep_conditions(st.betas):
            # same generator seeding as E1-VE-D, so erase_rand draws the same third values
            gen = torch.Generator().manual_seed(dg.RAND_SEED + int(cell["seed"]))
            parts[(cond, beta)] = [sweep_stream(model, lay, vst, cell, tok, meta, cond, beta, gen)
                                   for tok, meta in data]
        rec.update(summarize_sweep(parts))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- scoring


def gate_r0(rec: dict, diag: dict[str, dict]) -> tuple[bool | None, float | None]:
    """beta = .5 reproduces the E1-VE-D erase_v0 / add_v1 accuracies and median margins."""
    r = diag.get(rec["cell_id"])
    if r is None or not rec.get("swept"):
        return None, None
    worst = 0.0
    ok = True
    for cond in ("erase_v0", "add_v1"):
        for t in ve.TYPES:
            d = abs(rec[f"{cond}@0.5.acc_{t}"] - float(r[f"{cond}.acc_{t}"]))
            ok &= d <= R0_ACC_TOL
            worst = max(worst, d)
        d = abs(rec[f"{cond}@0.5.margin_A"] - float(r[f"{cond}.margin_A"]))
        ok &= d <= R0_MARGIN_TOL
        worst = max(worst, d)
    return ok, worst


def beta90(rec: dict, betas: list[float], cond: str = "add_v1") -> float:
    """Smallest beta where acc_A reaches .90, linear between grid points from beta 0 (base)."""
    xs = [0.0] + [float(b) for b in betas]
    ys = [rec["base.acc_A"]] + [rec[f"{cond}@{b:g}.acc_A"] for b in betas]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= .90:
            return x0
        if y1 >= .90:
            return x0 + (x1 - x0) * (.90 - y0) / (y1 - y0)
    return math.inf


def _ranks(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    order = a.argsort(kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    for val in np.unique(a):                      # average ranks over ties (inf included)
        same = a == val
        r[same] = r[same].mean()
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def score(recs: list[dict], betas: list[float], diag: dict[str, dict]) -> dict:
    """Registered S1-S3 and B1 over FO-CE-lesion cells passing G0, V3 and R0."""
    out: dict[str, Any] = {}
    les_all = []
    for eta in sorted({float(r["eta"]) for r in recs}):
        les = [r for r in recs if r["rule"] == "ce" and r.get("edge") and float(r["eta"]) == eta
               and r.get("g0") is not False and r.get("r0") is not False and r.get("swept")
               and r["base.acc_U"] >= .90 and r["base.acc_S"] >= .90]
        les_all += les
        k = len(les)
        tag = f"eta{eta:g}"
        s1 = sum(all(abs(r[f"erase_v0@{b:g}.acc_A"] - r[f"add_v1@{b:g}.acc_A"]) <= .03
                     for b in betas) for r in les)
        s2 = sum(r[f"add_v1@{S2_BETA:g}.acc_A"] >= .90
                 and all(r[f"add_v1@{S2_BETA:g}.acc_{t}"] >= .90 for t in ("U", "S", "X", "N"))
                 for r in les)
        ctrl = sum(r[f"erase_rand@{S2_BETA:g}.acc_A"] >= .90 for r in les)
        b1 = sum(r["clip_frac_E"] <= .05 and r["clip_frac_C"] <= .05 for r in les)
        out[f"S1_{tag}"] = dict(verdict=dg.count_verdict(s1, k), n=s1, k=k)
        out[f"S2_{tag}"] = dict(verdict=dg.count_verdict(s2, k), n=s2, k=k)
        out[f"S2c_{tag}"] = dict(verdict=dg.count_verdict(k - ctrl, k), rescued=ctrl, k=k)
        out[f"B1_{tag}"] = dict(verdict=dg.count_verdict(b1, k), n=b1, k=k)
    ratio, b90 = [], []
    for r in les_all:
        d = diag.get(r["cell_id"])
        if d is None:
            continue
        ratio.append(-float(d["contrib_E_A"]) / float(d["contrib_C_A"]))
        b90.append(beta90(r, betas))
    rho = spearman(ratio, b90) if len(ratio) >= 4 else float("nan")
    verdict = "NE" if len(ratio) < 8 else ("S" if rho >= .6 else "F" if rho <= .2 else "I")
    out["S3"] = dict(verdict=verdict, rho=rho, k=len(ratio),
                     pairs=[[round(a, 3), b] for a, b in zip(ratio, b90)])
    return out


def run(cfg: dict, seed: int) -> dict:
    st = settings(cfg)
    tt._mark_process(st.process_name)
    torch.set_num_threads(max(1, int(st.threads)))
    dev = tt._device(st.device)
    tt.configure_tf32(False, st.process_name)
    vst = ve.VeSettings(**{**st.ve, "n_eval": st.n_eval, "eval_batch": st.eval_batch,
                           "device": st.device})
    lay = vst.layout()
    if not st.cells:
        raise ValueError("no cells")
    for cell in st.cells:
        ve.check_cell(cell)
    if S2_BETA not in [float(b) for b in st.betas] or 0.5 not in [float(b) for b in st.betas]:
        raise ValueError(f"betas must include 0.5 and {S2_BETA}")
    ref = dg.load_reference(st.reference_csv)
    diag = dg.load_reference(st.diag_csv)
    print("VE_BUDGET_CONFIG " + json.dumps(dict(dataclasses.asdict(st), cells=len(st.cells),
                                               device_resolved=str(dev))), flush=True)
    recs = []
    for i, cell in enumerate(st.cells):
        rec = run_cell(cell, st, vst, lay, dev)
        rec["g0"], rec["g0_maxdiff"] = dg.gate_g0(rec, ref)
        rec["r0"], rec["r0_maxdiff"] = gate_r0(rec, diag)
        recs.append(rec)
        print("CELL_BUDGET " + json.dumps(rec), flush=True)
        line = (f"  [{i + 1}/{len(st.cells)}] {rec['cell_id']} g0={rec['g0']} r0={rec['r0']} "
                f"A={rec['base.acc_A']:.3f} clipE/C={rec['clip_frac_E']:.3f}/"
                f"{rec['clip_frac_C']:.3f} probe={rec['probe_maxdiff']:.1e}")
        if rec["swept"]:
            line += " add_v1 A@" + "/".join(f"{rec[f'add_v1@{b:g}.acc_A']:.2f}" for b in st.betas)
        print(line + f" {rec['wall_s']}s", flush=True)
    bad_g0 = [r["cell_id"] for r in recs if r["g0"] is False]
    bad_r0 = [r["cell_id"] for r in recs if r["r0"] is False]
    print(f"G0 reproduction: {len(recs) - len(bad_g0)}/{len(recs)} within {dg.G0_TOL}; "
          f"failing: {bad_g0 or 'none'}", flush=True)
    n_r0 = sum(r["r0"] is not None for r in recs)
    print(f"R0 beta=.5 reproduction: {n_r0 - len(bad_r0)}/{n_r0}; failing: {bad_r0 or 'none'}; "
          f"max diff {max((r['r0_maxdiff'] or 0) for r in recs):.2e}", flush=True)
    print(f"probe max diff {max(r['probe_maxdiff'] for r in recs):.2e}", flush=True)
    res = score(recs, [float(b) for b in st.betas], diag)
    for key, val in res.items():
        print(f"{key}: {json.dumps(val)}", flush=True)
    print("VE_BUDGET_SUMMARY " + json.dumps(dict(cells=len(recs), g0_fail=bad_g0, r0_fail=bad_r0,
                                                scores=res)), flush=True)
    return dict(cells=len(recs), g0_fail=len(bad_g0), r0_fail=len(bad_r0), scores=res)
